import os
import json
import torch
import torch.nn.functional as F


def _safe_top2_token_id(prob_row, top1_token_id):
    top_k = min(2, prob_row.shape[-1])
    top_ids = torch.topk(prob_row, k=top_k).indices
    for token_id in top_ids.tolist():
        if int(token_id) != int(top1_token_id):
            return int(token_id)
    return int(top1_token_id)


def _run_helper_top2_perturbation_probes(
    *,
    model,
    x,
    x0,
    p_curr,
    batch_idx,
    candidate_idx,
    helper_positions,
    full_probe_prob,
    target_token,
):
    if len(helper_positions) == 0:
        return {
            "helper_top2_perturbation_count": 0,
            "helper_top2_mean_drop": 0.0,
            "helper_top2_max_drop": 0.0,
            "helper_top2_min_drop": 0.0,
            "helper_top2_drop_std_proxy": 0.0,
            "helper_top2_sensitive_count": 0,
            "helper_top2_any_sensitive": False,
            "helper_top2_details": [],
            "mean_perturbed_rise": 0.0,
            "min_perturbed_rise": 0.0,
            "max_perturbed_rise": 0.0,
            "positive_perturbed_rise_count": 0,
            "positive_perturbed_rise_rate": 0.0,
            "extra_probe_forwards": 0,
        }

    perturbation_details = []
    drops = []
    perturbed_rises = []
    extra_probe_forwards = 0

    for helper_pos in helper_positions.tolist():
        helper_top1 = int(x0[batch_idx, helper_pos].item())
        helper_top2 = _safe_top2_token_id(p_curr[batch_idx, helper_pos], helper_top1)
        if helper_top2 == helper_top1:
            perturbed_probe_prob = float(full_probe_prob.item())
            drop = 0.0
        else:
            x_perturbed = x.clone()
            x_perturbed[batch_idx, helper_positions] = x0[batch_idx, helper_positions]
            x_perturbed[batch_idx, helper_pos] = helper_top2
            perturbed_logits = model(x_perturbed).logits
            perturbed_probs = F.softmax(perturbed_logits.to(torch.float64), dim=-1)
            perturbed_probe_prob = float(
                torch.gather(
                    perturbed_probs[batch_idx, candidate_idx].view(1, -1),
                    dim=-1,
                    index=target_token,
                ).squeeze().item()
            )
            drop = float(full_probe_prob.item()) - perturbed_probe_prob
            extra_probe_forwards += 1

        helper_top1_prob = float(p_curr[batch_idx, helper_pos, helper_top1].item())
        helper_top2_prob = float(p_curr[batch_idx, helper_pos, helper_top2].item())
        perturbed_rise = perturbed_probe_prob - float(p_curr[batch_idx, candidate_idx, int(target_token.item())].item())
        perturbation_details.append(
            {
                "helper_position": int(helper_pos),
                "helper_top1_token_id": helper_top1,
                "helper_top2_token_id": helper_top2,
                "helper_top1_prob": round(helper_top1_prob, 6),
                "helper_top2_prob": round(helper_top2_prob, 6),
                "helper_top1_top2_margin": round(helper_top1_prob - helper_top2_prob, 6),
                "perturbed_probe_prob": round(perturbed_probe_prob, 6),
                "probe_prob_drop": round(drop, 6),
                "perturbed_rise": round(perturbed_rise, 6),
                "sensitive": bool(drop > 0.05),
            }
        )
        drops.append(drop)
        perturbed_rises.append(perturbed_rise)

    mean_drop = sum(drops) / len(drops)
    max_drop = max(drops)
    min_drop = min(drops)
    drop_std_proxy = max_drop - min_drop
    sensitive_count = sum(1 for drop in drops if drop > 0.05)
    mean_perturbed_rise = sum(perturbed_rises) / len(perturbed_rises)
    min_perturbed_rise = min(perturbed_rises)
    max_perturbed_rise = max(perturbed_rises)
    positive_perturbed_rise_count = sum(1 for rise in perturbed_rises if rise > 0)

    return {
        "helper_top2_perturbation_count": len(perturbation_details),
        "helper_top2_mean_drop": round(mean_drop, 6),
        "helper_top2_max_drop": round(max_drop, 6),
        "helper_top2_min_drop": round(min_drop, 6),
        "helper_top2_drop_std_proxy": round(drop_std_proxy, 6),
        "helper_top2_sensitive_count": int(sensitive_count),
        "helper_top2_any_sensitive": bool(sensitive_count > 0),
        "helper_top2_details": perturbation_details,
        "mean_perturbed_rise": round(mean_perturbed_rise, 6),
        "min_perturbed_rise": round(min_perturbed_rise, 6),
        "max_perturbed_rise": round(max_perturbed_rise, 6),
        "positive_perturbed_rise_count": int(positive_perturbed_rise_count),
        "positive_perturbed_rise_rate": round(positive_perturbed_rise_count / len(perturbed_rises), 6),
        "extra_probe_forwards": extra_probe_forwards,
    }


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = (
        torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64)
        + base
    )
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


def _to_int_list(indices):
    return [int(x) for x in indices]


def _select_before_high_quality_helpers(
    batch_idx,
    candidate_idx,
    masked_indices,
    confidence,
    margin,
    helper_k,
    target_conditioned_x=None,
    x0=None,
    model=None,
):
    empty_debug = {
        "candidate_idx": int(candidate_idx),
        "helper_candidates_considered": [],
        "selected_helpers": [],
        "selection_mode": "strict_target_conditioned_per_helper",
    }
    if helper_k <= 0:
        return torch.empty(0, dtype=torch.long, device=masked_indices.device), empty_debug

    helper_candidates = masked_indices[masked_indices < candidate_idx]
    helper_candidates = helper_candidates[helper_candidates != candidate_idx]
    if len(helper_candidates) == 0:
        return torch.empty(0, dtype=torch.long, device=masked_indices.device), empty_debug

    distances = torch.abs(helper_candidates - candidate_idx)
    near_limit = min(len(helper_candidates), max(helper_k * 4, helper_k))
    _, near_rel = torch.topk(-distances.to(torch.float64), k=near_limit)
    nearby_candidates = helper_candidates[near_rel]

    helper_scores = confidence[nearby_candidates] * margin[nearby_candidates]
    conditioned_scores = None
    if target_conditioned_x is not None and x0 is not None and model is not None:
        target_conditioned_logits = model(target_conditioned_x).logits
        target_conditioned_probs = F.softmax(target_conditioned_logits.to(torch.float64), dim=-1)
        helper_targets = x0[batch_idx, nearby_candidates].unsqueeze(-1)
        conditioned_scores = torch.gather(
            target_conditioned_probs[batch_idx, nearby_candidates],
            dim=-1,
            index=helper_targets,
        ).squeeze(-1)
        helper_scores = helper_scores * conditioned_scores.to(helper_scores.dtype)

    keep_k = min(helper_k, len(nearby_candidates))
    _, helper_rel = torch.topk(helper_scores, k=keep_k)
    selected_helpers = nearby_candidates[helper_rel]
    selected_helper_set = set(int(x) for x in selected_helpers.tolist())

    helper_candidates_considered = []
    for idx_in_list, helper_idx in enumerate(nearby_candidates.tolist()):
        item = {
            "position": int(helper_idx),
            "distance": int(abs(helper_idx - int(candidate_idx))),
            "base_confidence": round(float(confidence[helper_idx].item()), 6),
            "base_margin": round(float(margin[helper_idx].item()), 6),
            "base_score": round(float((confidence[helper_idx] * margin[helper_idx]).item()), 6),
            "final_score": round(float(helper_scores[idx_in_list].item()), 6),
            "selected": bool(int(helper_idx) in selected_helper_set),
        }
        if conditioned_scores is not None:
            item["target_conditioned_support"] = round(float(conditioned_scores[idx_in_list].item()), 6)
        helper_candidates_considered.append(item)

    return selected_helpers, {
        "candidate_idx": int(candidate_idx),
        "helper_candidates_considered": helper_candidates_considered,
        "selected_helpers": [int(x) for x in selected_helpers.tolist()],
        "selection_mode": "strict_target_conditioned_per_helper",
    }


def _target_conditioned_local_commit_risk_hard_gate(
    x,
    p_curr,
    x0,
    mask_index,
    confidence,
    reveal_budget_per_batch,
    helper_k,
    risk_threshold,
    min_confidence,
    candidate_multiplier,
    model,
    force_accept_all=False,
    collect_probe_details=False,
    collect_step_debug=False,
    collect_helper_top2_perturbation=False,
    selection_strategy="hard_gate",
):
    batch_size = mask_index.shape[0]
    ready_mask = torch.zeros_like(mask_index, dtype=torch.bool)
    debug_info = []

    sorted_probs, _ = torch.sort(p_curr, dim=-1, descending=True)
    top1_prob = sorted_probs[..., 0]
    top2_prob = sorted_probs[..., 1]
    margin = top1_prob - top2_prob

    step_debug_enabled = collect_step_debug

    for j in range(batch_size):
        masked_indices = torch.where(mask_index[j])[0]
        reveal_budget = int(reveal_budget_per_batch[j])
        if len(masked_indices) == 0 or reveal_budget <= 0:
            debug_info.append(
                {
                    "reveal_budget": int(reveal_budget),
                    "candidate_pool": [],
                    "initial_reveal_candidates": [],
                    "confidence_filtered_positions": [],
                    "gated_off_positions": [],
                    "accepted_positions": [],
                    "fallback_triggered": False,
                    "fallback_position": None,
                    "selected_positions": [],
                    "gate_details": [],
                    "gate_rejected_count": 0,
                    "extra_probe_forwards": 0,
                }
            )
            continue

        candidate_pool_size = min(
            len(masked_indices), max(reveal_budget, candidate_multiplier * reveal_budget)
        )
        _, pool_rel = torch.topk(confidence[j, masked_indices], k=candidate_pool_size)
        candidate_pool = masked_indices[pool_rel]
        confidence_filtered_positions = [
            int(idx) for idx in candidate_pool if float(confidence[j, idx].item()) < min_confidence
        ]
        initial_reveal = torch.tensor(
            [
                int(idx)
                for idx in candidate_pool
                if float(confidence[j, idx].item()) >= min_confidence
            ],
            device=masked_indices.device,
            dtype=torch.long,
        )

        accepted_positions = []
        accepted_set = set()
        gated_off_positions = []
        gate_details = [] if step_debug_enabled else None
        candidate_ordering_details = [] if step_debug_enabled else None
        probe_forwards = 0
        candidate_eval_count = 0
        positive_helper_rise_count = 0
        negative_helper_rise_count = 0
        accepted_count = 0
        rejected_count = 0
        top1_changed_count = 0
        helper_count_total = 0
        commit_risk_total = 0.0
        rise_total = 0.0
        helper_top2_mean_drop_total = 0.0
        helper_top2_max_drop_total = 0.0
        helper_top2_sensitive_total = 0
        helper_top2_event_count = 0
        mean_perturbed_rise_total = 0.0
        positive_perturbed_rise_rate_total = 0.0
        self_rise_total = 0.0
        helper_rise_total = 0.0
        self_stronger_than_helper_count = 0
        self_positive_helper_negative_count = 0

        for candidate_idx in initial_reveal:
            target_conditioned_x = x.clone()
            target_conditioned_x[j, candidate_idx] = x0[j, candidate_idx]
            helper_positions, helper_candidate_debug = _select_before_high_quality_helpers(
                batch_idx=j,
                candidate_idx=candidate_idx,
                masked_indices=masked_indices,
                confidence=confidence[j],
                margin=margin[j],
                helper_k=helper_k,
                target_conditioned_x=target_conditioned_x,
                x0=x0,
                model=model,
            )
            if not step_debug_enabled:
                helper_candidate_debug = None

            x_probe = x.clone()
            if len(helper_positions) > 0:
                x_probe[j, helper_positions] = x0[j, helper_positions]

            probe_logits = model(x_probe).logits
            probe_probs = F.softmax(probe_logits.to(torch.float64), dim=-1)
            probe_forwards += 1

            target_token = x0[j, candidate_idx].view(1, 1)
            original_prob = torch.gather(
                p_curr[j, candidate_idx].view(1, -1), dim=-1, index=target_token
            ).squeeze()
            probe_prob = torch.gather(
                probe_probs[j, candidate_idx].view(1, -1), dim=-1, index=target_token
            ).squeeze()
            stability_gain = float((probe_prob - original_prob).item())
            commit_risk = float((original_prob - probe_prob).item())
            accepted = True if force_accept_all else (commit_risk <= risk_threshold)
            candidate_eval_count += 1
            helper_count_total += len(helper_positions)
            commit_risk_total += commit_risk
            rise_total += stability_gain
            if stability_gain > 0:
                positive_helper_rise_count += 1
            elif stability_gain < 0:
                negative_helper_rise_count += 1
            if accepted:
                accepted_count += 1
            else:
                rejected_count += 1
            if collect_probe_details or step_debug_enabled:
                candidate_changed_top1 = int(torch.argmax(probe_probs[j, candidate_idx]).item()) != int(
                    x0[j, candidate_idx].item()
                )
                candidate_rank = int(
                    (probe_probs[j, candidate_idx] > probe_prob).sum().item()
                ) + 1
                rise_per_helper = float(stability_gain / len(helper_positions)) if len(helper_positions) > 0 else 0.0
                if candidate_changed_top1:
                    top1_changed_count += 1
            else:
                candidate_changed_top1 = None
                candidate_rank = None
                rise_per_helper = None
            self_probe_prob = None
            helper_probe_prob = probe_prob
            helper_top2_perturbation = None

            if collect_helper_top2_perturbation:
                helper_top2_perturbation = _run_helper_top2_perturbation_probes(
                    model=model,
                    x=x,
                    x0=x0,
                    p_curr=p_curr,
                    batch_idx=j,
                    candidate_idx=candidate_idx,
                    helper_positions=helper_positions,
                    full_probe_prob=probe_prob,
                    target_token=target_token,
                )
                probe_forwards += helper_top2_perturbation["extra_probe_forwards"]
                helper_top2_mean_drop_total += helper_top2_perturbation["helper_top2_mean_drop"]
                helper_top2_max_drop_total += helper_top2_perturbation["helper_top2_max_drop"]
                helper_top2_sensitive_total += helper_top2_perturbation["helper_top2_sensitive_count"]
                helper_top2_event_count += 1
                mean_perturbed_rise_total += helper_top2_perturbation["mean_perturbed_rise"]
                positive_perturbed_rise_rate_total += helper_top2_perturbation["positive_perturbed_rise_rate"]

            if collect_probe_details:
                x_self_probe = x.clone()
                x_self_probe[j, candidate_idx] = x0[j, candidate_idx]
                self_probe_logits = model(x_self_probe).logits
                self_probe_probs = F.softmax(self_probe_logits.to(torch.float64), dim=-1)
                self_probe_prob = torch.gather(
                    self_probe_probs[j, candidate_idx].view(1, -1), dim=-1, index=target_token
                ).squeeze()
                probe_forwards += 1
                self_rise_value = float((self_probe_prob - original_prob).item())
                helper_rise_value = float((helper_probe_prob - original_prob).item())
                self_rise_total += self_rise_value
                helper_rise_total += helper_rise_value
                if self_rise_value > helper_rise_value:
                    self_stronger_than_helper_count += 1
                if self_rise_value > 0 and helper_rise_value < 0:
                    self_positive_helper_negative_count += 1

            if step_debug_enabled:
                gate_detail = {
                    "position": int(candidate_idx),
                    "pred_token_id": int(x0[j, candidate_idx].item()),
                    "confidence": round(float(confidence[j, candidate_idx].item()), 6),
                    "margin": round(float(margin[j, candidate_idx].item()), 6),
                    "helper_positions": _to_int_list(helper_positions),
                    "helper_selection_debug": helper_candidate_debug,
                    "original_prob": round(float(original_prob.item()), 6),
                    "probe_prob": round(float(probe_prob.item()), 6),
                    "helper_masked_probe_prob": round(float(probe_prob.item()), 6),
                    "rise": round(stability_gain, 6),
                    "helper_masked_rise": round(stability_gain, 6),
                    "rise_per_helper": round(rise_per_helper, 6),
                    "candidate_rank_after_helper_masked_probe": candidate_rank,
                    "candidate_changed_top1_after_helper_masked_probe": bool(candidate_changed_top1),
                    "helper_selection_mode": "strict_target_conditioned_per_helper",
                    "commit_risk": round(commit_risk, 6),
                    "accepted": bool(accepted),
                }
                if collect_probe_details:
                    gate_detail["self_probe_prob"] = round(float(self_probe_prob.item()), 6)
                    gate_detail["self_rise"] = round(self_rise_value, 6)
                    gate_detail["helper_probe_prob"] = round(float(helper_probe_prob.item()), 6)
                    gate_detail["helper_rise"] = round(helper_rise_value, 6)
                if collect_helper_top2_perturbation and helper_top2_perturbation is not None:
                    gate_detail["helper_top2_mean_drop"] = helper_top2_perturbation["helper_top2_mean_drop"]
                    gate_detail["helper_top2_max_drop"] = helper_top2_perturbation["helper_top2_max_drop"]
                    gate_detail["helper_top2_min_drop"] = helper_top2_perturbation["helper_top2_min_drop"]
                    gate_detail["helper_top2_drop_std_proxy"] = helper_top2_perturbation["helper_top2_drop_std_proxy"]
                    gate_detail["helper_top2_sensitive_count"] = helper_top2_perturbation["helper_top2_sensitive_count"]
                    gate_detail["helper_top2_any_sensitive"] = helper_top2_perturbation["helper_top2_any_sensitive"]
                    gate_detail["mean_perturbed_rise"] = helper_top2_perturbation["mean_perturbed_rise"]
                    gate_detail["min_perturbed_rise"] = helper_top2_perturbation["min_perturbed_rise"]
                    gate_detail["max_perturbed_rise"] = helper_top2_perturbation["max_perturbed_rise"]
                    gate_detail["positive_perturbed_rise_count"] = helper_top2_perturbation["positive_perturbed_rise_count"]
                    gate_detail["positive_perturbed_rise_rate"] = helper_top2_perturbation["positive_perturbed_rise_rate"]
                    gate_detail["helper_top2_details"] = helper_top2_perturbation["helper_top2_details"]

                if candidate_ordering_details is not None:
                    ordering_detail = dict(gate_detail)
                    min_probe_support = float(probe_prob.item())
                    identity_consistent = not bool(candidate_changed_top1)
                    if collect_helper_top2_perturbation and helper_top2_perturbation is not None:
                        min_probe_support = float(probe_prob.item()) - float(
                            helper_top2_perturbation["helper_top2_max_drop"]
                        )
                        identity_consistent = identity_consistent and not bool(
                            helper_top2_perturbation["helper_top2_any_sensitive"]
                        )
                    ordering_detail["min_probe_support"] = round(max(min_probe_support, 0.0), 6)
                    ordering_detail["identity_consistent_under_local_perturbation"] = bool(identity_consistent)
                    candidate_ordering_details.append(ordering_detail)

                gate_details.append(gate_detail)

            if accepted:
                accepted_positions.append(int(candidate_idx))
                accepted_set.add(int(candidate_idx))
            else:
                gated_off_positions.append(int(candidate_idx))

        fallback_triggered = False
        fallback_position = None
        selected_positions = list(accepted_positions)
        selected_from_gate = []
        selected_via_backfill = []

        if selection_strategy == "confidence_only":
            selected_positions = _to_int_list(initial_reveal[:reveal_budget])
        elif selection_strategy == "future_support":
            ranking_source = candidate_ordering_details if candidate_ordering_details is not None else []
            ranking_source = sorted(
                ranking_source,
                key=lambda item: (
                    item.get("min_probe_support", 0.0),
                    item.get("probe_prob", 0.0),
                    item.get("confidence", 0.0),
                ),
                reverse=True,
            )
            selected_positions = [int(item["position"]) for item in ranking_source[:reveal_budget]]
        else:
            if len(selected_positions) == 0:
                fallback_triggered = True
                if len(initial_reveal) > 0:
                    fallback_position = int(initial_reveal[0])
                else:
                    fallback_position = int(candidate_pool[0])
                selected_positions = [fallback_position]
            selected_positions = selected_positions[:reveal_budget]

        if selection_strategy != "hard_gate" and len(selected_positions) < reveal_budget:
            backfill_pool = [int(idx) for idx in candidate_pool.tolist() if int(idx) not in set(selected_positions)]
            for idx in backfill_pool:
                if len(selected_positions) >= reveal_budget:
                    break
                selected_positions.append(idx)
                selected_via_backfill.append(idx)

        selected_from_gate = [idx for idx in selected_positions if idx in accepted_positions]

        ready_mask[j, torch.tensor(selected_positions, device=mask_index.device, dtype=torch.long)] = True

        signal_summary = {
            "candidate_eval_count": int(candidate_eval_count),
            "accepted_count": int(accepted_count),
            "rejected_count": int(rejected_count),
            "positive_helper_rise_count": int(positive_helper_rise_count),
            "negative_helper_rise_count": int(negative_helper_rise_count),
            "avg_commit_risk": round(commit_risk_total / candidate_eval_count, 6)
            if candidate_eval_count > 0
            else 0.0,
            "avg_helper_masked_rise": round(rise_total / candidate_eval_count, 6)
            if candidate_eval_count > 0
            else 0.0,
            "avg_helper_count": round(helper_count_total / candidate_eval_count, 6)
            if candidate_eval_count > 0
            else 0.0,
        }
        if collect_helper_top2_perturbation:
            signal_summary["avg_helper_top2_mean_drop"] = round(
                helper_top2_mean_drop_total / helper_top2_event_count, 6
            ) if helper_top2_event_count > 0 else 0.0
            signal_summary["avg_helper_top2_max_drop"] = round(
                helper_top2_max_drop_total / helper_top2_event_count, 6
            ) if helper_top2_event_count > 0 else 0.0
            signal_summary["helper_top2_sensitive_count"] = int(helper_top2_sensitive_total)
            signal_summary["helper_top2_event_count"] = int(helper_top2_event_count)
            signal_summary["avg_mean_perturbed_rise"] = round(
                mean_perturbed_rise_total / helper_top2_event_count, 6
            ) if helper_top2_event_count > 0 else 0.0
            signal_summary["avg_positive_perturbed_rise_rate"] = round(
                positive_perturbed_rise_rate_total / helper_top2_event_count, 6
            ) if helper_top2_event_count > 0 else 0.0
        if collect_probe_details:
            signal_summary["avg_self_rise"] = round(self_rise_total / candidate_eval_count, 6) if candidate_eval_count > 0 else 0.0
            signal_summary["avg_helper_rise"] = round(helper_rise_total / candidate_eval_count, 6) if candidate_eval_count > 0 else 0.0
            signal_summary["self_stronger_than_helper_count"] = int(self_stronger_than_helper_count)
            signal_summary["self_positive_helper_negative_count"] = int(self_positive_helper_negative_count)
        if collect_probe_details or step_debug_enabled:
            signal_summary["top1_changed_count"] = int(top1_changed_count)

        if step_debug_enabled:
            debug_info.append(
                {
                    "reveal_budget": int(reveal_budget),
                    "candidate_pool": [
                        {"position": int(idx), "confidence": round(float(confidence[j, idx].item()), 6)}
                        for idx in candidate_pool
                    ],
                    "initial_reveal_candidates": _to_int_list(initial_reveal),
                    "confidence_filtered_positions": confidence_filtered_positions,
                    "gated_off_positions": gated_off_positions,
                    "accepted_positions": accepted_positions,
                    "fallback_triggered": fallback_triggered,
                    "fallback_position": fallback_position,
                    "fallback_from_accepted_pool": bool(fallback_position in accepted_positions) if fallback_position is not None else False,
                    "selected_positions": selected_positions,
                    "selected_from_gate": selected_from_gate,
                    "selected_via_backfill": selected_via_backfill,
                    "candidate_ordering_details": candidate_ordering_details or [],
                    "selection_strategy": selection_strategy,
                    "gate_details": gate_details,
                    "signal_summary": signal_summary,
                    "gate_rejected_count": len(gated_off_positions),
                    "acceptance_rate_among_initial_candidates": round(
                        len(accepted_positions) / len(initial_reveal), 6
                    ) if len(initial_reveal) > 0 else 0.0,
                    "extra_probe_forwards": probe_forwards,
                }
            )
        else:
            debug_info.append(
                {
                    "reveal_budget": int(reveal_budget),
                    "selected_positions": selected_positions,
                    "selected_from_gate": selected_from_gate,
                    "selected_via_backfill": selected_via_backfill,
                    "selection_strategy": selection_strategy,
                    "signal_summary": signal_summary,
                    "gate_rejected_count": len(gated_off_positions),
                    "fallback_triggered": fallback_triggered,
                    "acceptance_rate_among_initial_candidates": round(
                        len(accepted_positions) / len(initial_reveal), 6
                    ) if len(initial_reveal) > 0 else 0.0,
                    "extra_probe_forwards": probe_forwards,
                }
            )

    return ready_mask, debug_info


def _merge_signal_summaries(signal_summaries):
    total_candidate_evals = sum(s.get("candidate_eval_count", 0) for s in signal_summaries)
    total_accepted = sum(s.get("accepted_count", 0) for s in signal_summaries)
    total_rejected = sum(s.get("rejected_count", 0) for s in signal_summaries)
    total_positive_helper_rise = sum(s.get("positive_helper_rise_count", 0) for s in signal_summaries)
    total_negative_helper_rise = sum(s.get("negative_helper_rise_count", 0) for s in signal_summaries)
    total_top1_changed = sum(s.get("top1_changed_count", 0) for s in signal_summaries)
    total_self_stronger_than_helper = sum(s.get("self_stronger_than_helper_count", 0) for s in signal_summaries)
    total_self_positive_helper_negative = sum(s.get("self_positive_helper_negative_count", 0) for s in signal_summaries)
    commit_risk_sum = sum(s.get("avg_commit_risk", 0.0) * s.get("candidate_eval_count", 0) for s in signal_summaries)
    helper_masked_rise_sum = sum(s.get("avg_helper_masked_rise", 0.0) * s.get("candidate_eval_count", 0) for s in signal_summaries)
    helper_count_sum = sum(s.get("avg_helper_count", 0.0) * s.get("candidate_eval_count", 0) for s in signal_summaries)
    helper_top2_mean_drop_sum = sum(s.get("avg_helper_top2_mean_drop", 0.0) * s.get("candidate_eval_count", 0) for s in signal_summaries)
    helper_top2_max_drop_sum = sum(s.get("avg_helper_top2_max_drop", 0.0) * s.get("candidate_eval_count", 0) for s in signal_summaries)
    helper_top2_sensitive_total = sum(s.get("helper_top2_sensitive_count", 0) for s in signal_summaries)
    helper_top2_event_total = sum(s.get("helper_top2_event_count", 0) for s in signal_summaries)
    mean_perturbed_rise_sum = sum(s.get("avg_mean_perturbed_rise", 0.0) * s.get("candidate_eval_count", 0) for s in signal_summaries)
    positive_perturbed_rise_rate_sum = sum(s.get("avg_positive_perturbed_rise_rate", 0.0) * s.get("candidate_eval_count", 0) for s in signal_summaries)
    self_rise_sum = sum(s.get("avg_self_rise", 0.0) * s.get("candidate_eval_count", 0) for s in signal_summaries)
    helper_rise_sum = sum(s.get("avg_helper_rise", 0.0) * s.get("candidate_eval_count", 0) for s in signal_summaries)

    summary = {
        "total_candidate_evals": total_candidate_evals,
        "accepted_count": total_accepted,
        "rejected_count": total_rejected,
        "positive_helper_rise_count": total_positive_helper_rise,
        "negative_helper_rise_count": total_negative_helper_rise,
        "top1_changed_count": total_top1_changed,
        "acceptance_rate": round(total_accepted / total_candidate_evals, 6)
        if total_candidate_evals > 0
        else 0.0,
        "positive_helper_rise_rate": round(total_positive_helper_rise / total_candidate_evals, 6)
        if total_candidate_evals > 0
        else 0.0,
        "negative_helper_rise_rate": round(total_negative_helper_rise / total_candidate_evals, 6)
        if total_candidate_evals > 0
        else 0.0,
        "avg_commit_risk": round(commit_risk_sum / total_candidate_evals, 6)
        if total_candidate_evals > 0
        else 0.0,
        "avg_helper_masked_rise": round(helper_masked_rise_sum / total_candidate_evals, 6)
        if total_candidate_evals > 0
        else 0.0,
        "avg_helper_count": round(helper_count_sum / total_candidate_evals, 6)
        if total_candidate_evals > 0
        else 0.0,
    }
    if helper_top2_event_total > 0:
        summary["avg_helper_top2_mean_drop"] = round(helper_top2_mean_drop_sum / total_candidate_evals, 6) if total_candidate_evals > 0 else 0.0
        summary["avg_helper_top2_max_drop"] = round(helper_top2_max_drop_sum / total_candidate_evals, 6) if total_candidate_evals > 0 else 0.0
        summary["helper_top2_sensitive_count"] = helper_top2_sensitive_total
        summary["helper_top2_event_count"] = helper_top2_event_total
        summary["helper_top2_sensitive_rate"] = round(helper_top2_sensitive_total / helper_top2_event_total, 6)
        summary["avg_mean_perturbed_rise"] = round(mean_perturbed_rise_sum / total_candidate_evals, 6) if total_candidate_evals > 0 else 0.0
        summary["avg_positive_perturbed_rise_rate"] = round(positive_perturbed_rise_rate_sum / total_candidate_evals, 6) if total_candidate_evals > 0 else 0.0
    if self_rise_sum != 0.0 or helper_rise_sum != 0.0 or total_self_stronger_than_helper > 0 or total_self_positive_helper_negative > 0:
        summary["avg_self_rise"] = round(self_rise_sum / total_candidate_evals, 6) if total_candidate_evals > 0 else 0.0
        summary["avg_helper_rise"] = round(helper_rise_sum / total_candidate_evals, 6) if total_candidate_evals > 0 else 0.0
        summary["self_stronger_than_helper_count"] = total_self_stronger_than_helper
        summary["self_positive_helper_negative_count"] = total_self_positive_helper_negative
    return summary


def _aggregate_signal_summary(step_outputs):
    total_steps = len(step_outputs)
    signal_summaries = [step_entry.get("monitoring", {}).get("signal_summary", {}) for step_entry in step_outputs]
    summary = _merge_signal_summaries(signal_summaries)
    summary["total_steps"] = total_steps
    return summary


def _annotate_step_outputs_with_final_tokens(all_step_outputs, final_tokens, tokenizer):
    final_tokens_list = final_tokens.tolist()
    final_token_text_cache = {}

    def _decode_token(token_id):
        if token_id not in final_token_text_cache:
            final_token_text_cache[token_id] = tokenizer.decode(
                [token_id], skip_special_tokens=False
            )
        return final_token_text_cache[token_id]

    for step_idx, step_entry in enumerate(all_step_outputs):
        total_steps = max(len(all_step_outputs), 1)
        normalized_progress = round((step_idx + 1) / total_steps, 6)
        if normalized_progress <= 0.25:
            step_phase = "early"
        elif normalized_progress <= 0.75:
            step_phase = "middle"
        else:
            step_phase = "late"

        monitoring = step_entry.setdefault("monitoring", {})
        monitoring["step_phase"] = step_phase
        monitoring["normalized_progress"] = normalized_progress

        for detail in monitoring.get("gate_details", []):
            position = detail.get("position")
            if position is None or position >= len(final_tokens_list):
                continue
            final_token_id = int(final_tokens_list[position])
            pred_token_id = int(detail.get("pred_token_id", -1))
            detail["final_token_id"] = final_token_id
            detail["final_token_text"] = _decode_token(final_token_id)
            detail["matches_final_token"] = bool(pred_token_id == final_token_id)
            detail["step_phase"] = step_phase
            detail["normalized_progress"] = normalized_progress

        for detail in monitoring.get("candidate_ordering_details", []):
            position = detail.get("position")
            if position is None or position >= len(final_tokens_list):
                continue
            final_token_id = int(final_tokens_list[position])
            pred_token_id = int(detail.get("pred_token_id", -1))
            detail["final_token_id"] = final_token_id
            detail["final_token_text"] = _decode_token(final_token_id)
            detail["matches_final_token"] = bool(pred_token_id == final_token_id)
            detail["step_phase"] = step_phase
            detail["normalized_progress"] = normalized_progress


@torch.no_grad()
def llada_ours_hard_gate_decode(
    model,
    tokenizer,
    input_ids_original,
    gen_length,
    steps,
    block_length,
    temperature=0.0,
    mask_id=126336,
    step_save_dir=None,
    example_idx=0,
    helper_k=4,
    risk_threshold=0.05,
    min_confidence=0.0,
    disable_risk_gate=False,
    candidate_multiplier=2,
    collect_probe_details=False,
    selection_strategy="hard_gate",
):
    x = torch.full(
        (1, input_ids_original.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=model.device,
    )
    x[:, : input_ids_original.shape[1]] = input_ids_original.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    used_steps = 0
    all_step_outputs = [] if step_save_dir else None
    aggregate_signal_summaries = []
    total_reveal_budget = 0
    total_selected = 0
    total_selected_via_backfill = 0
    total_selected_from_gate = 0
    total_gated = 0
    total_probe_forwards = 0
    fallback_count = 0
    acceptance_rate_sum = 0.0
    collect_helper_top2_perturbation = collect_probe_details

    for num_block in range(num_blocks):
        block_start = input_ids_original.shape[1] + num_block * block_length
        block_end = input_ids_original.shape[1] + (num_block + 1) * block_length
        block_mask_index = x[:, block_start:block_end] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for step in range(steps_per_block):
            mask_index = x == mask_id
            block_mask = torch.zeros_like(mask_index)
            block_mask[:, block_start:block_end] = True
            mask_index = mask_index & block_mask

            if not mask_index[:, block_start:block_end].any():
                break

            logits = model(x).logits
            if temperature > 0:
                logits = add_gumbel_noise(logits, temperature)
            p_curr = F.softmax(logits.to(torch.float64), dim=-1)
            x0 = torch.argmax(p_curr, dim=-1)
            confidence = torch.squeeze(
                torch.gather(p_curr, dim=-1, index=torch.unsqueeze(x0, -1)), -1
            )

            reveal_budget_per_batch = num_transfer_tokens[:, step].tolist()
            ready_mask, gate_debug = _target_conditioned_local_commit_risk_hard_gate(
                x=x,
                p_curr=p_curr,
                x0=x0,
                mask_index=mask_index,
                confidence=confidence,
                reveal_budget_per_batch=reveal_budget_per_batch,
                helper_k=helper_k,
                risk_threshold=risk_threshold,
                min_confidence=min_confidence,
                candidate_multiplier=candidate_multiplier,
                model=model,
                force_accept_all=disable_risk_gate,
                collect_probe_details=collect_probe_details,
                collect_step_debug=bool(step_save_dir),
                collect_helper_top2_perturbation=collect_helper_top2_perturbation,
                selection_strategy=selection_strategy,
            )

            decoded_token_info = []
            if step_save_dir:
                all_tokens_info = []
                for j in range(mask_index.shape[0]):
                    masked_indices_in_block = (
                        torch.where(mask_index[j, block_start:block_end])[0] + block_start
                    )
                    for idx in masked_indices_in_block:
                        token_id = x0[j, idx].item()
                        conf_val = confidence[j, idx].item()
                        all_tokens_info.append(
                            {
                                "position": int(idx),
                                "token_id": token_id,
                                "confidence": round(float(conf_val), 6),
                            }
                        )
                for batch_idx, debug_entry in enumerate(gate_debug):
                    all_tokens_info.append(
                        {
                            "batch_idx": batch_idx,
                            "target_conditioned_local_commit_risk_hard_gate": debug_entry,
                        }
                    )

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x.device)
            for j in range(ready_mask.shape[0]):
                selected_indices = torch.where(ready_mask[j])[0]
                transfer_index[j, selected_indices] = True

                if step_save_dir:
                    for idx in selected_indices:
                        token_id = x0[j, idx].item()
                        decoded_token = tokenizer.decode([token_id], skip_special_tokens=False)
                        conf_val = confidence[j, idx].item()
                        decoded_token_info.append(
                            {
                                "position": int(idx),
                                "token_id": token_id,
                                "decoded_token": decoded_token,
                                "confidence": round(float(conf_val), 6),
                            }
                        )

            x[transfer_index] = x0[transfer_index]

            step_monitor = gate_debug[0] if len(gate_debug) > 0 else {}
            aggregate_signal_summaries.append(step_monitor.get("signal_summary", {}))
            total_reveal_budget += int(step_monitor.get("reveal_budget", 0))
            total_selected += len(step_monitor.get("selected_positions", []))
            total_selected_via_backfill += len(step_monitor.get("selected_via_backfill", []))
            total_selected_from_gate += len(step_monitor.get("selected_from_gate", []))
            total_gated += int(step_monitor.get("gate_rejected_count", 0))
            total_probe_forwards += int(step_monitor.get("extra_probe_forwards", 0))
            if step_monitor.get("fallback_triggered", False):
                fallback_count += 1
            acceptance_rate_sum += float(step_monitor.get("acceptance_rate_among_initial_candidates", 0.0))

            if step_save_dir:
                decoded_text = tokenizer.batch_decode(
                    x[:, input_ids_original.shape[1] :], skip_special_tokens=True
                )[0]
                all_step_outputs.append(
                    {
                        "step": used_steps + 1,
                        "decoded_text": decoded_text,
                        "decoded_tokens_num": len(decoded_token_info),
                        "decoded_tokens": decoded_token_info,
                        "monitoring": step_monitor,
                        "all_tokens": all_tokens_info,
                    }
                )

            used_steps += 1

    monitor_summary = {
        "total_steps": used_steps,
        "total_reveal_budget": total_reveal_budget,
        "total_selected": total_selected,
        "total_selected_via_backfill": total_selected_via_backfill,
        "total_selected_from_gate": total_selected_from_gate,
        "total_gated": total_gated,
        "total_probe_forwards": total_probe_forwards,
        "fallback_count": fallback_count,
        "avg_acceptance_rate_among_initial_candidates": round(
            acceptance_rate_sum / used_steps,
            6,
        ) if used_steps > 0 else 0.0,
        "signal_summary": _merge_signal_summaries(aggregate_signal_summaries),
    }
    if used_steps > 0:
        monitor_summary["avg_reveal_budget_per_step"] = round(
            monitor_summary["total_reveal_budget"] / used_steps, 4
        )
        monitor_summary["avg_selected_per_step"] = round(
            monitor_summary["total_selected"] / used_steps, 4
        )
        monitor_summary["avg_gated_per_step"] = round(
            monitor_summary["total_gated"] / used_steps, 4
        )
        monitor_summary["avg_selected_via_backfill_per_step"] = round(
            monitor_summary["total_selected_via_backfill"] / used_steps, 4
        )
        monitor_summary["avg_selected_from_gate_per_step"] = round(
            monitor_summary["total_selected_from_gate"] / used_steps, 4
        )
    else:
        monitor_summary["avg_reveal_budget_per_step"] = 0.0
        monitor_summary["avg_selected_per_step"] = 0.0
        monitor_summary["avg_gated_per_step"] = 0.0
        monitor_summary["avg_selected_via_backfill_per_step"] = 0.0
        monitor_summary["avg_selected_from_gate_per_step"] = 0.0

    if step_save_dir:
        _annotate_step_outputs_with_final_tokens(
            all_step_outputs=all_step_outputs,
            final_tokens=x[0, input_ids_original.shape[1] :],
            tokenizer=tokenizer,
        )
        all_steps_path = os.path.join(step_save_dir, f"all_steps_{example_idx}.json")
        with open(all_steps_path, "w") as f:
            json.dump(all_step_outputs, f, indent=2)

    return x, used_steps, monitor_summary
