import os
import json
import torch
import numpy as np
import torch.nn.functional as F
import math

from .llada_ours_hard_gate import _target_conditioned_local_commit_risk_hard_gate


def _to_int_list(indices):
    return [int(x) for x in indices]


def _build_confidence_candidates(confidence, mask_index, num_select):
    candidate_mask = torch.zeros_like(mask_index, dtype=torch.bool)
    for j in range(mask_index.shape[0]):
        masked_indices = torch.where(mask_index[j])[0]
        if len(masked_indices) == 0:
            continue
        k_select = min(num_select[j], len(masked_indices))
        if k_select <= 0:
            continue
        _, candidate_rel = torch.topk(confidence[j, masked_indices], k=k_select)
        candidate_indices = masked_indices[candidate_rel]
        candidate_mask[j, candidate_indices] = True
    return candidate_mask


def _select_local_helper_positions(
    candidate_idx,
    masked_indices,
    confidence,
    margin,
    helper_k,
    helper_direction,
):
    if helper_k <= 0:
        return torch.empty(0, dtype=torch.long, device=masked_indices.device)

    if helper_direction == "before":
        local_mask = masked_indices < candidate_idx
    else:
        local_mask = torch.ones_like(masked_indices, dtype=torch.bool)

    helper_candidates = masked_indices[local_mask]
    helper_candidates = helper_candidates[helper_candidates != candidate_idx]
    if len(helper_candidates) == 0:
        return torch.empty(0, dtype=torch.long, device=masked_indices.device)

    distances = torch.abs(helper_candidates - candidate_idx)
    near_limit = min(len(helper_candidates), max(helper_k * 4, helper_k))
    _, near_rel = torch.topk(-distances.to(torch.float64), k=near_limit)
    nearby_candidates = helper_candidates[near_rel]
    helper_scores = confidence[nearby_candidates] * margin[nearby_candidates]
    keep_k = min(helper_k, len(nearby_candidates))
    _, helper_rel = torch.topk(helper_scores, k=keep_k)
    return nearby_candidates[helper_rel]


def _target_conditioned_local_commit_risk_rerank(
    x,
    p_curr,
    x0,
    mask_index,
    confidence,
    reveal_budget_per_batch,
    candidate_multiplier,
    boundary_window,
    helper_k,
    helper_direction,
    lambda_r,
    model,
):
    batch_size = mask_index.shape[0]
    ready_mask = torch.zeros_like(mask_index, dtype=torch.bool)
    debug_info = []
    sorted_probs, _ = torch.sort(p_curr, dim=-1, descending=True)
    top1_prob = sorted_probs[..., 0]
    top2_prob = sorted_probs[..., 1]
    margin = top1_prob - top2_prob

    for j in range(batch_size):
        masked_indices = torch.where(mask_index[j])[0]
        if len(masked_indices) == 0:
            debug_info.append({
                "candidate_pool": [],
                "probed_boundary_candidates": [],
                "helper_positions": {},
                "commit_risk": {},
                "reranked_boundary_order": [],
                "final_revealed_positions": [],
            })
            continue

        reveal_budget = int(reveal_budget_per_batch[j])
        if reveal_budget <= 0:
            debug_info.append({
                "candidate_pool": [],
                "probed_boundary_candidates": [],
                "helper_positions": {},
                "commit_risk": {},
                "reranked_boundary_order": [],
                "final_revealed_positions": [],
            })
            continue

        candidate_pool_size = min(len(masked_indices), max(reveal_budget, candidate_multiplier * reveal_budget))
        _, pool_rel = torch.topk(confidence[j, masked_indices], k=candidate_pool_size)
        candidate_pool = masked_indices[pool_rel]

        boundary_start = max(0, reveal_budget - boundary_window - 1)
        boundary_end = min(len(candidate_pool), reveal_budget + boundary_window)
        boundary_candidates = candidate_pool[boundary_start:boundary_end]

        rerank_score_map = {int(idx): float(confidence[j, idx].item()) for idx in candidate_pool}
        helper_log = {}
        commit_risk_log = {}
        probe_forwards = 0

        for candidate_idx in boundary_candidates:
            helper_positions = _select_local_helper_positions(
                candidate_idx=candidate_idx,
                masked_indices=masked_indices,
                confidence=confidence[j],
                margin=margin[j],
                helper_k=helper_k,
                helper_direction=helper_direction,
            )
            helper_log[int(candidate_idx)] = _to_int_list(helper_positions)

            x_probe = x.clone()
            x_probe[j, candidate_idx] = x0[j, candidate_idx]
            if len(helper_positions) > 0:
                x_probe[j, helper_positions] = x0[j, helper_positions]

            probe_logits = model(x_probe).logits
            probe_probs = F.softmax(probe_logits.to(torch.float64), dim=-1)
            probe_forwards += 1

            target_token = x0[j, candidate_idx].view(1, 1)
            original_prob = torch.gather(p_curr[j, candidate_idx].view(1, -1), dim=-1, index=target_token).squeeze()
            probe_prob = torch.gather(probe_probs[j, candidate_idx].view(1, -1), dim=-1, index=target_token).squeeze()
            commit_risk = (original_prob - probe_prob).item()
            commit_risk_log[int(candidate_idx)] = float(commit_risk)
            rerank_score_map[int(candidate_idx)] = float(confidence[j, candidate_idx].item() - lambda_r * commit_risk)

        boundary_sorted = sorted(
            [int(idx) for idx in boundary_candidates],
            key=lambda idx: rerank_score_map[idx],
            reverse=True,
        )

        final_order = [int(idx) for idx in candidate_pool]
        final_order[boundary_start:boundary_end] = boundary_sorted
        final_revealed = final_order[:reveal_budget]
        ready_mask[j, torch.tensor(final_revealed, device=mask_index.device, dtype=torch.long)] = True

        debug_info.append({
            "candidate_pool": [
                {"position": int(idx), "confidence": round(float(confidence[j, idx].item()), 6)}
                for idx in candidate_pool
            ],
            "probed_boundary_candidates": _to_int_list(boundary_candidates),
            "helper_positions": {str(k): v for k, v in helper_log.items()},
            "commit_risk": {str(k): round(v, 6) for k, v in commit_risk_log.items()},
            "reranked_boundary_order": boundary_sorted,
            "final_revealed_positions": final_revealed,
            "extra_probe_forwards": probe_forwards,
        })

    return ready_mask, debug_info


def _compute_shared_anchor_commit_risk(
    x,
    p_curr,
    x0,
    mask_index,
    original_conf,
    m,
    k,
    model,
):
    sorted_probs, _ = torch.sort(p_curr, dim=-1, descending=True)
    top1_prob = sorted_probs[..., 0]
    top2_prob = sorted_probs[..., 1]
    margin = top1_prob - top2_prob
    rho = original_conf * margin
    rho = torch.where(mask_index, rho, torch.full_like(rho, float("-inf")))

    anchor_mask = torch.zeros_like(mask_index, dtype=torch.bool)

    for j in range(mask_index.shape[0]):
        masked_indices = torch.where(mask_index[j])[0]
        if len(masked_indices) == 0:
            continue

        pool_size = min(m, len(masked_indices))
        if pool_size <= 0:
            continue

        _, stable_pool_rel = torch.topk(rho[j, masked_indices], k=pool_size)
        stable_pool = masked_indices[stable_pool_rel]

        anchor_size = min(k, len(stable_pool))
        if anchor_size <= 0:
            continue

        _, anchor_rel = torch.topk(rho[j, stable_pool], k=anchor_size)
        anchors = stable_pool[anchor_rel]
        anchor_mask[j, anchors] = True

    if not anchor_mask.any():
        return None, anchor_mask

    x_probe = x.clone()
    x_probe[anchor_mask] = x0[anchor_mask]
    probe_logits = model(x_probe).logits
    probe_probs = F.softmax(probe_logits.to(torch.float64), dim=-1)

    gather_index = torch.unsqueeze(x0, -1)
    original_top1_prob = torch.squeeze(torch.gather(p_curr, dim=-1, index=gather_index), -1)
    probe_top1_prob = torch.squeeze(torch.gather(probe_probs, dim=-1, index=gather_index), -1)
    commit_risk = original_top1_prob - probe_top1_prob

    return commit_risk, anchor_mask


def _compute_shared_anchor_gate(
    x,
    p_curr,
    x0,
    mask_index,
    original_conf,
    candidate_mask,
    m,
    k,
    tau_r,
    model,
):
    if candidate_mask is None or not candidate_mask.any() or m <= 0 or k <= 0:
        return candidate_mask, None

    commit_risk, anchor_mask = _compute_shared_anchor_commit_risk(
        x=x,
        p_curr=p_curr,
        x0=x0,
        mask_index=mask_index,
        original_conf=original_conf,
        m=m,
        k=k,
        model=model,
    )

    if commit_risk is None:
        return candidate_mask, anchor_mask

    gated_mask = candidate_mask & (~anchor_mask) & (commit_risk <= tau_r)
    fallback_mask = gated_mask.sum(dim=-1) == 0
    if fallback_mask.any():
        gated_mask[fallback_mask] = candidate_mask[fallback_mask]

    return gated_mask, anchor_mask

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Token-wise entropy H(p) where p=softmax(logits). Returns shape [...]."""
    p = F.softmax(logits.to(torch.float64), dim=-1)
    eps = 1e-12
    return -(p * torch.log(p + eps)).sum(dim=-1)


def _build_boundary_reranked_ready_mask(
    confidence: torch.Tensor,
    rerank_bonus: torch.Tensor,
    mask_index: torch.Tensor,
    reveal_budget_per_batch,
    boundary_window: int,
    rerank_lambda: float,
):
    ready_mask = torch.zeros_like(mask_index, dtype=torch.bool)
    debug_info = []

    for j in range(mask_index.shape[0]):
        masked_indices = torch.where(mask_index[j])[0]
        if len(masked_indices) == 0:
            debug_info.append({
                "candidate_pool": [],
                "boundary_candidates": [],
                "baseline_boundary_order": [],
                "reranked_boundary_order": [],
                "boundary_bonus": {},
                "rerank_scores": {},
                "final_revealed_positions": [],
            })
            continue

        reveal_budget = int(reveal_budget_per_batch[j])
        if reveal_budget <= 0:
            debug_info.append({
                "candidate_pool": [],
                "boundary_candidates": [],
                "baseline_boundary_order": [],
                "reranked_boundary_order": [],
                "boundary_bonus": {},
                "rerank_scores": {},
                "final_revealed_positions": [],
            })
            continue

        pool_size = min(len(masked_indices), max(reveal_budget + max(int(boundary_window), 0), reveal_budget))
        _, pool_rel = torch.topk(confidence[j, masked_indices], k=pool_size)
        candidate_pool = masked_indices[pool_rel]

        boundary_start = max(0, reveal_budget - max(int(boundary_window), 0) - 1)
        boundary_end = min(len(candidate_pool), reveal_budget + max(int(boundary_window), 0))
        boundary_candidates = candidate_pool[boundary_start:boundary_end]
        baseline_boundary_order = [int(idx) for idx in boundary_candidates]

        rerank_score_map = {int(idx): float(confidence[j, idx].item()) for idx in candidate_pool}
        boundary_bonus_map = {}
        for idx in boundary_candidates:
            idx_int = int(idx)
            boundary_bonus_map[idx_int] = float(rerank_bonus[j, idx].item())
            rerank_score_map[idx_int] = float(
                confidence[j, idx].item() + float(rerank_lambda) * rerank_bonus[j, idx].item()
            )

        boundary_sorted = sorted(
            [int(idx) for idx in boundary_candidates],
            key=lambda idx: rerank_score_map[idx],
            reverse=True,
        )

        final_order = [int(idx) for idx in candidate_pool]
        final_order[boundary_start:boundary_end] = boundary_sorted
        final_revealed = final_order[:reveal_budget]
        if len(final_revealed) > 0:
            ready_mask[j, torch.tensor(final_revealed, device=mask_index.device, dtype=torch.long)] = True

        debug_info.append({
            "candidate_pool": [int(idx) for idx in candidate_pool],
            "boundary_candidates": [int(idx) for idx in boundary_candidates],
            "baseline_boundary_order": baseline_boundary_order,
            "reranked_boundary_order": boundary_sorted,
            "boundary_bonus": {str(k): round(v, 6) for k, v in boundary_bonus_map.items()},
            "rerank_scores": {str(k): round(v, 6) for k, v in rerank_score_map.items()},
            "final_revealed_positions": final_revealed,
        })

    return ready_mask, debug_info

@ torch.no_grad()
def stable_confident_decode(
    model, tokenizer, input_ids_original, gen_length, steps, block_length, temperature=0., mask_id=126336,
    conf_threshold=0.9, kl_threshold=0.01, kl_history_length=2, 
    step_save_dir=None, example_idx=0,
    alg="default",
    unmask_strategy="all",
    # TILG-style proposal (then gate with KLASS)
    guidance_weight: float = 1.5,
    tilg_proposal_ratio: float = 0.1,
    tilg_ema_decay: float = 0.0,
    tilg_rerank_lambda: float = 0.2,
    tilg_boundary_window: int = 2,
    tilg_extra_conf_floor: float = 0.85,
    tilg_extra_ratio: float = 0.25,
    tilg_extra_max: int = 2,
    tilg_extra_allow_empty_base: int = 1,
    # Hard token-id stability gate (majority vote over last K steps)
    hard_token_gate_k: int = 0,
    hard_token_gate_min_agree: int = 0,
    history_gate_min_streak: int = 0,
    history_gate_confidence_escape: float = 1.0,
    history_gate_max_wait: int = 0,
    use_shared_anchor_commit_risk_gate=False,
    shared_anchor_m=8,
    shared_anchor_k=2,
    shared_anchor_tau_r=0.05,
    shared_anchor_rerank_lambda=0.2,
    target_candidate_multiplier=2,
    target_boundary_window=2,
    target_helper_k=1,
    target_helper_selection_mode="nearby_high_conf_stable",
    target_helper_direction="before",
    use_target_conditioned_commit_risk=True,
):
    """
    Decoding strategy: Unmask tokens that are both high-confidence and have stable (low KL-divergence) softmax distributions over H steps.
    Implements alg options: default, random, topk_margin, entropy.
    """
    mask_id = 126336
    x = torch.full((1, input_ids_original.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :input_ids_original.shape[1]] = input_ids_original.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    used_steps = 0

    # History buffers
    V = model.lm_head.out_features if hasattr(model, "lm_head") else model.config.vocab_size
    kl_history = torch.zeros((1, x.shape[1], kl_history_length), dtype=torch.float64, device=x.device)
    p_prev = torch.zeros((1, x.shape[1], V), dtype=torch.float64, device=x.device)
    # For TILG-style temporal uncond proxy (logits EMA).
    # We must use *previous* step logits (or slow EMA of history) as the proxy.
    slow_ema_logits = None
    prev_step_logits = None
    prev_step_x0 = None
    prev_top1_for_streak = None
    top1_streak = None
    ready_wait = None
    # For hard token-id stability gate (track per-position top1 token history).
    top1_hist = None

    all_step_outputs = []
    total_decoded_tokens = 0

    for num_block in range(num_blocks):
        block_start = input_ids_original.shape[1] + num_block * block_length
        block_end = input_ids_original.shape[1] + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for step in range(steps_per_block):
            mask_index = (x == mask_id)
            # --- Restrict to current block ---
            block_mask = torch.zeros_like(mask_index)
            block_mask[:, block_start:block_end] = True
            mask_index = mask_index & block_mask

            # --- Break if all tokens in current block are unmasked ---
            if not mask_index[:, block_start:block_end].any():
                break

            logits = model(x).logits
            if temperature > 0:
                logits = add_gumbel_noise(logits, temperature)
            p_curr = F.softmax(logits.to(torch.float64), dim=-1)
            x0 = torch.argmax(p_curr, dim=-1)
            x0_write = x0

            logits_fp64 = logits.to(torch.float64)

            # Build temporal uncond proxy BEFORE updating buffers.
            # - tilg_ema_decay == 0: use previous step logits directly (weak but "free" proxy)
            # - tilg_ema_decay  > 0: use slow EMA of conditional logits (stronger temporal lag)
            if float(tilg_ema_decay) > 0:
                if slow_ema_logits is None:
                    slow_ema_logits = logits_fp64.clone()
                uncond_proxy_logits = slow_ema_logits
            else:
                uncond_proxy_logits = prev_step_logits if prev_step_logits is not None else logits_fp64

            # --- Compute confidence according to alg ---
            if alg == "random":
                curr_conf = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            elif alg == "topk_margin":
                sorted_probs, _ = torch.sort(p_curr, dim=-1, descending=True)
                top1 = sorted_probs[..., 0]
                top2 = sorted_probs[..., 1]
                curr_conf = top1 - top2
            elif alg == "entropy":
                eps_ent = 1e-10
                log_p = torch.log(p_curr + eps_ent)
                curr_conf = -torch.sum(p_curr * log_p, dim=-1)  # negative entropy (lower entropy = higher confidence)
            else:  # default (top confidence)
                curr_conf = torch.squeeze(torch.gather(p_curr, dim=-1, index=torch.unsqueeze(x0, -1)), -1)

            # KL divergence between current and previous step
            eps = 1e-12
            kl_current_prev = (p_curr * (torch.log(p_curr + eps)
                            - torch.log(p_prev + eps))
                 ).sum(dim=-1)
            # Shift kl_history and insert new KL at the end
            kl_history = torch.roll(kl_history, shifts=-1, dims=-1)
            kl_history[..., -1] = kl_current_prev

            p_prev = p_curr.clone()

            if alg in ["klass", "confidence_threshold"]:
                # --- KL threshold logic ---
                if step >= kl_history_length - 1:
                    stable_mask = torch.all(kl_history < kl_threshold, dim=-1)
                else:
                    stable_mask = torch.zeros_like(curr_conf, dtype=torch.bool)
                # --- Confidence threshold logic ---
                conf_mask = curr_conf > conf_threshold

                ready_mask = stable_mask & conf_mask & mask_index
                if use_shared_anchor_commit_risk_gate:
                    ready_mask, anchor_mask = _compute_shared_anchor_gate(
                        x=x,
                        p_curr=p_curr,
                        x0=x0,
                        mask_index=mask_index,
                        original_conf=curr_conf,
                        candidate_mask=ready_mask,
                        m=shared_anchor_m,
                        k=shared_anchor_k,
                        tau_r=shared_anchor_tau_r,
                        model=model,
                    )
                    local_rerank_debug = None
                else:
                    anchor_mask = None
                    local_rerank_debug = None
            elif alg == "confidence_threshold_history_gate":
                cond_top1_prob = torch.squeeze(
                    torch.gather(p_curr, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
                if step >= kl_history_length - 1:
                    stable_mask = torch.all(kl_history < kl_threshold, dim=-1)
                else:
                    stable_mask = torch.zeros_like(cond_top1_prob, dtype=torch.bool)
                ready_mask = stable_mask & (cond_top1_prob > float(conf_threshold)) & mask_index
                x0_write = x0
                curr_conf = cond_top1_prob
                anchor_mask = None
                local_rerank_debug = None
            elif alg in ["confidence_threshold_tilg", "confidence_threshold_tilg_discrete", "confidence_threshold_tilg_added_gate", "confidence_threshold_tilg_history_gate", "confidence_threshold_tilg_history_gate_rerank_only", "confidence_threshold_tilg_history_gate_capped_extra", "confidence_threshold_tilg_history_gate_capped_extra_dual", "klass_tilg_confidence"]:
                guided_logits = uncond_proxy_logits + (1.0 + float(guidance_weight)) * (logits_fp64 - uncond_proxy_logits)
                guided_probs = F.softmax(guided_logits, dim=-1)
                cond_top1_prob = torch.squeeze(
                    torch.gather(p_curr, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
                prev_probs = F.softmax(uncond_proxy_logits, dim=-1)
                prev_on_cond = torch.squeeze(
                    torch.gather(prev_probs, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
                guided_on_cond = torch.squeeze(
                    torch.gather(guided_probs, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
                continuous_bonus = torch.clamp(guided_on_cond - prev_on_cond, min=0.0)
                tilg_score = cond_top1_prob + float(tilg_rerank_lambda) * continuous_bonus
                if step >= kl_history_length - 1:
                    stable_mask = torch.all(kl_history < kl_threshold, dim=-1)
                else:
                    stable_mask = torch.zeros_like(tilg_score, dtype=torch.bool)
                base_ready_mask = stable_mask & (cond_top1_prob > float(conf_threshold)) & mask_index
                tilg_ready_mask = stable_mask & (tilg_score > float(conf_threshold)) & mask_index
                if alg == "confidence_threshold_tilg_history_gate_rerank_only":
                    ready_mask = base_ready_mask
                elif alg in ["confidence_threshold_tilg_history_gate_capped_extra", "confidence_threshold_tilg_history_gate_capped_extra_dual"]:
                    ready_mask = base_ready_mask.clone()
                    near_threshold_mask = (
                        stable_mask
                        & mask_index
                        & (cond_top1_prob > float(tilg_extra_conf_floor))
                        & (cond_top1_prob <= float(conf_threshold))
                    )
                    for batch_idx in range(ready_mask.shape[0]):
                        base_count = int(base_ready_mask[batch_idx].sum().item())
                        if base_count <= 0 and int(tilg_extra_allow_empty_base) <= 0:
                            continue
                        raw_budget = int(math.ceil(float(tilg_extra_ratio) * float(max(base_count, 1))))
                        extra_budget = min(int(tilg_extra_max), max(0, raw_budget))
                        if extra_budget <= 0:
                            continue
                        extra_indices = torch.where(near_threshold_mask[batch_idx])[0]
                        if len(extra_indices) == 0:
                            continue
                        k_extra = min(extra_budget, len(extra_indices))
                        _, extra_rel = torch.topk(tilg_score[batch_idx, extra_indices], k=k_extra)
                        ready_mask[batch_idx, extra_indices[extra_rel]] = True
                else:
                    ready_mask = tilg_ready_mask
                if alg in ["confidence_threshold_tilg_discrete", "confidence_threshold_tilg_added_gate"]:
                    if prev_step_x0 is None:
                        top1_stable_mask = torch.zeros_like(ready_mask, dtype=torch.bool)
                    else:
                        top1_stable_mask = (x0 == prev_step_x0) & mask_index
                    if alg == "confidence_threshold_tilg_discrete":
                        ready_mask = tilg_ready_mask & top1_stable_mask
                    else:
                        tilg_added_mask = tilg_ready_mask & (~base_ready_mask)
                        ready_mask = base_ready_mask | (tilg_added_mask & top1_stable_mask)
                if alg == "confidence_threshold_tilg_history_gate_capped_extra_dual":
                    x0_write = torch.argmax(guided_probs, dim=-1)
                else:
                    x0_write = x0
                curr_conf = tilg_score
                anchor_mask = None
                local_rerank_debug = None
            elif alg == "tilg_original":
                # Legacy port kept for reference: guidance also affects commit ordering.
                guided_logits = uncond_proxy_logits + (1.0 + float(guidance_weight)) * (logits_fp64 - uncond_proxy_logits)
                guided_probs = F.softmax(guided_logits, dim=-1)
                x0_write = torch.argmax(guided_probs, dim=-1)
                curr_conf = torch.squeeze(torch.gather(guided_probs, dim=-1, index=torch.unsqueeze(x0_write, -1)), -1)
                ready_mask = mask_index.clone()
                anchor_mask = None
                local_rerank_debug = None
            elif alg == "tilg_token_only":
                # Corrected TILG port for LLaDA:
                # - position selection stays with native KLASS/confidence logic
                # - TILG only changes the written token via guided logits
                guided_logits = uncond_proxy_logits + (1.0 + float(guidance_weight)) * (logits_fp64 - uncond_proxy_logits)
                guided_probs = F.softmax(guided_logits, dim=-1)
                x0_write = torch.argmax(guided_probs, dim=-1)

                if step >= kl_history_length - 1:
                    stable_mask = torch.all(kl_history < kl_threshold, dim=-1)
                else:
                    stable_mask = torch.zeros_like(curr_conf, dtype=torch.bool)
                conf_mask = curr_conf > conf_threshold
                ready_mask = stable_mask & conf_mask & mask_index
                anchor_mask = None
                local_rerank_debug = None
            elif alg == "tilg_klass_gate":
                # Proposal: pick a subset of positions by (guided) entropy, then gate with KLASS stability+confidence.
                # Guidance (CFG-like) with temporal uncond proxy from EMA logits.
                guided_logits = uncond_proxy_logits + (1.0 + float(guidance_weight)) * (logits_fp64 - uncond_proxy_logits)
                guided_probs = F.softmax(guided_logits, dim=-1)
                x0_write = torch.argmax(guided_probs, dim=-1)
                ent = _entropy_from_logits(guided_logits)

                # KLASS gate (computed on conditional distribution p_curr / history).
                if step >= kl_history_length - 1:
                    stable_mask = torch.all(kl_history < kl_threshold, dim=-1)
                else:
                    stable_mask = torch.zeros_like(curr_conf, dtype=torch.bool)
                conf_mask = torch.squeeze(torch.gather(p_curr, dim=-1, index=torch.unsqueeze(x0, -1)), -1) > conf_threshold

                proposal_mask = torch.zeros_like(mask_index, dtype=torch.bool)
                for j in range(mask_index.shape[0]):
                    masked_indices = torch.where(mask_index[j])[0]
                    if len(masked_indices) == 0:
                        continue
                    reveal_budget = int(num_transfer_tokens[j, step].item())
                    if reveal_budget <= 0:
                        continue
                    # pool size: max(reveal_budget, ratio * masked_count)
                    pool_size = max(reveal_budget, int(math.ceil(float(tilg_proposal_ratio) * float(len(masked_indices)))))
                    pool_size = min(pool_size, len(masked_indices))
                    _, rel = torch.topk(ent[j, masked_indices], k=pool_size)  # high-entropy positions
                    proposal_indices = masked_indices[rel]
                    proposal_mask[j, proposal_indices] = True

                # Restrict ready_mask to proposal positions, then apply stability+confidence gate.
                ready_mask = proposal_mask & stable_mask & conf_mask & mask_index
                anchor_mask = None
                local_rerank_debug = None
            elif alg == "tilg_klass_gate_v1":
                # KLASS decides *which positions* to commit (stable + confident).
                # TILG decides *what token* to write there (guided logits from temporal proxy).
                guided_logits = uncond_proxy_logits + (1.0 + float(guidance_weight)) * (logits_fp64 - uncond_proxy_logits)
                guided_probs = F.softmax(guided_logits, dim=-1)
                x0_write = torch.argmax(guided_probs, dim=-1)

                if step >= kl_history_length - 1:
                    stable_mask = torch.all(kl_history < kl_threshold, dim=-1)
                else:
                    stable_mask = torch.zeros_like(curr_conf, dtype=torch.bool)
                conf_mask = curr_conf > conf_threshold
                ready_mask = stable_mask & conf_mask & mask_index
                anchor_mask = None
                local_rerank_debug = None
            elif alg == "confidence_first_tilg_rerank":
                # Confidence-first mainline + TILG-style continuous rerank.
                # Keep token identity from cond/x0, and only use temporal CFG-style
                # guidance to rerank boundary candidates.
                guided_logits = uncond_proxy_logits + (1.0 + float(guidance_weight)) * (logits_fp64 - uncond_proxy_logits)
                guided_probs = F.softmax(guided_logits, dim=-1)

                cond_top1_prob = torch.squeeze(
                    torch.gather(p_curr, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
                prev_probs = F.softmax(uncond_proxy_logits, dim=-1)
                prev_on_cond = torch.squeeze(
                    torch.gather(prev_probs, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
                guided_on_cond = torch.squeeze(
                    torch.gather(guided_probs, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
                continuous_bonus = torch.clamp(guided_on_cond - prev_on_cond, min=0.0)
                continuous_bonus = continuous_bonus * mask_index.to(continuous_bonus.dtype)

                reveal_budget_per_batch = num_transfer_tokens[:, step].tolist()
                ready_mask, local_rerank_debug = _build_boundary_reranked_ready_mask(
                    confidence=cond_top1_prob,
                    rerank_bonus=continuous_bonus,
                    mask_index=mask_index,
                    reveal_budget_per_batch=reveal_budget_per_batch,
                    boundary_window=tilg_boundary_window,
                    rerank_lambda=tilg_rerank_lambda,
                )
                x0_write = x0
                curr_conf = cond_top1_prob
                anchor_mask = None
            elif alg == "confidence_threshold_tilg_rerank":
                guided_logits = uncond_proxy_logits + (1.0 + float(guidance_weight)) * (logits_fp64 - uncond_proxy_logits)
                guided_probs = F.softmax(guided_logits, dim=-1)

                cond_top1_prob = torch.squeeze(
                    torch.gather(p_curr, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
                prev_probs = F.softmax(uncond_proxy_logits, dim=-1)
                prev_on_cond = torch.squeeze(
                    torch.gather(prev_probs, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
                guided_on_cond = torch.squeeze(
                    torch.gather(guided_probs, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
                continuous_bonus = torch.clamp(guided_on_cond - prev_on_cond, min=0.0)
                tilg_score = cond_top1_prob + float(tilg_rerank_lambda) * continuous_bonus
                ready_mask = (tilg_score > float(conf_threshold)) & mask_index
                if step == steps_per_block - 1:
                    ready_mask = mask_index.clone()
                x0_write = x0
                curr_conf = cond_top1_prob
                anchor_mask = None
                local_rerank_debug = None
            elif alg == "confidence_first_tilg_rerank_klass_gate":
                # Continuous TILG rerank proposes boundary candidates first,
                # then KLASS-style stability + confidence works as the discrete gate.
                guided_logits = uncond_proxy_logits + (1.0 + float(guidance_weight)) * (logits_fp64 - uncond_proxy_logits)
                guided_probs = F.softmax(guided_logits, dim=-1)

                cond_top1_prob = torch.squeeze(
                    torch.gather(p_curr, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
                prev_probs = F.softmax(uncond_proxy_logits, dim=-1)
                prev_on_cond = torch.squeeze(
                    torch.gather(prev_probs, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
                guided_on_cond = torch.squeeze(
                    torch.gather(guided_probs, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
                continuous_bonus = torch.clamp(guided_on_cond - prev_on_cond, min=0.0)
                continuous_bonus = continuous_bonus * mask_index.to(continuous_bonus.dtype)

                reveal_budget_per_batch = num_transfer_tokens[:, step].tolist()
                proposal_mask, local_rerank_debug = _build_boundary_reranked_ready_mask(
                    confidence=cond_top1_prob,
                    rerank_bonus=continuous_bonus,
                    mask_index=mask_index,
                    reveal_budget_per_batch=reveal_budget_per_batch,
                    boundary_window=tilg_boundary_window,
                    rerank_lambda=tilg_rerank_lambda,
                )

                if step >= kl_history_length - 1:
                    stable_mask = torch.all(kl_history < kl_threshold, dim=-1)
                else:
                    stable_mask = torch.zeros_like(cond_top1_prob, dtype=torch.bool)
                conf_mask = cond_top1_prob > conf_threshold

                ready_mask = proposal_mask & stable_mask & conf_mask & mask_index
                x0_write = x0
                curr_conf = cond_top1_prob
                anchor_mask = None
            elif alg == "confidence_first_hard_gate":
                # Confidence-first LLaDA + our target-conditioned hard gate.
                # Commit candidates come from top-confidence positions (equivalent to making KL gate inactive
                # with a huge threshold / history_length=1), then we apply the local commit-risk hard gate.
                reveal_budget_per_batch = num_transfer_tokens[:, step].tolist()
                ready_mask, local_rerank_debug = _target_conditioned_local_commit_risk_hard_gate(
                    x=x,
                    p_curr=p_curr,
                    x0=x0,
                    mask_index=mask_index,
                    confidence=curr_conf,
                    reveal_budget_per_batch=reveal_budget_per_batch,
                    helper_k=target_helper_k,
                    risk_threshold=shared_anchor_tau_r,
                    min_confidence=conf_threshold,
                    candidate_multiplier=target_candidate_multiplier,
                    model=model,
                    force_accept_all=False,
                    collect_probe_details=bool(step_save_dir),
                    collect_step_debug=bool(step_save_dir),
                    collect_helper_top2_perturbation=False,
                    selection_strategy="hard_gate",
                )
                anchor_mask = None
            elif alg == "shared_anchor_commit_risk_rerank":
                candidate_counts = num_transfer_tokens[:, step].tolist()
                candidate_mask = _build_confidence_candidates(curr_conf, mask_index, candidate_counts)
                commit_risk, anchor_mask = _compute_shared_anchor_commit_risk(
                    x=x,
                    p_curr=p_curr,
                    x0=x0,
                    mask_index=mask_index,
                    original_conf=curr_conf,
                    m=shared_anchor_m,
                    k=shared_anchor_k,
                    model=model,
                )
                ready_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)
                for j in range(candidate_mask.shape[0]):
                    candidate_indices = torch.where(candidate_mask[j])[0]
                    if len(candidate_indices) == 0:
                        continue
                    allowed_mask = candidate_mask[j].clone()
                    if anchor_mask is not None:
                        allowed_mask = allowed_mask & (~anchor_mask[j])
                    allowed_indices = torch.where(allowed_mask)[0]
                    if len(allowed_indices) == 0:
                        ready_mask[j, candidate_indices] = True
                        continue
                    rerank_scores = curr_conf[j, allowed_indices] - shared_anchor_rerank_lambda * commit_risk[j, allowed_indices]
                    keep_count = min(len(candidate_indices), len(allowed_indices))
                    _, rerank_rel = torch.topk(rerank_scores, k=keep_count)
                    selected_indices = allowed_indices[rerank_rel]
                    ready_mask[j, selected_indices] = True
                local_rerank_debug = None
            elif alg == "target_conditioned_local_commit_risk_rerank":
                if target_helper_selection_mode != "nearby_high_conf_stable":
                    raise ValueError("target_helper_selection_mode must be 'nearby_high_conf_stable'")
                if not use_target_conditioned_commit_risk:
                    raise ValueError("target_conditioned_local_commit_risk_rerank requires use_target_conditioned_commit_risk=True")
                reveal_budget_per_batch = num_transfer_tokens[:, step].tolist()
                ready_mask, local_rerank_debug = _target_conditioned_local_commit_risk_rerank(
                    x=x,
                    p_curr=p_curr,
                    x0=x0,
                    mask_index=mask_index,
                    confidence=curr_conf,
                    reveal_budget_per_batch=reveal_budget_per_batch,
                    candidate_multiplier=target_candidate_multiplier,
                    boundary_window=target_boundary_window,
                    helper_k=target_helper_k,
                    helper_direction=target_helper_direction,
                    lambda_r=shared_anchor_rerank_lambda,
                    model=model,
                )
                anchor_mask = None
            else:
                ready_mask = torch.zeros_like(curr_conf, dtype=torch.bool)
                anchor_mask = None
                local_rerank_debug = None

            # Select top-k tokens to unmask
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x.device)
            decoded_token_info = []
            history_gate_backbone_ready_mask = None

            # ── Optional top-1 history streak gate with raw-confidence escape ──
            if alg in ["confidence_threshold_history_gate", "confidence_threshold_tilg_history_gate", "confidence_threshold_tilg_history_gate_rerank_only", "confidence_threshold_tilg_history_gate_capped_extra"]:
                if top1_streak is None or prev_top1_for_streak is None:
                    top1_streak = torch.ones_like(x0, dtype=torch.long, device=x0.device)
                else:
                    same_top1 = (x0 == prev_top1_for_streak)
                    top1_streak = torch.where(
                        same_top1,
                        top1_streak + 1,
                        torch.ones_like(top1_streak),
                    )
                raw_conf = torch.squeeze(torch.gather(p_curr, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                history_gate_backbone_ready_mask = torch.zeros_like(ready_mask, dtype=torch.bool)
                for batch_idx in range(ready_mask.shape[0]):
                    candidate_indices = torch.where(ready_mask[batch_idx])[0]
                    if len(candidate_indices) > 0:
                        history_gate_backbone_ready_mask[batch_idx, candidate_indices] = True
                    else:
                        k_budget = int(num_transfer_tokens[batch_idx, step].item())
                        if k_budget > 0:
                            fallback_conf = torch.where(mask_index[batch_idx], curr_conf[batch_idx], -np.inf)
                            if not torch.isneginf(fallback_conf).all():
                                _, fallback_indices = torch.topk(fallback_conf, k=k_budget)
                                history_gate_backbone_ready_mask[batch_idx, fallback_indices] = True
                streak_ready = top1_streak >= int(history_gate_min_streak)
                escape_ready = raw_conf >= float(history_gate_confidence_escape)
                history_gate_ready = (streak_ready | escape_ready) & mask_index
                ready_mask = history_gate_backbone_ready_mask & history_gate_ready

            # ── Optional hard token-id stability gate (majority vote) ──
            if int(hard_token_gate_k) > 0:
                k = int(hard_token_gate_k)
                min_agree = int(hard_token_gate_min_agree) if int(hard_token_gate_min_agree) > 0 else max(1, (k + 1) // 2)
                # init / update history buffer: [B, L, K]
                if top1_hist is None or top1_hist.shape[-1] != k:
                    top1_hist = torch.full((x0_write.shape[0], x0_write.shape[1], k), -1, dtype=torch.long, device=x0_write.device)
                top1_hist = torch.roll(top1_hist, shifts=-1, dims=-1)
                top1_hist[..., -1] = x0_write
                # count agreement with current top1 across last K steps
                agree_count = (top1_hist == x0_write.unsqueeze(-1)).sum(dim=-1)
                token_stable_mask = agree_count >= min_agree
                token_stable_mask = token_stable_mask & mask_index
                ready_mask = ready_mask & token_stable_mask

            if step_save_dir:
                all_tokens_info = []
                top2_vals, _ = torch.topk(p_curr, k=2, dim=-1)
                top1_prob = top2_vals[..., 0]
                top2_prob = top2_vals[..., 1]
                margin_full = top1_prob - top2_prob
                log_conf_full = torch.squeeze(
                    torch.gather(p_curr, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
                if 'continuous_bonus' in locals() and isinstance(continuous_bonus, torch.Tensor):
                    tilg_bonus_full = continuous_bonus
                else:
                    tilg_bonus_full = torch.zeros_like(curr_conf)
                if top1_streak is not None:
                    streak_full = top1_streak
                else:
                    streak_full = torch.zeros_like(x0, dtype=torch.long)
                stable_mask_full = stable_mask if 'stable_mask' in locals() else torch.zeros_like(curr_conf, dtype=torch.bool)
                conf_mask_full = log_conf_full > float(conf_threshold)

                for j in range(mask_index.shape[0]):
                    masked_indices_in_block = torch.where(mask_index[j, block_start:block_end])[0] + block_start

                    for idx in masked_indices_in_block:
                        token_id = x0[j, idx].item()
                        conf_val = log_conf_full[j, idx].item()
                        all_tokens_info.append({
                            "position": int(idx),
                            "token_id": token_id,
                            "confidence": round(float(conf_val), 4),
                            "margin": round(float(margin_full[j, idx].item()), 6),
                            "tilg_bonus": round(float(tilg_bonus_full[j, idx].item()), 6),
                            "history_streak": int(streak_full[j, idx].item()),
                            "stable": bool(stable_mask_full[j, idx].item()),
                            "high_conf": bool(conf_mask_full[j, idx].item()),
                            "ready": bool(ready_mask[j, idx].item()),
                            "kl_divergence": "inf" if math.isinf(kl_current_prev[j, idx]) else round(float(kl_current_prev[j, idx]), 6)
                        })
                if local_rerank_debug is not None:
                    for batch_idx, debug_entry in enumerate(local_rerank_debug):
                        all_tokens_info.append({
                            "batch_idx": batch_idx,
                            "target_conditioned_local_commit_risk": debug_entry,
                        })

            for j in range(ready_mask.shape[0]):
                ready_indices = torch.where(ready_mask[j])[0]
                if len(ready_indices) > 0:
                    if len(ready_indices) > 1 and unmask_strategy != "all":
                        if unmask_strategy == "max_conf":
                            # Pick the one with highest confidence
                            conf_vals = curr_conf[j, ready_indices]
                            max_idx = torch.argmax(conf_vals)
                            selected_indices = ready_indices[max_idx:max_idx+1]
                        elif unmask_strategy == "min_kl":
                            # Pick the one with lowest KL divergence
                            kl_vals = kl_current_prev[j, ready_indices]
                            min_idx = torch.argmin(kl_vals)
                            selected_indices = ready_indices[min_idx:min_idx+1]
                        elif unmask_strategy == "random":
                            selected_indices = ready_indices[torch.randint(0, len(ready_indices), (1,))]
                        else:
                            selected_indices = ready_indices
                    else:
                        selected_indices = ready_indices
                    transfer_index[j, selected_indices] = True
                # If no tokens meet both criteria, select top-k by confidence
                else:
                    if history_gate_backbone_ready_mask is not None:
                        candidate_indices = torch.where(history_gate_backbone_ready_mask[j])[0]
                        if len(candidate_indices) > 0:
                            candidate_scores = curr_conf[j, candidate_indices]
                            max_idx = torch.argmax(candidate_scores)
                            selected_indices = candidate_indices[max_idx:max_idx + 1]
                            transfer_index[j, selected_indices] = True
                        elif step < steps_per_block - 1:
                            selected_indices = torch.empty(0, dtype=torch.long, device=x0.device)
                        else:
                            confidence = torch.where(mask_index, curr_conf, -np.inf)
                            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                            _, selected_indices = torch.topk(confidence[j], k=num_transfer_tokens[j, step].item())
                            transfer_index[j, selected_indices] = True
                    elif alg in ["confidence_threshold_tilg_rerank", "confidence_threshold_tilg_discrete"] and step < steps_per_block - 1:
                        selected_indices = torch.empty(0, dtype=torch.long, device=x0.device)
                    else:
                        curr_conf[:, input_ids_original.shape[1] + (num_block + 1) * block_length:] = -np.inf
                        # For tilg_klass_gate, fallback should prefer proposal positions first.
                        if alg == "tilg_klass_gate":
                            confidence = torch.where(mask_index & proposal_mask, curr_conf, -np.inf)
                            if torch.isneginf(confidence[j]).all():
                                confidence = torch.where(mask_index, curr_conf, -np.inf)
                        else:
                            confidence = torch.where(mask_index, curr_conf, -np.inf)
                        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                        _, selected_indices = torch.topk(confidence[j], k=num_transfer_tokens[j, step].item())
                        transfer_index[j, selected_indices] = True
                # If using original TILG, always commit scheduled budget (top-k by guided confidence).
                if alg == "tilg_original":
                    confidence = torch.where(mask_index, curr_conf, -np.inf)
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                    k_budget = int(num_transfer_tokens[j, step].item())
                    if k_budget > 0:
                        _, selected_indices = torch.topk(confidence[j], k=k_budget)
                        transfer_index[j, selected_indices] = True
                
                # Save info for each selected token
                if step_save_dir:
                    for idx in selected_indices:
                        token_id = x0[j, idx].item()
                        decoded_token = tokenizer.decode([token_id], skip_special_tokens=False)
                        conf_val = log_conf_full[j, idx].item()
                        decoded_token_info.append({
                            "position": int(idx),
                            "token_id": token_id,
                            "decoded_token": decoded_token,
                            "confidence": round(float(conf_val), 4),
                            "margin": round(float(margin_full[j, idx].item()), 6) if 'margin_full' in locals() else None,
                            "tilg_bonus": round(float(tilg_bonus_full[j, idx].item()), 6) if 'tilg_bonus_full' in locals() else 0.0,
                            "history_streak": int(streak_full[j, idx].item()) if 'streak_full' in locals() else 0,
                            "stable": bool(stable_mask_full[j, idx].item()) if 'stable_mask_full' in locals() else False,
                            "high_conf": bool(conf_mask_full[j, idx].item()) if 'conf_mask_full' in locals() else False,
                            "ready": bool(ready_mask[j, idx].item()),
                            "kl_divergence": "inf" if math.isinf(kl_current_prev[j, idx]) else round(float(kl_current_prev[j, idx]), 6),
                        })

            x[transfer_index] = x0_write[transfer_index]
            decoded_tokens_this_step = int(transfer_index.sum().item())
            total_decoded_tokens += decoded_tokens_this_step

            if step_save_dir:
                decoded_text = tokenizer.batch_decode(x[:, input_ids_original.shape[1]:], skip_special_tokens=True)[0]
                step_out = {
                    "step": used_steps + 1,
                    "decoded_text": decoded_text,
                    "decoded_tokens_num": decoded_tokens_this_step,
                    "decoded_tokens": decoded_token_info,
                    "all_tokens": all_tokens_info
                }
                all_step_outputs.append(step_out)

            used_steps += 1

            # Update temporal buffers AFTER completing this step.
            prev_step_logits = logits_fp64
            prev_step_x0 = x0.detach().clone()
            if alg in ["confidence_threshold_history_gate", "confidence_threshold_tilg_history_gate", "confidence_threshold_tilg_history_gate_rerank_only", "confidence_threshold_tilg_history_gate_capped_extra"]:
                prev_top1_for_streak = x0.detach().clone()
            if float(tilg_ema_decay) > 0:
                a = float(tilg_ema_decay)
                slow_ema_logits = a * slow_ema_logits + (1.0 - a) * logits_fp64

    if step_save_dir:
        all_steps_path = os.path.join(step_save_dir, f"all_steps_{example_idx}.json")
        with open(all_steps_path, "w") as f:
            json.dump(all_step_outputs, f, indent=2)

    decode_stats = {
        "total_decoded_tokens": int(total_decoded_tokens),
        "forward_steps": int(used_steps),
        "avg_tpf": float(total_decoded_tokens / used_steps) if used_steps > 0 else 0.0,
    }

    return x, used_steps, decode_stats