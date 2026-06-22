import os
import json
import torch
import numpy as np
from tqdm import tqdm
import random
import re
import argparse
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

from utils import *
from model.llada_klass import stable_confident_decode
from human_eval.data import read_problems, write_jsonl
from human_eval.evaluation import evaluate_functional_correctness


def test_dataset(
    model, tokenizer, save_dir, dataset,
    gen_length, steps, block_length,
    conf_threshold=0.9, kl_threshold=0.01, history_length=2,
    alg="confidence_threshold_tilg_history_gate_capped_extra", unmask_strategy="all",
    temperature=0.0,
    test_size=None, random_sampling=False,
    num_samples=1,
    save_steps=False,
    split="test",
    guidance_weight=0.3,
    tilg_ema_decay=0.95,
    tilg_rerank_lambda=0.05,
    tilg_extra_conf_floor=0.8,
    tilg_extra_ratio=0.25,
    tilg_extra_max=2,
    tilg_extra_allow_empty_base=1,
    history_gate_min_streak=3,
    history_gate_confidence_escape=0.95,
):
    supported_algs = {
        "confidence_threshold_tilg_history_gate_capped_extra",
        "klass_tilg_confidence",
    }
    if alg not in supported_algs:
        raise ValueError(f"Unsupported alg '{alg}'. Choose from {sorted(supported_algs)}.")

    save_base = f"{save_dir}/LLaDA/{dataset}/{alg}/{unmask_strategy}/len_{gen_length}_block_{block_length}/steps_{steps}"
    if alg == "confidence_threshold_tilg_history_gate_capped_extra":
        run_tag = (
            f"conf{conf_threshold}_floor{tilg_extra_conf_floor}"
            f"_extra{tilg_extra_max}_ratio{tilg_extra_ratio}_empty{tilg_extra_allow_empty_base}"
            f"_kl{kl_threshold}_ema{tilg_ema_decay}_gw{guidance_weight}_l{tilg_rerank_lambda}"
            f"_streak{history_gate_min_streak}_escape{history_gate_confidence_escape}_s{num_samples}"
        )
    else:
        run_tag = (
            f"conf{conf_threshold}_kl{kl_threshold}_ema{tilg_ema_decay}"
            f"_gw{guidance_weight}_l{tilg_rerank_lambda}_s{num_samples}"
        )
    save_dir = f"{save_base}/{run_tag}"
    os.makedirs(save_dir, exist_ok=True)
    step_save_dir = None
    if save_steps: 
        step_save_dir = os.path.join(save_dir, "stepwise")
        os.makedirs(step_save_dir, exist_ok=True)

    data_path = f"./data/{dataset}_{split}.json"
    data = process_file(data_path)

    if test_size:
        random.seed(516)
        data = random.sample(data, test_size) if random_sampling else data[:test_size]

    correct_count = 0
    used_steps_list = []
    tpf_list = []
    results = {"config": {
        "dataset": dataset,
        "alg": alg,
        "unmask_strategy": unmask_strategy,
        "gen_length": gen_length,
        "block_length": block_length,
        "steps": steps,
        "temperature": temperature,
        "test_size": test_size,
        "random_sampling": random_sampling,
        "num_samples": num_samples,
        "save_steps": save_steps,
        "split": split,
        "conf_threshold": conf_threshold,
        "kl_threshold": kl_threshold,
        "history_length": history_length,
        "guidance_weight": guidance_weight,
        "tilg_ema_decay": tilg_ema_decay,
        "tilg_rerank_lambda": tilg_rerank_lambda,
        "tilg_extra_conf_floor": tilg_extra_conf_floor,
        "tilg_extra_ratio": tilg_extra_ratio,
        "tilg_extra_max": tilg_extra_max,
        "tilg_extra_allow_empty_base": tilg_extra_allow_empty_base,
        "history_gate_min_streak": history_gate_min_streak,
        "history_gate_confidence_escape": history_gate_confidence_escape,
    }, "summary": {}, "results": []}

    for i, example in tqdm(enumerate(data), total=len(data), desc=f"Generating completions for {dataset.capitalize()}"):
        if dataset == "gsm8k":
            prompt = example['question']
            answer = example['answer']
            ground_truth_answer = parse_ground_truth_answer(answer)
        elif dataset == "math":
            prompt = example['problem']
            solution = example['solution']
            ground_truth_answer = extract_math_answer(prompt, solution)

        example_samples = []
        example_correct = False
        example_steps = []

        m = [
            {"role": "system", "content": "Your task is to answer the question below. Give step by step reasoning before you answer, and when you're ready to answer, please use the format 'The final answer is'."},
            {"role": "user", "content": prompt}
        ]
        prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids_original = torch.tensor(tokenizer(prompt)['input_ids']).to("cuda").unsqueeze(0)
        
        for sample_idx in range(num_samples):
            x_output, used_steps, decode_stats = stable_confident_decode(
                model, tokenizer, input_ids_original, gen_length, steps, block_length,
                temperature=temperature, mask_id=126336,
                conf_threshold=conf_threshold, kl_threshold=kl_threshold, kl_history_length=history_length,
                step_save_dir=step_save_dir, example_idx=f"q{i}_s{sample_idx}",
                alg=alg,
                unmask_strategy=unmask_strategy,
                guidance_weight=guidance_weight,
                tilg_ema_decay=tilg_ema_decay,
                tilg_rerank_lambda=tilg_rerank_lambda,
                tilg_extra_conf_floor=tilg_extra_conf_floor,
                tilg_extra_ratio=tilg_extra_ratio,
                tilg_extra_max=tilg_extra_max,
                tilg_extra_allow_empty_base=tilg_extra_allow_empty_base,
                history_gate_min_streak=history_gate_min_streak,
                history_gate_confidence_escape=history_gate_confidence_escape,
            )

            generated_text = tokenizer.batch_decode(x_output[:, input_ids_original.shape[1]:], skip_special_tokens=True)[0]
            
            if dataset == "gsm8k":
                generated_answer = parse_answer(generated_text)
                is_correct = generated_answer == ground_truth_answer
            elif dataset == "math":
                generated_answer = extract_math_answer(prompt, generated_text)
                is_correct = compare_answers(prompt, ground_truth_answer, generated_answer)

            if is_correct:
                example_correct = True

            example_steps.append(used_steps)
            tpf_list.append(decode_stats["avg_tpf"])
            example_samples.append({
                "task_id": i,
                "sample_idx": sample_idx,
                "used_steps": used_steps,
                "tpf": decode_stats["avg_tpf"],
                "total_decoded_tokens": decode_stats["total_decoded_tokens"],
                "forward_steps": decode_stats["forward_steps"],
                "generation": generated_text,
                "parsed_answer": generated_answer,
                "is_correct": is_correct
            })

        if example_correct:
            correct_count += 1

        used_steps_list.extend(example_steps)
        results["results"].append({
            "task_id": i,
            "input_prompt": prompt,
            "ground_truth_answer": ground_truth_answer,
            "any_correct": example_correct,
            "avg_steps": round(sum(example_steps) / len(example_steps), 2),
            "samples": example_samples
        })

    accuracy = correct_count / len(data)
    avg_steps = sum(used_steps_list) / len(used_steps_list)
    avg_tpf = sum(tpf_list) / len(tpf_list) if tpf_list else 0.0
    results["summary"] = {
        "accuracy": round(accuracy*100, 2),
        "average_steps": round(avg_steps, 2),
        "average_tpf": round(avg_tpf, 4),
        "total_questions": len(data),
        "correct_questions": correct_count,
        "num_samples_per_question": num_samples
    }
    
    save_path = f"{save_dir}/all_results.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"[{dataset.capitalize()}]")
    print(f"Accuracy: {round(accuracy*100, 2)}")
    print(f"Average steps: {round(avg_steps, 2)}")
    print(f"Average TPF: {round(avg_tpf, 4)}")
    print(f"Results saved to {save_path}")

    
def test_humaneval(
    model, tokenizer, save_dir,
    gen_length, steps, block_length, 
    conf_threshold=0.9, kl_threshold=0.01, history_length=2,
    alg="klass", unmask_strategy="all",
    temperature=0.0,
    k=(1,10,100), n_workers=4, timeout=3.0,
    test_size=None, random_sampling=False,
    num_samples=1, 
    save_steps=False,
    guidance_weight=1.5,
    tilg_proposal_ratio=0.1,
    tilg_ema_decay=0.0,
    tilg_rerank_lambda=0.2,
    tilg_boundary_window=2,
    tilg_extra_conf_floor=0.85,
    tilg_extra_ratio=0.25,
    tilg_extra_max=2,
    tilg_extra_allow_empty_base=1,
    hard_token_gate_k=0,
    hard_token_gate_min_agree=0,
    history_gate_min_streak=0,
    history_gate_confidence_escape=1.0,
    history_gate_max_wait=0,
):
    if alg in ['klass', 'confidence_threshold']:
        save_dir = f"{save_dir}/LLaDA/humaneval/{alg}/{unmask_strategy}/len_{gen_length}_block_{block_length}/steps_{steps}/conf{conf_threshold}_kl{kl_threshold}_s{num_samples}"
    elif alg in ['confidence_threshold_tilg_history_gate', 'confidence_threshold_tilg_history_gate_rerank_only']:
        save_dir = f"{save_dir}/LLaDA/humaneval/{alg}/{unmask_strategy}/len_{gen_length}_block_{block_length}/steps_{steps}/conf{conf_threshold}_kl{kl_threshold}_ema{tilg_ema_decay}_gw{guidance_weight}_l{tilg_rerank_lambda}_streak{history_gate_min_streak}_escape{history_gate_confidence_escape}_s{num_samples}"
    elif alg == 'confidence_threshold_tilg_history_gate_capped_extra':
        save_dir = f"{save_dir}/LLaDA/humaneval/{alg}/{unmask_strategy}/len_{gen_length}_block_{block_length}/steps_{steps}/conf{conf_threshold}_floor{tilg_extra_conf_floor}_extra{tilg_extra_max}_ratio{tilg_extra_ratio}_empty{tilg_extra_allow_empty_base}_kl{kl_threshold}_ema{tilg_ema_decay}_gw{guidance_weight}_l{tilg_rerank_lambda}_streak{history_gate_min_streak}_escape{history_gate_confidence_escape}_s{num_samples}"
    elif alg in ['confidence_threshold_tilg', 'confidence_threshold_tilg_discrete', 'confidence_threshold_tilg_added_gate', 'klass_tilg_confidence']:
        save_dir = f"{save_dir}/LLaDA/humaneval/{alg}/{unmask_strategy}/len_{gen_length}_block_{block_length}/steps_{steps}/conf{conf_threshold}_kl{kl_threshold}_ema{tilg_ema_decay}_gw{guidance_weight}_l{tilg_rerank_lambda}_s{num_samples}"
    elif alg == 'confidence_threshold_history_gate':
        save_dir = f"{save_dir}/LLaDA/humaneval/{alg}/{unmask_strategy}/len_{gen_length}_block_{block_length}/steps_{steps}/conf{conf_threshold}_kl{kl_threshold}_streak{history_gate_min_streak}_escape{history_gate_confidence_escape}_s{num_samples}"
    else:
        save_dir = f"{save_dir}/LLaDA/humaneval/{alg}/{unmask_strategy}/len_{gen_length}_block_{block_length}/steps_{steps}/s{num_samples}"
    os.makedirs(save_dir, exist_ok=True)
    step_save_dir = None
    if save_steps: 
        step_save_dir = os.path.join(save_dir, "stepwise")
        os.makedirs(step_save_dir, exist_ok=True)

    problems = read_problems()
    # By default, evaluate on the full HumanEval set. If test_size is provided,
    # optionally subsample below.
    data = problems

    if test_size:
        random.seed(516)
        problems = list(problems.items())
        data = random.sample(problems, test_size) if random_sampling else problems[:test_size]
        data = dict(data)

    samples = []
    steps_per_problem = []
    i = 0
    for task_id, info in tqdm(data.items(), desc="Generating completions for HumanEval"):
        prompt = info["prompt"]
        task_samples = []
        task_steps = []
        task_tpfs = []

        m = [
            {"role": "system", "content": "You complete only Python code."},
            {"role": "user", "content": prompt}
        ]
        prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids_original = torch.tensor(tokenizer(prompt)['input_ids']).to("cuda").unsqueeze(0)
        
        for sample_idx in range(num_samples):
            x_output, used_steps, decode_stats = stable_confident_decode(
                model, tokenizer, input_ids_original, gen_length, steps, block_length,
                temperature=temperature, mask_id=126336,
                conf_threshold=conf_threshold, kl_threshold=kl_threshold, kl_history_length=history_length,
                step_save_dir=step_save_dir, example_idx=f"q{i}_s{sample_idx}",
                alg=alg,
                unmask_strategy=unmask_strategy,
                guidance_weight=guidance_weight,
                tilg_ema_decay=tilg_ema_decay,
                tilg_rerank_lambda=tilg_rerank_lambda,
                tilg_extra_conf_floor=tilg_extra_conf_floor,
                tilg_extra_ratio=tilg_extra_ratio,
                tilg_extra_max=tilg_extra_max,
                tilg_extra_allow_empty_base=tilg_extra_allow_empty_base,
                history_gate_min_streak=history_gate_min_streak,
                history_gate_confidence_escape=history_gate_confidence_escape,
            )
            
            task_steps.append(used_steps)
            task_tpfs.append(decode_stats["avg_tpf"])
            generations = tokenizer.batch_decode(x_output[:, input_ids_original.shape[1]:], skip_special_tokens=True)
            decoded_text = generations[0].split(tokenizer.eos_token)[0]

            code_match = re.search(r"```(?:python)?\n(.*?)(?:```|$)", decoded_text, re.DOTALL)
            code_only = code_match.group(1).strip() if code_match else decoded_text.strip()
            
            sample_data = {
                "task_id": task_id, 
                "sample_idx": sample_idx,
                "completion": code_only, 
                "used_steps": used_steps,
                "tpf": decode_stats["avg_tpf"],
                "total_decoded_tokens": decode_stats["total_decoded_tokens"],
                "forward_steps": decode_stats["forward_steps"],
            }
            task_samples.append(sample_data)
            samples.append(sample_data)

        steps_per_problem.append({
            "task_id": task_id, 
            "input_prompt": prompt,
            "avg_steps": sum(task_steps) / len(task_steps),
            "avg_tpf": sum(task_tpfs) / len(task_tpfs) if task_tpfs else 0.0,
            "samples": task_samples
        })
        i += 1

    samples_file = os.path.join(save_dir, "humaneval_samples.jsonl")
    write_jsonl(samples_file, samples)

    results = evaluate_functional_correctness(
        samples_file,
        k=",".join(map(str, k)),
        n_workers=n_workers,
        timeout=timeout
    )

    avg_steps = sum([entry["avg_steps"] for entry in steps_per_problem]) / len(steps_per_problem)
    avg_tpf = sum([entry["avg_tpf"] for entry in steps_per_problem]) / len(steps_per_problem)

    all_results = {
        "config": {
            "dataset": "humaneval",
            "alg": alg,
            "unmask_strategy": unmask_strategy,
            "gen_length": gen_length,
            "block_length": block_length,
            "steps": steps,
            "temperature": temperature,
            "test_size": test_size,
            "random_sampling": random_sampling,
            "num_samples": num_samples,
            "save_steps": save_steps,
            "conf_threshold": conf_threshold,
            "kl_threshold": kl_threshold,
            "history_length": history_length,
            "guidance_weight": guidance_weight,
            "tilg_proposal_ratio": tilg_proposal_ratio,
            "tilg_ema_decay": tilg_ema_decay,
            "tilg_rerank_lambda": tilg_rerank_lambda,
            "tilg_boundary_window": tilg_boundary_window,
            "tilg_extra_conf_floor": tilg_extra_conf_floor,
            "tilg_extra_ratio": tilg_extra_ratio,
            "tilg_extra_max": tilg_extra_max,
            "tilg_extra_allow_empty_base": tilg_extra_allow_empty_base,
            "hard_token_gate_k": hard_token_gate_k,
            "hard_token_gate_min_agree": hard_token_gate_min_agree,
            "history_gate_min_streak": history_gate_min_streak,
            "history_gate_confidence_escape": history_gate_confidence_escape,
            "history_gate_max_wait": history_gate_max_wait,
        }
    }
    all_results["summary"] = {
        "accuracy": round(results['pass@1']*100, 2),
        "average_steps": avg_steps,
        "average_tpf": avg_tpf,
    }
    all_results["results"] = steps_per_problem

    save_path = os.path.join(save_dir, "all_results.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print('[HumanEval]')
    print("Accuracy:", round(results['pass@1']*100, 2))
    print(f"Average steps: {avg_steps}")
    print(f"Average TPF: {avg_tpf}")
    print(f"Results saved to {save_path}")


def test_mbpp(
    model, tokenizer, save_dir,
    gen_length, steps, block_length,
    conf_threshold=0.9, kl_threshold=0.01, history_length=2,
    alg="klass", unmask_strategy="all",
    temperature=0.0,
    eval_timeout=3.0,
    test_size=None, random_sampling=False,
    num_samples=1, 
    save_steps=False,
    guidance_weight=1.5,
    tilg_proposal_ratio=0.1,
    tilg_ema_decay=0.0,
    tilg_rerank_lambda=0.2,
    tilg_boundary_window=2,
    tilg_extra_conf_floor=0.85,
    tilg_extra_ratio=0.25,
    tilg_extra_max=2,
    tilg_extra_allow_empty_base=1,
    hard_token_gate_k=0,
    hard_token_gate_min_agree=0,
    history_gate_min_streak=0,
    history_gate_confidence_escape=1.0,
    history_gate_max_wait=0,
):
    if alg == 'klass':
        save_dir = f"{save_dir}/LLaDA/mbpp/{alg}/{unmask_strategy}/len_{gen_length}/steps_{steps}/conf{conf_threshold}_kl{kl_threshold}_s{num_samples}"
    else:
        save_dir = f"{save_dir}/LLaDA/mbpp/{alg}/{unmask_strategy}/len_{gen_length}/steps_{steps}/s{num_samples}"
    os.makedirs(save_dir, exist_ok=True)

    if save_steps: 
        step_save_dir = os.path.join(save_dir, "stepwise")
        os.makedirs(step_save_dir, exist_ok=True)

    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")

    if test_size:
        random.seed(516)
        ds = random.sample(list(ds), test_size) if random_sampling else ds.select(range(test_size))

    steps_per_problem = []

    for idx, ex in enumerate(tqdm(ds, desc="Generating completions for MBPP")):
        task_id = ex["task_id"]
        prompt = ex["prompt"]
        tests = ex["test_list"]
        code = ex["code"]

        task_samples = []
        task_steps = []
        task_tpfs = []
        task_passed = False

        m = [
            {"role": "user", "content": f"You are an expert Python programmer, and here is your task: {prompt} Your code should pass these tests:\n\n{tests}\n[BEGIN]"}
        ]
        prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids_original = torch.tensor(tokenizer(prompt)['input_ids']).to("cuda").unsqueeze(0)

        for sample_idx in range(num_samples):
            x_output, used_steps, decode_stats = stable_confident_decode(
                model, tokenizer, input_ids_original, gen_length, steps, block_length,
                temperature=temperature, mask_id=126336,
                conf_threshold=conf_threshold, kl_threshold=kl_threshold, kl_history_length=history_length,
                step_save_dir=step_save_dir, example_idx=f"q{idx}_s{sample_idx}",
                alg=alg,
                unmask_strategy=unmask_strategy,
                guidance_weight=guidance_weight,
                tilg_ema_decay=tilg_ema_decay,
                tilg_rerank_lambda=tilg_rerank_lambda,
                tilg_extra_conf_floor=tilg_extra_conf_floor,
                tilg_extra_ratio=tilg_extra_ratio,
                tilg_extra_max=tilg_extra_max,
                tilg_extra_allow_empty_base=tilg_extra_allow_empty_base,
                history_gate_min_streak=history_gate_min_streak,
                history_gate_confidence_escape=history_gate_confidence_escape,
            )
            generations = tokenizer.batch_decode(x_output[:, input_ids_original.shape[1]:], skip_special_tokens=True)
            decoded_text = generations[0].split(tokenizer.eos_token)[0]

            sample_data = {
                "task_id": task_id,
                "sample_idx": sample_idx,
                "used_steps": used_steps,
                "tpf": decode_stats["avg_tpf"],
                "total_decoded_tokens": decode_stats["total_decoded_tokens"],
                "forward_steps": decode_stats["forward_steps"],
                "generation": decoded_text
            }
            
            # Evaluate this sample
            passed = evaluate_task(sample_data, tests, timeout=eval_timeout)
            if passed:
                task_passed = True
            sample_data["passed"] = passed
            task_samples.append(sample_data)
            task_steps.append(used_steps)
            task_tpfs.append(decode_stats["avg_tpf"])

        steps_per_problem.append({
            "task_id": task_id,
            "input_prompt": prompt,
            "solution_code": code,
            "any_passed": task_passed,
            "avg_steps": sum(task_steps) / len(task_steps),
            "avg_tpf": sum(task_tpfs) / len(task_tpfs) if task_tpfs else 0.0,
            "samples": task_samples
        })

    total_passed_tasks = sum(1 for task_data in steps_per_problem if task_data["any_passed"])
    accuracy = total_passed_tasks / len(steps_per_problem) if steps_per_problem else 0
    average_steps = sum([entry["avg_steps"] for entry in steps_per_problem]) / len(steps_per_problem)
    average_tpf = sum([entry["avg_tpf"] for entry in steps_per_problem]) / len(steps_per_problem)

    all_results = {}
    all_results["summary"] = {
        "accuracy": round(accuracy*100, 2),
        "average_steps": average_steps,
        "average_tpf": average_tpf,
        "total_tasks": len(steps_per_problem),
        "passed_tasks": total_passed_tasks,
        "num_samples_per_task": num_samples
    }
    all_results["results"] = steps_per_problem

    samples_file = os.path.join(save_dir, "all_results.json")
    with open(samples_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"[MBPP]")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Average steps: {average_steps}")
    print(f"Average TPF: {average_tpf}")
    print(f"Results saved to {samples_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--dataset', type=str, required=True, choices=['gsm8k', 'math', 'humaneval', 'mbpp'], help='Dataset to use')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--gen_length', type=int, default=256, help='Generation length')
    parser.add_argument('--block_length', type=int, default=64, help='Block length')
    parser.add_argument('--steps', type=int, default=256, help='Number of steps')
    parser.add_argument('--conf_threshold', type=float, default=0.9, help='Confidence threshold')
    parser.add_argument('--kl_threshold', type=float, default=0.01, help='KL threshold')
    parser.add_argument('--history_length', type=int, default=2, help='History length for KL calculation')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--test_size', type=int, default=None, help='Size of the test set')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'train'], help='Dataset split for file-based datasets (gsm8k/math)')
    parser.add_argument('--random_sampling', action='store_true', help='Enable random sampling')
    parser.add_argument(
        '--alg',
        type=str,
        default='confidence_threshold_tilg_history_gate_capped_extra',
        choices=[
            'confidence_threshold_tilg_history_gate_capped_extra',
            'klass_tilg_confidence',
        ],
        help='Algorithm: capped_extra (Conf.+TACG) or klass_tilg_confidence (KLASS+TACG on gsm8k)',
    )
    parser.add_argument('--unmask_strategy', type=str, default='all', choices=['all', 'max_conf', 'min_kl', 'random'], help='Unmasking strategy')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to generate per question/task')
    parser.add_argument('--save_steps', action='store_true', help='Save the results of each step')
    parser.add_argument('--guidance_weight', type=float, default=0.3, help='Guidance weight for HG+TILG')
    parser.add_argument('--tilg_ema_decay', type=float, default=0.95, help='EMA decay for temporal logits proxy')
    parser.add_argument('--tilg_rerank_lambda', type=float, default=0.05, help='Continuous TILG bonus weight')
    parser.add_argument('--tilg_extra_conf_floor', type=float, default=0.8, help='Lower confidence floor for capped extra candidates')
    parser.add_argument('--tilg_extra_ratio', type=float, default=0.25, help='Extra candidate budget ratio')
    parser.add_argument('--tilg_extra_max', type=int, default=2, help='Maximum capped extra candidates per step')
    parser.add_argument('--tilg_extra_allow_empty_base', type=int, default=1, help='Allow capped extra when base-ready set is empty')
    parser.add_argument('--history_gate_min_streak', type=int, default=3, help='Minimum top-1 streak for History Gate')
    parser.add_argument('--history_gate_confidence_escape', type=float, default=0.95, help='Raw confidence escape threshold for History Gate')
    # HumanEval-specific arguments
    parser.add_argument('--humaneval_k', type=str, default='1', help='Values of k for HumanEval pass@k')
    parser.add_argument('--humaneval_workers', type=int, default=4, help='Number of workers for HumanEval evaluation')
    parser.add_argument('--humaneval_timeout', type=float, default=3.0, help='Timeout for each test case in HumanEval evaluation')
    args = parser.parse_args()

    print("Parsed arguments:", args)

    device = 'cuda'

    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if args.dataset in ['gsm8k', 'math']:
        test_dataset(
            model, tokenizer, 
            save_dir=args.save_dir,
            dataset=args.dataset, 
            gen_length=args.gen_length, 
            steps=args.steps, 
            block_length=args.block_length, 
            conf_threshold=args.conf_threshold, 
            kl_threshold=args.kl_threshold, 
            history_length=args.history_length,
            temperature=args.temperature,
            alg=args.alg,
            unmask_strategy=args.unmask_strategy,
            guidance_weight=args.guidance_weight,
            tilg_ema_decay=args.tilg_ema_decay,
            tilg_rerank_lambda=args.tilg_rerank_lambda,
            tilg_extra_conf_floor=args.tilg_extra_conf_floor,
            tilg_extra_ratio=args.tilg_extra_ratio,
            tilg_extra_max=args.tilg_extra_max,
            tilg_extra_allow_empty_base=args.tilg_extra_allow_empty_base,
            history_gate_min_streak=args.history_gate_min_streak,
            history_gate_confidence_escape=args.history_gate_confidence_escape,
            test_size=args.test_size, 
            random_sampling=args.random_sampling,
            num_samples=args.num_samples,
            save_steps=args.save_steps,
            split=args.split,
        )
    elif args.dataset == 'humaneval':
        ks = tuple(map(int, args.humaneval_k.split(',')))
        test_humaneval(
            model, tokenizer, 
            save_dir=args.save_dir,
            gen_length=args.gen_length,
            block_length=args.block_length,
            steps=args.steps,
            conf_threshold=args.conf_threshold,
            kl_threshold=args.kl_threshold,
            history_length=args.history_length,
            temperature=args.temperature,
            alg=args.alg,
            unmask_strategy=args.unmask_strategy,
            guidance_weight=args.guidance_weight,
            tilg_ema_decay=args.tilg_ema_decay,
            tilg_rerank_lambda=args.tilg_rerank_lambda,
            tilg_extra_conf_floor=args.tilg_extra_conf_floor,
            tilg_extra_ratio=args.tilg_extra_ratio,
            tilg_extra_max=args.tilg_extra_max,
            tilg_extra_allow_empty_base=args.tilg_extra_allow_empty_base,
            history_gate_min_streak=args.history_gate_min_streak,
            history_gate_confidence_escape=args.history_gate_confidence_escape,
            k=ks,
            n_workers=args.humaneval_workers,
            timeout=args.humaneval_timeout,
            test_size=args.test_size, 
            random_sampling=args.random_sampling,
            num_samples=args.num_samples,
            save_steps=args.save_steps,
        )
    elif args.dataset == 'mbpp':
        test_mbpp(
            model, tokenizer, 
            save_dir=args.save_dir,
            gen_length=args.gen_length,
            block_length=args.block_length,
            steps=args.steps,
            conf_threshold=args.conf_threshold,
            kl_threshold=args.kl_threshold,
            history_length=args.history_length,
            temperature=args.temperature,
            alg=args.alg,
            unmask_strategy=args.unmask_strategy,
            guidance_weight=args.guidance_weight,
            tilg_ema_decay=args.tilg_ema_decay,
            tilg_rerank_lambda=args.tilg_rerank_lambda,
            tilg_extra_conf_floor=args.tilg_extra_conf_floor,
            tilg_extra_ratio=args.tilg_extra_ratio,
            tilg_extra_max=args.tilg_extra_max,
            tilg_extra_allow_empty_base=args.tilg_extra_allow_empty_base,
            history_gate_min_streak=args.history_gate_min_streak,
            history_gate_confidence_escape=args.history_gate_confidence_escape,
            test_size=args.test_size,
            random_sampling=args.random_sampling,
            num_samples=args.num_samples,
            save_steps=args.save_steps,
        )


if __name__ == '__main__':
    main()
