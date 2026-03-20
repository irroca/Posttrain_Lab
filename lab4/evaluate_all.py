#!/usr/bin/env python3
"""
Lab 4: Four-way GSM8K evaluation and reasoning chain analysis.
Base vs GRPO vs Distill vs Qwen3-Instruct-Think
"""
import os
import re
import json
import torch
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_gsm8k_answer(answer_text: str) -> str:
    match = re.search(r'####\s*(.+)', answer_text)
    if match:
        return match.group(1).strip().replace(",", "")
    return ""


def extract_answer_from_response(response: str) -> str:
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
    if boxed_match:
        return boxed_match.group(1).strip().replace(",", "")
    answer_match = re.search(
        r'(?:答案|结果|answer|result)\s*(?:是|为|=|:)\s*(-?\d+\.?\d*)',
        response, re.IGNORECASE
    )
    if answer_match:
        return answer_match.group(1).strip()
    hash_match = re.search(r'####\s*(-?\d+\.?\d*)', response)
    if hash_match:
        return hash_match.group(1).strip()
    numbers = re.findall(r'-?\d+\.?\d*', response)
    if numbers:
        return numbers[-1].replace(",", "")
    return ""


def generate_response(model, tokenizer, question, use_think=False, max_new_tokens=1024):
    """Generate response from a model."""
    if use_think:
        messages = [{"role": "user", "content":
            f"Solve: {question}\nPut your answer in \\boxed{{}}."}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
    else:
        prompt = (
            f"Solve the following math problem step by step. "
            f"Put your final answer in \\boxed{{}}.\n\n"
            f"Problem: {question}\n\nSolution:"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
        )
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return response


def check_correct(predicted, ground_truth):
    try:
        return abs(float(predicted) - float(ground_truth)) < 1e-5
    except (ValueError, TypeError):
        return predicted.strip() == ground_truth.strip()


def evaluate_model(model, tokenizer, test_data, name, use_think=False, max_samples=50):
    """Evaluate a model on GSM8K test subset."""
    correct = 0
    total = 0
    details = []
    for idx, sample in enumerate(test_data):
        if idx >= max_samples:
            break
        question = sample["question"]
        ground_truth = extract_gsm8k_answer(sample["answer"])
        response = generate_response(model, tokenizer, question, use_think=use_think)
        predicted = extract_answer_from_response(response)
        is_correct = check_correct(predicted, ground_truth)
        total += 1
        if is_correct:
            correct += 1
        details.append({
            "idx": idx,
            "question": question,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "correct": is_correct,
            "response": response[:2000],
        })
        if (idx + 1) % 10 == 0:
            print(f"  [{name}] {idx+1}/{min(max_samples, len(test_data))} done, "
                  f"accuracy so far: {correct}/{total} ({correct/total*100:.1f}%)")

    acc = correct / total * 100 if total > 0 else 0
    print(f"  [{name}] Final: {correct}/{total} = {acc:.1f}%")
    return {"name": name, "correct": correct, "total": total, "accuracy": acc, "details": details}


def analyze_emergence(details, name):
    """Analyze emergent reasoning behaviors in model responses."""
    checklist = {
        "step_by_step": ["step", "first", "then", "next", "首先", "然后", "接下来"],
        "self_check": ["verify", "check", "let me", "确认", "验证", "检查"],
        "backtrack": ["wait", "actually", "reconsider", "mistake", "不对", "重新"],
        "structured": ["therefore", "so", "thus", "because", "因此", "所以"],
        "math_symbols": ["\\boxed", "=", "+", "-"],
    }
    behavior_counts = {k: 0 for k in checklist}
    for d in details:
        resp = d["response"].lower()
        for behavior, keywords in checklist.items():
            if any(kw.lower() in resp for kw in keywords):
                behavior_counts[behavior] += 1
    total = len(details)
    return {k: f"{v}/{total}" for k, v in behavior_counts.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grpo-model-dir", type=str, required=True,
                        help="Path to trained GRPO model (final dir)")
    parser.add_argument("--output-dir", type=str, default="./lab4_outputs")
    parser.add_argument("--num-eval-samples", type=int, default=50)
    parser.add_argument("--num-chain-examples", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0")

    gsm8k = load_dataset("openai/gsm8k", "main")
    test_data = gsm8k["test"]

    all_results = {}
    all_details = {}

    # ── 1) Base model ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Evaluating: Qwen3-1.7B-Base")
    print("=" * 60)
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B-Base",
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="flash_attention_2",
    )
    base_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    result = evaluate_model(base_model, base_tokenizer, test_data, "Base",
                           max_samples=args.num_eval_samples)
    all_results["Base"] = {"correct": result["correct"], "total": result["total"],
                           "accuracy": result["accuracy"]}
    all_details["Base"] = result["details"]
    # Get reasoning chain examples for base
    del base_model
    torch.cuda.empty_cache()

    # ── 2) GRPO model ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Evaluating: GRPO model from {args.grpo_model_dir}")
    print("=" * 60)
    grpo_model = AutoModelForCausalLM.from_pretrained(
        args.grpo_model_dir,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="flash_attention_2",
    )
    grpo_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
    if grpo_tokenizer.pad_token is None:
        grpo_tokenizer.pad_token = grpo_tokenizer.eos_token
    result = evaluate_model(grpo_model, grpo_tokenizer, test_data, "GRPO",
                           max_samples=args.num_eval_samples)
    all_results["GRPO"] = {"correct": result["correct"], "total": result["total"],
                           "accuracy": result["accuracy"]}
    all_details["GRPO"] = result["details"]
    del grpo_model
    torch.cuda.empty_cache()

    # ── 3) Distill model ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Evaluating: DeepSeek-R1-Distill-Qwen-1.5B")
    print("=" * 60)
    distill_model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="flash_attention_2",
    )
    distill_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    if distill_tokenizer.pad_token is None:
        distill_tokenizer.pad_token = distill_tokenizer.eos_token
    result = evaluate_model(distill_model, distill_tokenizer, test_data, "Distill",
                           max_samples=args.num_eval_samples)
    all_results["Distill"] = {"correct": result["correct"], "total": result["total"],
                              "accuracy": result["accuracy"]}
    all_details["Distill"] = result["details"]
    del distill_model
    torch.cuda.empty_cache()

    # ── 4) Qwen3 Instruct (think mode) ───────────────────────────────────
    print("\n" + "=" * 60)
    print("Evaluating: Qwen3-1.7B Instruct (Think mode)")
    print("=" * 60)
    instruct_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B",
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="flash_attention_2",
    )
    instruct_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    if instruct_tokenizer.pad_token is None:
        instruct_tokenizer.pad_token = instruct_tokenizer.eos_token
    result = evaluate_model(instruct_model, instruct_tokenizer, test_data, "Instruct-Think",
                           use_think=True, max_samples=args.num_eval_samples)
    all_results["Instruct-Think"] = {"correct": result["correct"], "total": result["total"],
                                     "accuracy": result["accuracy"]}
    all_details["Instruct-Think"] = result["details"]
    del instruct_model
    torch.cuda.empty_cache()

    # ── Save all results ──────────────────────────────────────────────────
    results_path = os.path.join(args.output_dir, "gsm8k_comparison.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ── Save reasoning chain examples ─────────────────────────────────────
    chain_examples = []
    n_examples = min(args.num_chain_examples, len(all_details.get("Base", [])))
    for i in range(n_examples):
        example = {"question": all_details["Base"][i]["question"],
                   "ground_truth": all_details["Base"][i]["ground_truth"]}
        for model_name in ["Base", "GRPO", "Distill", "Instruct-Think"]:
            if model_name in all_details and i < len(all_details[model_name]):
                d = all_details[model_name][i]
                example[model_name] = {
                    "response": d["response"],
                    "predicted": d["predicted"],
                    "correct": d["correct"],
                }
        chain_examples.append(example)

    chains_path = os.path.join(args.output_dir, "reasoning_chains.json")
    with open(chains_path, "w") as f:
        json.dump(chain_examples, f, indent=2, ensure_ascii=False)
    print(f"Reasoning chains saved to {chains_path}")

    # ── Save emergence analysis ───────────────────────────────────────────
    emergence = {}
    for name, details in all_details.items():
        emergence[name] = analyze_emergence(details, name)
    emergence_path = os.path.join(args.output_dir, "emergence_analysis.json")
    with open(emergence_path, "w") as f:
        json.dump(emergence, f, indent=2)
    print(f"Emergence analysis saved to {emergence_path}")

    # ── Print comparison table ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"GSM8K Accuracy Comparison ({args.num_eval_samples} samples)")
    print("=" * 60)
    print(f"{'Model':<25} {'Correct':<10} {'Accuracy':<10}")
    print("-" * 45)
    for name, r in all_results.items():
        print(f"{name:<25} {r['correct']}/{r['total']:<10} {r['accuracy']:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
