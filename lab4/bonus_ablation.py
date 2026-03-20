#!/usr/bin/env python3
"""
Lab 4 Bonus: Ablation experiments for G (num_generations) and KL coefficient.
Runs shorter training (100 steps) with different hyperparameters.
"""
import os
import re
import json
import torch
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


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


def math_reward_fn(completions, answer, **kwargs):
    rewards = []
    for completion, ans_text in zip(completions, answer):
        ground_truth = extract_gsm8k_answer(ans_text)
        predicted = extract_answer_from_response(completion)
        try:
            correct = abs(float(predicted) - float(ground_truth)) < 1e-5
        except (ValueError, TypeError):
            correct = predicted.strip() == ground_truth.strip()
        rewards.append(1.0 if correct else 0.0)
    return rewards


def format_gsm8k_for_grpo(example):
    prompt = (
        f"Solve the following math problem step by step. "
        f"Put your final answer in \\boxed{{}}.\n\n"
        f"Problem: {example['question']}\n\n"
        f"Solution:"
    )
    return {"prompt": prompt, "answer": example["answer"]}


def run_ablation(G, beta, train_dataset, output_base, max_steps=100):
    """Run a single ablation experiment."""
    exp_name = f"G{G}_beta{beta}"
    output_dir = os.path.join(output_base, exp_name)
    print(f"\n{'='*60}")
    print(f"Ablation: G={G}, beta={beta}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B-Base",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = GRPOConfig(
        output_dir=output_dir,
        num_generations=G,
        max_completion_length=1024,
        beta=beta,
        learning_rate=5e-6,
        per_device_train_batch_size=G,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        max_steps=max_steps,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        logging_steps=1,
        save_steps=max_steps,  # save only at end
        save_total_limit=1,
        seed=42,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=math_reward_fn,
    )

    result = trainer.train()
    log_history = trainer.state.log_history

    # Extract final metrics
    final_reward = 0.0
    final_kl = 0.0
    for entry in reversed(log_history):
        if "reward/mean" in entry or "rewards/mean" in entry:
            final_reward = entry.get("reward/mean", entry.get("rewards/mean", 0))
            break
    for entry in reversed(log_history):
        if "kl" in entry:
            final_kl = entry["kl"]
            break

    # Save log
    log_path = os.path.join(output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(log_history, f, indent=2)

    del model, trainer
    torch.cuda.empty_cache()

    summary = {
        "G": G,
        "beta": beta,
        "max_steps": max_steps,
        "final_reward_mean": final_reward,
        "final_kl": final_kl,
        "total_steps": result.global_step,
    }
    print(f"  Final reward mean: {final_reward:.4f}, KL: {final_kl:.4f}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="./ablation_results")
    parser.add_argument("--max-steps", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    gsm8k = load_dataset("openai/gsm8k", "main")
    train_dataset = gsm8k["train"].map(format_gsm8k_for_grpo)

    # Ablation experiments
    experiments = [
        # Vary G with fixed KL
        {"G": 4,  "beta": 0.04},
        {"G": 8,  "beta": 0.04},
        {"G": 16, "beta": 0.04},
        # Vary beta with fixed G=8
        {"G": 8, "beta": 0.004},
        # G=8, beta=0.04 already covered above
        {"G": 8, "beta": 0.1},
    ]

    all_summaries = []
    for exp in experiments:
        summary = run_ablation(
            G=exp["G"],
            beta=exp["beta"],
            train_dataset=train_dataset,
            output_base=args.output_dir,
            max_steps=args.max_steps,
        )
        all_summaries.append(summary)

    # Save summary
    summary_path = os.path.join(args.output_dir, "ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nAll ablation results saved to {summary_path}")

    # Print table
    print("\n" + "=" * 60)
    print("Ablation Summary")
    print("=" * 60)
    print(f"{'G':<5} {'Beta':<10} {'Final Reward':<15} {'Final KL':<10}")
    print("-" * 40)
    for s in all_summaries:
        print(f"{s['G']:<5} {s['beta']:<10} {s['final_reward_mean']:<15.4f} {s['final_kl']:<10.4f}")


if __name__ == "__main__":
    main()
