#!/usr/bin/env python3
"""
Lab 4: Mini DeepSeek-R1-Zero — GRPO Training on GSM8K
Train Qwen3-1.7B-Base with GRPO to learn math reasoning.
"""
import os
import re
import json
import torch
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# ── Reward Functions ──────────────────────────────────────────────────────────

def extract_gsm8k_answer(answer_text: str) -> str:
    """Extract final numeric answer from GSM8K answer text."""
    match = re.search(r'####\s*(.+)', answer_text)
    if match:
        return match.group(1).strip().replace(",", "")
    return ""


def extract_answer_from_response(response: str) -> str:
    """Extract final numeric answer from model response."""
    # Strategy 1: Match \boxed{...}
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
    if boxed_match:
        return boxed_match.group(1).strip().replace(",", "")
    # Strategy 2: Match "answer is X" patterns
    answer_match = re.search(
        r'(?:答案|结果|answer|result)\s*(?:是|为|=|:)\s*(-?\d+\.?\d*)',
        response, re.IGNORECASE
    )
    if answer_match:
        return answer_match.group(1).strip()
    # Strategy 3: Match #### format
    hash_match = re.search(r'####\s*(-?\d+\.?\d*)', response)
    if hash_match:
        return hash_match.group(1).strip()
    # Strategy 4: Fallback to last number
    numbers = re.findall(r'-?\d+\.?\d*', response)
    if numbers:
        return numbers[-1].replace(",", "")
    return ""


def math_reward_fn(completions, answer, **kwargs):
    """GRPO reward function: correctness check."""
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


def format_reward_fn(completions, **kwargs):
    """Format reward: encourage <think>...</think> tags."""
    rewards = []
    for completion in completions:
        if "<think>" in completion and "</think>" in completion:
            rewards.append(0.1)
        else:
            rewards.append(0.0)
    return rewards


def combined_reward_fn(completions, answer, **kwargs):
    """Correctness + format reward."""
    correctness = math_reward_fn(completions, answer)
    format_bonus = format_reward_fn(completions)
    return [c + f for c, f in zip(correctness, format_bonus)]


# ── Dataset Preparation ──────────────────────────────────────────────────────

def format_gsm8k_for_grpo(example):
    """Format GSM8K example for GRPO training."""
    prompt = (
        f"Solve the following math problem step by step. "
        f"Put your final answer in \\boxed{{}}.\n\n"
        f"Problem: {example['question']}\n\n"
        f"Solution:"
    )
    return {
        "prompt": prompt,
        "answer": example["answer"],
    }


def main():
    parser = argparse.ArgumentParser(description="GRPO Training")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--output-dir", type=str, default="./grpo-qwen3-1.7b-math")
    parser.add_argument("--use-format-reward", action="store_true",
                        help="Use combined reward (correctness + format)")
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    args = parser.parse_args()

    print("=" * 60)
    print("Lab 4: GRPO Training — Mini DeepSeek-R1-Zero")
    print("=" * 60)

    # ── Load model ────────────────────────────────────────────────────────
    model_name = "Qwen/Qwen3-1.7B-Base"
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")

    # ── Load dataset ──────────────────────────────────────────────────────
    gsm8k = load_dataset("openai/gsm8k", "main")
    train_dataset = gsm8k["train"].map(format_gsm8k_for_grpo)
    print(f"Training data: {len(train_dataset)} examples")

    # ── GRPO Config ───────────────────────────────────────────────────────
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        # GRPO core
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        # Training
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.num_generations,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        max_steps=args.max_steps,
        # Optimization
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        # Logging & saving
        logging_steps=1,
        save_steps=args.save_steps,
        save_total_limit=5,
        # Other
        seed=42,
        report_to="none",
    )

    # ── Select reward function ────────────────────────────────────────────
    if args.use_format_reward:
        reward_fn = combined_reward_fn
        print("Using combined reward (correctness + format)")
    else:
        reward_fn = math_reward_fn
        print("Using math correctness reward only")

    # ── Create trainer ────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
    )

    print("=" * 60)
    print("Starting GRPO training")
    print(f"Model: {model_name}")
    print(f"Group size G: {grpo_config.num_generations}")
    print(f"Max steps: {grpo_config.max_steps}")
    print(f"Learning rate: {grpo_config.learning_rate}")
    print(f"Beta (KL coef): {grpo_config.beta}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 60)

    result = trainer.train()
    print(f"\nTraining complete! Total steps: {result.global_step}")

    # ── Save final model ──────────────────────────────────────────────────
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Final model saved to {final_dir}")

    # ── Save training log ─────────────────────────────────────────────────
    log_path = os.path.join(args.output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(f"Training log saved to {log_path}")


if __name__ == "__main__":
    main()
