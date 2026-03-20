#!/usr/bin/env python3
"""Quick ablation: run G=8 with different beta values only."""
import os, re, json, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

def extract_gsm8k_answer(answer_text):
    match = re.search(r'####\s*(.+)', answer_text)
    return match.group(1).strip().replace(",", "") if match else ""

def extract_answer_from_response(response):
    boxed = re.search(r'\\boxed\{([^}]+)\}', response)
    if boxed: return boxed.group(1).strip().replace(",", "")
    nums = re.findall(r'-?\d+\.?\d*', response)
    return nums[-1].replace(",", "") if nums else ""

def math_reward_fn(completions, answer, **kwargs):
    rewards = []
    for c, a in zip(completions, answer):
        gt = extract_gsm8k_answer(a)
        pred = extract_answer_from_response(c)
        try: correct = abs(float(pred) - float(gt)) < 1e-5
        except: correct = pred.strip() == gt.strip()
        rewards.append(1.0 if correct else 0.0)
    return rewards

def format_gsm8k(example):
    return {"prompt": f"Solve the following math problem step by step. Put your final answer in \\boxed{{}}.\n\nProblem: {example['question']}\n\nSolution:", "answer": example["answer"]}

gsm8k = load_dataset("openai/gsm8k", "main")
train_dataset = gsm8k["train"].map(format_gsm8k)
output_base = "./outputs/ablation"

for beta in [0.004, 0.1]:
    exp_name = f"G8_beta{beta}"
    output_dir = os.path.join(output_base, exp_name)
    if os.path.exists(os.path.join(output_dir, "training_log.json")):
        print(f"Skipping {exp_name} — already done")
        continue
    print(f"\n{'='*60}\nAblation: G=8, beta={beta}\n{'='*60}")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B-Base", torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    config = GRPOConfig(output_dir=output_dir, num_generations=8, max_completion_length=1024, beta=beta, learning_rate=5e-6, per_device_train_batch_size=8, gradient_accumulation_steps=4, max_steps=50, bf16=True, gradient_checkpointing=True, optim="adamw_torch", logging_steps=1, save_steps=50, save_total_limit=1, seed=42, report_to="none")
    trainer = GRPOTrainer(model=model, args=config, train_dataset=train_dataset, processing_class=tokenizer, reward_funcs=math_reward_fn)
    result = trainer.train()
    log_history = trainer.state.log_history
    with open(os.path.join(output_dir, "training_log.json"), "w") as f:
        json.dump(log_history, f, indent=2)
    del model, trainer
    torch.cuda.empty_cache()
    print(f"  {exp_name} done: {result.global_step} steps")

# Create ablation summary from all available results
summaries = []
for d in sorted(os.listdir(output_base)):
    log_path = os.path.join(output_base, d, "training_log.json")
    if not os.path.exists(log_path): continue
    with open(log_path) as f:
        log = json.load(f)
    # Parse G and beta from dir name
    parts = d.split("_")
    G = int(parts[0][1:])
    beta_val = float(parts[1][4:])
    final_reward = 0.0
    final_kl = 0.0
    for entry in reversed(log):
        if final_reward == 0 and ("reward" in entry):
            final_reward = entry.get("reward", 0)
        if final_kl == 0 and ("kl" in entry):
            final_kl = entry.get("kl", 0)
        if final_reward and final_kl: break
    summaries.append({"G": G, "beta": beta_val, "final_reward_mean": final_reward, "final_kl": final_kl, "max_steps": 50, "total_steps": 50})

with open(os.path.join(output_base, "ablation_summary.json"), "w") as f:
    json.dump(summaries, f, indent=2)
print(f"\nAblation summary saved with {len(summaries)} experiments")
for s in summaries:
    print(f"  G={s['G']}, beta={s['beta']}: reward={s['final_reward_mean']:.4f}, kl={s['final_kl']:.4f}")
