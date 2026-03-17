"""
步骤 5：超参数探索
- 修改学习率或 LoRA 秩，重新训练模型
- 对比不同超参数下的训练损失曲线
- 对比不同超参数下的输出质量
- 提供方案 A（修改学习率）和方案 B（修改 LoRA 秩）
"""

import sys
import os
import re
import json
import argparse
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_from_disk

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    MODEL_NAME, LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    NUM_TRAIN_EPOCHS, PER_DEVICE_TRAIN_BATCH_SIZE, PER_DEVICE_EVAL_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS, LR_SCHEDULER_TYPE, WARMUP_RATIO,
    MAX_SEQ_LENGTH, LOGGING_STEPS, SEED, OUTPUT_DIR,
    EVAL_PROMPTS, MAX_NEW_TOKENS, TEMPERATURE, TOP_P,
)


def clean_response(text):
    """清理模型输出"""
    text = re.sub(
        r'[^\u4e00-\u9fff\u0020-\u007ea-zA-Z0-9，。！？：；""''（）、\n\r{}\[\]"\',\.!?:;()\-+=#]',
        '', text
    )
    return text.strip()


def load_fresh_model(lora_r=LORA_R, lora_alpha=LORA_ALPHA):
    """加载全新的量化模型并配置 LoRA（支持自定义秩）"""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def train_with_params(model, tokenizer, train_dataset, eval_dataset,
                      learning_rate, output_dir, experiment_name):
    """用指定超参数训练"""
    print(f"\n{'=' * 60}")
    print(f"超参数实验: {experiment_name}")
    print(f"  learning_rate = {learning_rate}")
    print(f"  output_dir = {output_dir}")
    print(f"{'=' * 60}")

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=learning_rate,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_ratio=WARMUP_RATIO,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=MAX_SEQ_LENGTH,
        packing=False,
        logging_steps=LOGGING_STEPS,
        save_strategy="no",
        report_to="none",
        seed=SEED,
    )

    # 只保留 messages 列
    cols_to_remove = [c for c in train_dataset.column_names if c != "messages"]
    if cols_to_remove:
        train_dataset = train_dataset.remove_columns(cols_to_remove)
        eval_dataset = eval_dataset.remove_columns(cols_to_remove)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print("Starting training...")
    train_result = trainer.train()

    # 保存模型到 output_dir
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\nTraining completed!")
    print(f"  Train loss: {train_result.training_loss:.4f}")
    print(f"  Train runtime: {train_result.metrics['train_runtime']:.0f}s")
    print(f"  Model saved to: {output_dir}")

    return trainer


def generate_response_safe(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS):
    """生成回复"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=1.2,
        )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    return clean_response(response)


def compare_experiments(trainer_baseline, trainer_experiment, experiment_name):
    """对比两个实验的损失曲线"""
    print("\n绘制对比损失曲线...")

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # 基线实验
    log_baseline = trainer_baseline.state.log_history
    train_losses_baseline = [(x["step"], x["loss"]) for x in log_baseline if "loss" in x]
    if train_losses_baseline:
        ax.plot(*zip(*train_losses_baseline), label="Baseline (lr=2e-4)", alpha=0.8)

    # 对比实验
    log_exp = trainer_experiment.state.log_history
    train_losses_exp = [(x["step"], x["loss"]) for x in log_exp if "loss" in x]
    if train_losses_exp:
        ax.plot(*zip(*train_losses_exp), label=f"Experiment ({experiment_name})", alpha=0.8)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training Loss Comparison: {experiment_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    filename = f"outputs/hyperparam_comparison_{experiment_name.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=150)
    print(f"对比图已保存到 {filename}")


def collect_outputs(model, tokenizer, label):
    """在 EVAL_PROMPTS 上采样输出，返回结构化结果"""
    model.eval()
    outputs = []
    print(f"\n--- {label} 模型输出 ---")
    for prompt in EVAL_PROMPTS:
        response = generate_response_safe(model, tokenizer, prompt)
        outputs.append({"prompt": prompt, "response": response})
        print(f"  [Prompt]: {prompt}")
        print(f"  [Response]: {response[:200]}")
        print()
    return outputs


def run_plan_a():
    """方案 A：修改学习率 (2e-4 → 5e-5)"""
    print("\n" + "#" * 60)
    print("方案 A：学习率对比实验")
    print("#" * 60)

    train_dataset = load_from_disk("outputs/train_dataset")
    eval_dataset = load_from_disk("outputs/eval_dataset")

    # 实验 1: 基线 lr=2e-4
    model1, tokenizer1 = load_fresh_model()
    trainer1 = train_with_params(
        model1, tokenizer1, train_dataset, eval_dataset,
        learning_rate=2e-4,
        output_dir="./qwen3-sft-lr2e4",
        experiment_name="lr=2e-4 (baseline)",
    )
    outputs_baseline = collect_outputs(model1, tokenizer1, "lr=2e-4")

    del model1
    torch.cuda.empty_cache()

    # 实验 2: lr=5e-5
    model2, tokenizer2 = load_fresh_model()
    trainer2 = train_with_params(
        model2, tokenizer2, train_dataset, eval_dataset,
        learning_rate=5e-5,
        output_dir="./qwen3-sft-lr5e5",
        experiment_name="lr=5e-5",
    )
    outputs_experiment = collect_outputs(model2, tokenizer2, "lr=5e-5")

    # 绘制对比图
    compare_experiments(trainer1, trainer2, "lr 2e-4 vs 5e-5")

    # 组装输出质量对比
    quality_comparison = []
    for i, prompt in enumerate(EVAL_PROMPTS):
        quality_comparison.append({
            "prompt": prompt,
            "baseline_response (lr=2e-4)": outputs_baseline[i]["response"],
            "experiment_response (lr=5e-5)": outputs_experiment[i]["response"],
        })

    # 保存完整对比结果
    results = {
        "experiment": "learning_rate_comparison",
        "baseline_lr": 2e-4,
        "baseline_final_loss": trainer1.state.log_history[-2].get("loss", "N/A"),
        "experiment_lr": 5e-5,
        "experiment_final_loss": trainer2.state.log_history[-2].get("loss", "N/A"),
        "output_quality_comparison": quality_comparison,
    }
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/hyperparam_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n超参数实验结果（含输出质量对比）已保存到 outputs/hyperparam_results.json")

    del model2
    torch.cuda.empty_cache()


def run_plan_b():
    """方案 B：修改 LoRA 秩 (32 → 8)"""
    print("\n" + "#" * 60)
    print("方案 B：LoRA 秩对比实验")
    print("#" * 60)

    train_dataset = load_from_disk("outputs/train_dataset")
    eval_dataset = load_from_disk("outputs/eval_dataset")

    # 实验 1: 基线 r=32
    model1, tokenizer1 = load_fresh_model(lora_r=32, lora_alpha=64)
    trainer1 = train_with_params(
        model1, tokenizer1, train_dataset, eval_dataset,
        learning_rate=2e-4,
        output_dir="./qwen3-sft-r32",
        experiment_name="r=32 (baseline)",
    )
    outputs_r32 = collect_outputs(model1, tokenizer1, "r=32")

    del model1
    torch.cuda.empty_cache()

    # 实验 2: r=8
    model2, tokenizer2 = load_fresh_model(lora_r=8, lora_alpha=16)
    trainer2 = train_with_params(
        model2, tokenizer2, train_dataset, eval_dataset,
        learning_rate=2e-4,
        output_dir="./qwen3-sft-r8",
        experiment_name="r=8, alpha=16",
    )
    outputs_r8 = collect_outputs(model2, tokenizer2, "r=8")

    compare_experiments(trainer1, trainer2, "LoRA r=32 vs r=8")

    # 组装输出质量对比
    quality_comparison = []
    for i, prompt in enumerate(EVAL_PROMPTS):
        quality_comparison.append({
            "prompt": prompt,
            "baseline_response (r=32)": outputs_r32[i]["response"],
            "experiment_response (r=8)": outputs_r8[i]["response"],
        })

    results = {
        "experiment": "lora_rank_comparison",
        "baseline_r": 32,
        "baseline_final_loss": trainer1.state.log_history[-2].get("loss", "N/A"),
        "experiment_r": 8,
        "experiment_final_loss": trainer2.state.log_history[-2].get("loss", "N/A"),
        "output_quality_comparison": quality_comparison,
    }
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/hyperparam_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n超参数实验结果（含输出质量对比）已保存到 outputs/hyperparam_results.json")

    del model2
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="超参数探索实验")
    parser.add_argument(
        "--plan", type=str, default="a", choices=["a", "b"],
        help="选择实验方案: a=修改学习率, b=修改LoRA秩 (默认: a)"
    )
    args = parser.parse_args()

    if args.plan == "a":
        run_plan_a()
    else:
        run_plan_b()

    print("\n步骤 5 完成！")


if __name__ == "__main__":
    main()
