"""
步骤2：指令微调训练
- 加载 Qwen3-1.7B-Base + QLoRA
- 使用清洗后的数据集进行训练
- 保存最佳模型和训练曲线
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import config

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig


def main():
    print("=" * 60)
    print("步骤2：指令微调训练")
    print("=" * 60)

    # ===== 1. 加载数据集 =====
    print("\n[1/5] 加载预处理好的数据集...")
    data_dir = os.path.join(config.OUTPUTS_DIR, "datasets")
    train_dataset = load_from_disk(os.path.join(data_dir, "train"))
    val_dataset = load_from_disk(os.path.join(data_dir, "val"))
    print(f"  训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")

    # ===== 2. 加载模型 =====
    print("\n[2/5] 加载 4-bit 量化模型...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.INSTRUCT_MODEL, trust_remote_code=True
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # ===== 3. LoRA 配置 =====
    print("\n[3/5] 配置 LoRA 适配器...")
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_TARGET_MODULES,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ===== 4. 训练 =====
    print("\n[4/5] 开始训练...")
    sft_config = SFTConfig(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        lr_scheduler_type=config.LR_SCHEDULER_TYPE,
        warmup_ratio=config.WARMUP_RATIO,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        max_length=config.MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,
        logging_steps=config.LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        save_strategy="steps",
        save_steps=config.SAVE_STEPS,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        seed=config.SEED,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )
    train_result = trainer.train()

    # ===== 5. 保存模型和训练曲线 =====
    print("\n[5/5] 保存模型和训练曲线...")
    best_dir = os.path.join(config.OUTPUT_DIR, "best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"  最佳模型已保存到 {best_dir}")

    # 绘制损失曲线
    log_history = trainer.state.log_history
    train_losses = [(x["step"], x["loss"]) for x in log_history if "loss" in x]
    eval_losses = [(x["step"], x["eval_loss"]) for x in log_history if "eval_loss" in x]

    fig, ax = plt.subplots(figsize=(10, 5))
    if train_losses:
        ax.plot(*zip(*train_losses), label="Train Loss", alpha=0.8)
    if eval_losses:
        ax.plot(*zip(*eval_losses), label="Eval Loss", marker="o", markersize=4)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Lecture 2 SFT Training - Loss Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(config.OUTPUTS_DIR, "lecture2_loss.png")
    plt.savefig(loss_path, dpi=150)
    plt.close()
    print(f"  损失曲线已保存到 {loss_path}")

    # 训练状态分析
    print("\n=== 训练状态分析 ===")
    if eval_losses:
        last_eval = eval_losses[-1][1]
        min_eval = min(l[1] for l in eval_losses)
        if last_eval > min_eval * 1.05:
            print("  ⚠️  可能过拟合：最后的验证损失高于最低验证损失")
        else:
            print("  ✓ 训练状态良好：验证损失未明显上升")

    # 保存训练指标
    import json
    metrics = {
        "train_loss_final": train_losses[-1][1] if train_losses else None,
        "eval_loss_final": eval_losses[-1][1] if eval_losses else None,
        "eval_loss_best": min(l[1] for l in eval_losses) if eval_losses else None,
        "total_steps": train_losses[-1][0] if train_losses else 0,
        "train_runtime": train_result.metrics.get("train_runtime", 0),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
    }
    metrics_path = os.path.join(config.OUTPUTS_DIR, "train_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  训练指标已保存到 {metrics_path}")
    print("\n步骤2 完成！")


if __name__ == "__main__":
    main()
