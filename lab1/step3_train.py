"""
步骤 3：SFT 训练
- 加载 4-bit 量化模型 + LoRA 适配器
- 加载步骤 2 处理好的数据集
- 使用 SFTTrainer 进行 QLoRA 微调
- 保存训练好的模型
- 绘制训练损失曲线
"""

import sys
import os
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
    OUTPUT_DIR, NUM_TRAIN_EPOCHS, PER_DEVICE_TRAIN_BATCH_SIZE,
    PER_DEVICE_EVAL_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE, LR_SCHEDULER_TYPE, WARMUP_RATIO, MAX_SEQ_LENGTH,
    LOGGING_STEPS, EVAL_STEPS, SAVE_STEPS, SAVE_TOTAL_LIMIT, SEED,
)


def load_model():
    """加载量化模型并配置 LoRA"""
    print("=" * 60)
    print("加载模型")
    print("=" * 60)

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
        device_map={"": 0},
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def load_datasets():
    """加载步骤 2 保存的数据集"""
    print("\n加载处理好的数据集...")
    train_dataset = load_from_disk("outputs/train_dataset")
    eval_dataset = load_from_disk("outputs/eval_dataset")
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    return train_dataset, eval_dataset


def train(model, tokenizer, train_dataset, eval_dataset):
    """执行 SFT 训练"""
    print("\n" + "=" * 60)
    print("开始 SFT 训练")
    print("=" * 60)

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_ratio=WARMUP_RATIO,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=MAX_SEQ_LENGTH,
        packing=False,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
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

    # 保存模型
    final_dir = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    # 打印训练结果
    print(f"\nTraining completed!")
    print(f"  Train loss: {train_result.training_loss:.4f}")
    print(f"  Train runtime: {train_result.metrics['train_runtime']:.0f}s")
    print(f"  Train samples/sec: {train_result.metrics['train_samples_per_second']:.2f}")

    return trainer


def plot_loss_curve(trainer):
    """绘制训练损失曲线"""
    print("\n绘制训练损失曲线...")

    log_history = trainer.state.log_history
    train_losses = [(x["step"], x["loss"]) for x in log_history if "loss" in x]
    eval_losses = [(x["step"], x["eval_loss"]) for x in log_history if "eval_loss" in x]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(*zip(*train_losses), label="Train Loss", alpha=0.8)
    if eval_losses:
        ax.plot(*zip(*eval_losses), label="Eval Loss", marker="o")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("SFT Training Loss Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/training_loss.png", dpi=150)
    print("Loss curve saved to outputs/training_loss.png")


def main():
    # 1. 加载模型
    model, tokenizer = load_model()

    # 2. 加载数据集
    train_dataset, eval_dataset = load_datasets()

    # 3. 训练
    trainer = train(model, tokenizer, train_dataset, eval_dataset)

    # 4. 绘制损失曲线
    plot_loss_curve(trainer)

    print("\n步骤 3 完成！模型已保存到", os.path.join(OUTPUT_DIR, "final"))
    return model, tokenizer, trainer


if __name__ == "__main__":
    main()
