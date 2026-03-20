"""
Step 2: DPO 训练 —— 使用 DPOTrainer 对齐 SFT 模型
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import config

import json
import time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer


def main():
    print("=" * 60)
    print("Step 2: DPO 训练")
    print("=" * 60)

    # 1. 加载模型
    print("\n[1/5] 加载 SFT 基座模型...")
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
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"模型加载完成: {config.BASE_MODEL}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # 2. 配置 LoRA
    print("\n[2/5] 配置 LoRA...")
    peft_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 3. 加载数据
    print("\n[3/5] 加载偏好数据...")
    dataset = load_dataset(config.DATASET_NAME, split=config.TRAIN_SPLIT)
    train_dataset = dataset.select(range(min(config.TRAIN_SUBSET_SIZE, len(dataset))))
    eval_dataset = load_dataset(
        config.DATASET_NAME, split=config.TEST_SPLIT
    ).select(range(config.EVAL_SUBSET_SIZE))

    # TRL DPOTrainer 需要 chosen/rejected 为消息列表（list of dicts）
    # UltraFeedback 已经是这个格式，无需转换
    # 但需要去掉 prompt 字段（TRL 从 chosen/rejected 中自动提取）
    def preprocess_for_dpo(example):
        """确保数据格式兼容 TRL DPOTrainer"""
        # chosen/rejected 已经是 [{"role": "user", ...}, {"role": "assistant", ...}]
        return {
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }

    train_dataset = train_dataset.map(preprocess_for_dpo, remove_columns=[
        c for c in train_dataset.column_names if c not in ("chosen", "rejected")
    ])
    eval_dataset = eval_dataset.map(preprocess_for_dpo, remove_columns=[
        c for c in eval_dataset.column_names if c not in ("chosen", "rejected")
    ])
    print(f"训练集: {len(train_dataset)} 样本, 验证集: {len(eval_dataset)} 样本")
    print(f"训练集列: {train_dataset.column_names}")

    # 4. 配置 DPO 训练
    print("\n[4/5] 配置 DPO 训练参数...")
    dpo_config = DPOConfig(
        output_dir=config.DPO_OUTPUT_DIR,
        beta=config.DPO_BETA,
        loss_type=config.DPO_LOSS_TYPE,
        learning_rate=config.DPO_LEARNING_RATE,
        num_train_epochs=config.DPO_NUM_EPOCHS,
        per_device_train_batch_size=config.DPO_BATCH_SIZE,
        gradient_accumulation_steps=config.DPO_GRAD_ACCUM,
        max_length=config.DPO_MAX_LENGTH,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=10,
        save_steps=200,
        eval_strategy="steps",
        eval_steps=100,
        remove_unused_columns=False,
        seed=config.SEED,
        report_to="none",
    )

    # 5. 启动训练
    print("\n[5/5] 开始 DPO 训练...")
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # TRL 自动使用策略模型初始权重作为参考
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    dpo_result = dpo_trainer.train()
    train_time = time.time() - start_time
    peak_memory = torch.cuda.max_memory_allocated() / 1e9

    print(f"训练完成! 最终损失: {dpo_result.training_loss:.4f}")
    print(f"训练时间: {train_time / 60:.1f} 分钟")
    print(f"峰值显存: {peak_memory:.2f} GB")

    # 保存模型
    save_path = os.path.join(config.DPO_OUTPUT_DIR, "final")
    dpo_trainer.save_model(save_path)
    print(f"DPO 模型已保存到: {save_path}")

    # 记录训练指标
    training_log = dpo_trainer.state.log_history
    dpo_metrics = {
        "train_loss": [l.get("loss") for l in training_log if "loss" in l],
        "eval_loss": [l.get("eval_loss") for l in training_log if "eval_loss" in l],
        "rewards_chosen": [l.get("rewards/chosen") for l in training_log if "rewards/chosen" in l],
        "rewards_rejected": [l.get("rewards/rejected") for l in training_log if "rewards/rejected" in l],
        "reward_margins": [l.get("rewards/margins") for l in training_log if "rewards/margins" in l],
        "train_steps": [l.get("step") for l in training_log if "loss" in l],
        "eval_steps": [l.get("step") for l in training_log if "eval_loss" in l],
        "training_time_seconds": train_time,
        "peak_memory_gb": peak_memory,
        "final_loss": dpo_result.training_loss,
        "train_runtime": dpo_result.metrics.get("train_runtime", train_time),
    }

    if dpo_metrics["reward_margins"]:
        print(f"初始 reward margin: {dpo_metrics['reward_margins'][0]:.4f}")
        print(f"最终 reward margin: {dpo_metrics['reward_margins'][-1]:.4f}")

    metrics_path = os.path.join(config.OUTPUT_DIR, "dpo_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(dpo_metrics, f, indent=2)
    print(f"DPO 指标已保存到: {metrics_path}")

    # 清理显存
    del dpo_trainer, model
    torch.cuda.empty_cache()
    print("Step 2 完成!")


if __name__ == "__main__":
    main()
