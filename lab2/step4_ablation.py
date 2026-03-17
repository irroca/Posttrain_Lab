"""
步骤4：数据质量消融实验
- 对比三种数据质量级别：raw / dedup / clean
- 固定超参数和数据量，仅改变数据质量
- 使用 LLM-as-Judge 评估每个模型
- 生成消融对比图表
"""
import sys, os
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))
import config

import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from step3_eval import generate_response, judge_with_local, judge_with_api

plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def train_model_on_data(train_data, val_data, tokenizer, output_dir, label):
    """在指定数据上训练模型"""
    print(f"\n  训练 [{label}] 模型 (数据量: {len(train_data)})...")

    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_TARGET_MODULES,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=config.ABLATION_EPOCHS,
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
        save_strategy="no",
        ddp_find_unused_parameters=False,
        report_to="none",
        seed=config.SEED,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=tokenizer,
    )
    trainer.train()

    # 保存模型
    best_dir = os.path.join(output_dir, "best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    # 提取训练日志
    log_history = trainer.state.log_history
    train_losses = [(x["step"], x["loss"]) for x in log_history if "loss" in x]
    eval_losses = [(x["step"], x["eval_loss"]) for x in log_history if "eval_loss" in x]

    metrics = {
        "train_loss_final": train_losses[-1][1] if train_losses else None,
        "eval_loss_final": eval_losses[-1][1] if eval_losses else None,
        "eval_loss_best": min(l[1] for l in eval_losses) if eval_losses else None,
        "total_steps": train_losses[-1][0] if train_losses else 0,
        "data_size": len(train_data),
    }

    del model, trainer
    torch.cuda.empty_cache()

    return metrics, train_losses, eval_losses


def evaluate_ablation_model(adapter_path, tokenizer, prompts, judge_fn, label):
    """评估消融实验模型"""
    from peft import PeftModel
    print(f"\n  评估 [{label}] 模型...")

    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)

    results = []
    for i, p in enumerate(prompts):
        answer = generate_response(model, tokenizer, p["prompt"])
        judge_result = judge_fn(p["prompt"], answer)
        results.append({
            "prompt": p["prompt"],
            "category": p["category"],
            "answer": answer,
            "score": judge_result["score"],
        })
        print(f"    [{i+1}/{len(prompts)}] {p['category']}: Score={judge_result['score']}")

    del model
    torch.cuda.empty_cache()
    return results


def main():
    print("=" * 60)
    print("步骤4：数据质量消融实验")
    print("=" * 60)
    print("  消融变量: 数据质量 (raw / dedup / clean)")
    print(f"  固定数据量: {config.ABLATION_SAMPLE_SIZE}")
    print(f"  训练轮数: {config.ABLATION_EPOCHS}")

    data_dir = os.path.join(config.OUTPUTS_DIR, "datasets")
    tokenizer = AutoTokenizer.from_pretrained(config.INSTRUCT_MODEL, trust_remote_code=True)

    # 加载三种质量级别的数据
    train_raw = load_from_disk(os.path.join(data_dir, "train_raw"))
    train_dedup = load_from_disk(os.path.join(data_dir, "train_dedup"))
    train_clean = load_from_disk(os.path.join(data_dir, "train"))
    val_dataset = load_from_disk(os.path.join(data_dir, "val"))

    # 统一样本数量以公平比较
    sample_size = min(
        config.ABLATION_SAMPLE_SIZE,
        len(train_raw), len(train_dedup), len(train_clean)
    )
    print(f"\n  实际使用样本数: {sample_size}")
    print(f"    raw 数据集大小: {len(train_raw)}")
    print(f"    dedup 数据集大小: {len(train_dedup)}")
    print(f"    clean 数据集大小: {len(train_clean)}")

    train_raw_sub = train_raw.shuffle(seed=config.SEED).select(range(sample_size))
    train_dedup_sub = train_dedup.shuffle(seed=config.SEED).select(range(sample_size))
    train_clean_sub = train_clean.shuffle(seed=config.SEED).select(range(min(sample_size, len(train_clean))))

    conditions = {
        "raw (无QC)": train_raw_sub,
        "dedup (仅去重)": train_dedup_sub,
        "clean (完整QC)": train_clean_sub,
    }

    # ===== 训练阶段 =====
    print("\n" + "=" * 40)
    print("训练阶段")
    print("=" * 40)

    all_metrics = {}
    all_train_losses = {}
    all_eval_losses = {}

    for label, data in conditions.items():
        safe_label = label.split(" ")[0]
        output_dir = os.path.join(config.OUTPUT_DIR, f"ablation_{safe_label}")
        metrics, train_losses, eval_losses = train_model_on_data(
            data, val_dataset, tokenizer, output_dir, label
        )
        all_metrics[label] = metrics
        all_train_losses[label] = train_losses
        all_eval_losses[label] = eval_losses
        print(f"  [{label}] 完成 - 最终训练损失: {metrics['train_loss_final']:.4f}, "
              f"最佳验证损失: {metrics['eval_loss_best']:.4f}")

    # ===== 评估阶段 =====
    print("\n" + "=" * 40)
    print("评估阶段")
    print("=" * 40)

    # 配置 judge
    if config.USE_LOCAL_JUDGE:
        print("  使用本地 Instruct 模型作为 Judge...")
        judge_model = AutoModelForCausalLM.from_pretrained(
            config.INSTRUCT_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        judge_tokenizer = AutoTokenizer.from_pretrained(
            config.INSTRUCT_MODEL, trust_remote_code=True
        )
        def judge_fn(q, a):
            return judge_with_local(q, a, judge_model, judge_tokenizer)
    else:
        judge_fn = judge_with_api

    eval_prompts = config.EVAL_PROMPTS[:15]  # 使用 15 条评估
    ablation_eval_results = {}

    for label in conditions:
        safe_label = label.split(" ")[0]
        adapter_path = os.path.join(config.OUTPUT_DIR, f"ablation_{safe_label}", "best")
        results = evaluate_ablation_model(
            adapter_path, tokenizer, eval_prompts, judge_fn, label
        )
        ablation_eval_results[label] = results

    if config.USE_LOCAL_JUDGE and judge_model is not None:
        del judge_model
        torch.cuda.empty_cache()

    # ===== 生成图表 =====
    print("\n" + "=" * 40)
    print("生成图表")
    print("=" * 40)

    # 图1: 训练损失曲线对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#e74c3c", "#f39c12", "#27ae60"]
    for (label, losses), color in zip(all_train_losses.items(), colors):
        if losses:
            steps, vals = zip(*losses)
            axes[0].plot(steps, vals, label=label, alpha=0.8, color=color)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Training Loss")
    axes[0].set_title("Training Loss - Data Quality Ablation")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for (label, losses), color in zip(all_eval_losses.items(), colors):
        if losses:
            steps, vals = zip(*losses)
            axes[1].plot(steps, vals, label=label, marker="o", markersize=4,
                        alpha=0.8, color=color)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Eval Loss")
    axes[1].set_title("Eval Loss - Data Quality Ablation")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUTS_DIR, "ablation_loss_curves.png"), dpi=150)
    plt.close()

    # 图2: LLM-as-Judge 平均评分对比
    labels = list(ablation_eval_results.keys())
    avg_scores = []
    for label in labels:
        scores = [r["score"] for r in ablation_eval_results[label] if r["score"] is not None]
        avg_scores.append(sum(scores) / len(scores) if scores else 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(labels)), avg_scores, color=colors[:len(labels)], alpha=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Average LLM-as-Judge Score")
    ax.set_title("Data Quality Ablation - Average Score Comparison")
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, score in zip(bars, avg_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{score:.2f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUTS_DIR, "ablation_scores.png"), dpi=150)
    plt.close()

    # 图3: 分类别评分热力图
    categories = list(dict.fromkeys(p["category"] for p in eval_prompts))
    cat_scores = {}
    for label in labels:
        cat_scores[label] = {}
        for cat in categories:
            scores = [r["score"] for r in ablation_eval_results[label]
                     if r["category"] == cat and r["score"] is not None]
            cat_scores[label][cat] = sum(scores) / len(scores) if scores else 0

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(categories))
    width = 0.25
    for i, (label, color) in enumerate(zip(labels, colors)):
        vals = [cat_scores[label][cat] for cat in categories]
        ax.bar(x + i*width, vals, width, label=label, color=color, alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Average Score")
    ax.set_title("Data Quality Ablation - Category Breakdown")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUTS_DIR, "ablation_category_scores.png"), dpi=150)
    plt.close()

    # 保存消融实验结果
    ablation_report = {
        "ablation_type": "数据质量消融",
        "conditions": {
            label: {
                "data_size": all_metrics[label]["data_size"],
                "train_loss_final": all_metrics[label]["train_loss_final"],
                "eval_loss_best": all_metrics[label]["eval_loss_best"],
                "avg_judge_score": avg_scores[i],
            }
            for i, label in enumerate(labels)
        },
        "category_scores": cat_scores,
        "eval_details": {
            label: [
                {"prompt": r["prompt"], "category": r["category"],
                 "answer": r["answer"], "score": r["score"]}
                for r in ablation_eval_results[label]
            ]
            for label in labels
        },
    }
    report_path = os.path.join(config.OUTPUTS_DIR, "ablation_results.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(ablation_report, f, ensure_ascii=False, indent=2)

    print(f"\n  结果已保存到 {report_path}")
    print(f"  图表已保存到 outputs/ablation_*.png")

    # 打印摘要
    print("\n" + "=" * 60)
    print("消融实验摘要")
    print("=" * 60)
    print(f"{'条件':<25}{'训练损失':<15}{'验证损失':<15}{'Judge评分':<10}")
    print("-" * 65)
    for i, label in enumerate(labels):
        m = all_metrics[label]
        print(f"{label:<25}{m['train_loss_final']:.4f}{'':<9}"
              f"{m['eval_loss_best']:.4f}{'':<9}{avg_scores[i]:.2f}")

    print("\n步骤4 完成！")


if __name__ == "__main__":
    main()
