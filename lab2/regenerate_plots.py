"""重新生成所有图表（使用正确的中文字体）"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import config
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ===== 1. 重新生成 step1 数据分布图 =====
print("重新生成数据分布图...")
data_analysis = load_json(os.path.join(config.OUTPUTS_DIR, "data_analysis.json"))
ts = data_analysis["token_stats"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# 这里用英文标签避免字体问题
user_stats = ts["user_tokens"]
asst_stats = ts["assistant_tokens"]

axes[0].bar(["Mean", "Median", "P5", "P95", "Min", "Max"],
            [user_stats["mean"], user_stats["median"], user_stats["p5"],
             user_stats["p95"], user_stats["min"], user_stats["max"]],
            color="#3498db", alpha=0.8)
axes[0].set_title("User Input Token Distribution")
axes[0].set_ylabel("Tokens")
axes[0].grid(True, alpha=0.3)

axes[1].bar(["Mean", "Median", "P5", "P95", "Min", "Max"],
            [asst_stats["mean"], asst_stats["median"], asst_stats["p5"],
             asst_stats["p95"], asst_stats["min"], asst_stats["max"]],
            color="#e74c3c", alpha=0.8)
axes[1].set_title("Assistant Reply Token Distribution")
axes[1].set_ylabel("Tokens")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(config.OUTPUTS_DIR, "data_distribution.png"), dpi=150)
plt.close()
print("  data_distribution.png 已更新")

# ===== 2. 重新生成训练损失图 =====
print("重新生成训练损失图...")
train_metrics = load_json(os.path.join(config.OUTPUTS_DIR, "train_metrics.json"))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
if train_metrics.get("train_losses"):
    steps, vals = zip(*train_metrics["train_losses"])
    axes[0].plot(steps, vals, color="#3498db", alpha=0.8)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Training Loss")
    axes[0].set_title("Lab2 SFT Training Loss")
    axes[0].grid(True, alpha=0.3)

if train_metrics.get("eval_losses"):
    steps, vals = zip(*train_metrics["eval_losses"])
    axes[1].plot(steps, vals, color="#e74c3c", marker="o", markersize=4, alpha=0.8)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Eval Loss")
    axes[1].set_title("Lab2 SFT Eval Loss")
    axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(config.OUTPUTS_DIR, "lecture2_loss.png"), dpi=150)
plt.close()
print("  lecture2_loss.png 已更新")

# ===== 3. 重新生成消融实验图 =====
print("重新生成消融实验图表...")
ablation = load_json(os.path.join(config.OUTPUTS_DIR, "ablation_results.json"))

# 平均分对比
conditions = ablation["conditions"]
labels = list(conditions.keys())
avg_scores = [conditions[l]["avg_judge_score"] for l in labels]
colors = ["#e74c3c", "#f39c12", "#27ae60"]

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
print("  ablation_scores.png 已更新")

# 分类别评分
cat_scores = ablation["category_scores"]
categories = list(cat_scores[labels[0]].keys())

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(categories))
width = 0.25
for i, (label, color) in enumerate(zip(labels, colors)):
    vals = [cat_scores[label].get(cat, 0) for cat in categories]
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
print("  ablation_category_scores.png 已更新")

# 训练损失曲线 (如果有)
# 消融实验的训练日志没有保存loss曲线到json中
# 只生成上面的评分图即可

# ===== 4. 生成评估对比图 =====
print("生成 LLM-as-Judge 评分对比图...")
eval_results = load_json(os.path.join(config.OUTPUTS_DIR, "eval_results.json"))
category_stats = load_json(os.path.join(config.OUTPUTS_DIR, "category_stats.json"))

if eval_results:
    model_names = list(eval_results.keys())
    avg_per_model = []
    for name in model_names:
        scores = [r["score"] for r in eval_results[name] if r["score"] is not None]
        avg_per_model.append(sum(scores) / len(scores) if scores else 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    model_colors = ["#3498db", "#e74c3c", "#27ae60"]
    bars = ax.bar(range(len(model_names)), avg_per_model,
                  color=model_colors[:len(model_names)], alpha=0.8)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylabel("Average LLM-as-Judge Score")
    ax.set_title("Model Comparison - Average Score (25 prompts)")
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, score in zip(bars, avg_per_model):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{score:.2f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUTS_DIR, "eval_scores_comparison.png"), dpi=150)
    plt.close()
    print("  eval_scores_comparison.png 已更新")

    # 分类别对比图
    if category_stats:
        categories = list(category_stats.keys())
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(categories))
        width = 0.25
        for i, (name, color) in enumerate(zip(model_names, model_colors)):
            vals = [category_stats[cat].get(name, 0) for cat in categories]
            ax.bar(x + i*width, vals, width, label=name, color=color, alpha=0.8)
        ax.set_xticks(x + width)
        ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Average Score")
        ax.set_title("Model Comparison - Category Breakdown")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(config.OUTPUTS_DIR, "eval_category_comparison.png"), dpi=150)
        plt.close()
        print("  eval_category_comparison.png 已更新")

print("\n所有图表已重新生成！")
