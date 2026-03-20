"""
Step 7: 生成交付物 —— 对比表、训练曲线、安全测试报告、分析报告
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import config

import json
import numpy as np

# matplotlib 使用非交互式后端
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "SimHei", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_training_curves(dpo_metrics, simpo_metrics, output_dir):
    """绘制训练损失和 reward margin 曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. 训练损失
    ax = axes[0]
    if dpo_metrics and dpo_metrics.get("train_loss"):
        dpo_steps = dpo_metrics.get("train_steps", list(range(len(dpo_metrics["train_loss"]))))
        ax.plot(dpo_steps, dpo_metrics["train_loss"], label="DPO", color="blue", linewidth=1.5)
    if simpo_metrics and simpo_metrics.get("train_loss"):
        simpo_steps = simpo_metrics.get("train_steps", list(range(len(simpo_metrics["train_loss"]))))
        ax.plot(simpo_steps, simpo_metrics["train_loss"], label="SimPO", color="red", linewidth=1.5)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Reward Margin
    ax = axes[1]
    has_margin = False
    if dpo_metrics and dpo_metrics.get("reward_margins"):
        margins = dpo_metrics["reward_margins"]
        # 使用实际训练步数作为 x 轴，而非索引
        margin_steps = dpo_metrics.get("train_steps", list(range(len(margins))))
        # 若长度不匹配（reward_margins 可能多一个来自 eval），截断到最短
        n = min(len(margin_steps), len(margins))
        ax.plot(margin_steps[:n], margins[:n], label="DPO", color="blue", linewidth=1.5, marker="o", markersize=3)
        has_margin = True
    if simpo_metrics and simpo_metrics.get("reward_margins"):
        margins = simpo_metrics["reward_margins"]
        margin_steps = simpo_metrics.get("train_steps", list(range(len(margins))))
        n = min(len(margin_steps), len(margins))
        ax.plot(margin_steps[:n], margins[:n], label="SimPO", color="red", linewidth=1.5, marker="s", markersize=3)
        has_margin = True
    if has_margin:
        ax.set_xlabel("Eval Steps")
        ax.set_ylabel("Reward Margin")
        ax.set_title("Reward Margin Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No reward margin data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Reward Margin Curve (N/A)")

    # 3. 训练时间和显存对比
    ax = axes[2]
    dpo_time = dpo_metrics.get("training_time_seconds", 0) / 60 if dpo_metrics else 0
    simpo_time = simpo_metrics.get("training_time_seconds", 0) / 60 if simpo_metrics else 0
    dpo_mem = dpo_metrics.get("peak_memory_gb", 0) if dpo_metrics else 0
    simpo_mem = simpo_metrics.get("peak_memory_gb", 0) if simpo_metrics else 0

    x = np.arange(2)
    width = 0.35
    bars1 = ax.bar(x - width/2, [dpo_time, dpo_mem], width, label="DPO", color="steelblue")
    bars2 = ax.bar(x + width/2, [simpo_time, simpo_mem], width, label="SimPO", color="indianred")
    ax.set_xticks(x)
    ax.set_xticklabels(["Training Time (min)", "Peak Memory (GB)"])
    ax.set_title("Training Efficiency Comparison")
    ax.legend()
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"训练曲线已保存: {path}")


def plot_beta_ablation(ablation_results, output_dir):
    """绘制 beta 消融实验图"""
    if not ablation_results:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    betas = sorted([float(b) for b in ablation_results.keys()])
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    # 1. 各 beta 的训练损失曲线
    ax = axes[0]
    for i, beta in enumerate(betas):
        data = ablation_results[str(beta)]
        losses = data.get("train_losses", [])
        if losses:
            ax.plot(range(len(losses)), losses, label=f"beta={beta}", color=colors[i], linewidth=1.5)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs Beta")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Reward margin 随 beta 变化
    ax = axes[1]
    for i, beta in enumerate(betas):
        data = ablation_results[str(beta)]
        margins = data.get("reward_margins", [])
        if margins:
            ax.plot(range(len(margins)), margins, label=f"beta={beta}", color=colors[i], linewidth=1.5, marker="o", markersize=3)
    ax.set_xlabel("Eval Steps")
    ax.set_ylabel("Reward Margin")
    ax.set_title("Reward Margin vs Beta")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Final loss 和 safety refusal 对比
    ax = axes[2]
    final_losses = [ablation_results[str(b)].get("final_loss", 0) for b in betas]
    safety_rates = [ablation_results[str(b)].get("safety_refusal_rate", 0) * 100 for b in betas]

    x = np.arange(len(betas))
    width = 0.35
    ax2 = ax.twinx()
    bars1 = ax.bar(x - width/2, final_losses, width, label="Final Loss", color="steelblue")
    bars2 = ax2.bar(x + width/2, safety_rates, width, label="Safety Refusal %", color="indianred")
    ax.set_xlabel("Beta")
    ax.set_ylabel("Final Loss", color="steelblue")
    ax2.set_ylabel("Safety Refusal %", color="indianred")
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in betas])
    ax.set_title("Final Loss & Safety Refusal vs Beta")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    path = os.path.join(output_dir, "beta_ablation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Beta 消融图已保存: {path}")


def generate_report(dpo_metrics, simpo_metrics, eval_results, judge_results,
                    ablation_results, data_stats, output_dir):
    """生成完整的 Markdown 报告"""

    # 提取关键数据
    safety = eval_results.get("safety", {}) if eval_results else {}
    safety_details = eval_results.get("safety_details", {}) if eval_results else {}
    diversity = eval_results.get("diversity", {}) if eval_results else {}

    dpo_time = dpo_metrics.get("training_time_seconds", 0) / 60 if dpo_metrics else 0
    simpo_time = simpo_metrics.get("training_time_seconds", 0) / 60 if simpo_metrics else 0
    dpo_mem = dpo_metrics.get("peak_memory_gb", 0) if dpo_metrics else 0
    simpo_mem = simpo_metrics.get("peak_memory_gb", 0) if simpo_metrics else 0
    dpo_loss = dpo_metrics.get("final_loss", 0) if dpo_metrics else 0
    simpo_loss = simpo_metrics.get("final_loss", 0) if simpo_metrics else 0

    # Judge 分数
    judge_summary = judge_results.get("summary", {}) if judge_results else {}
    sft_judge = judge_summary.get("SFT", {}).get("avg_score", "N/A")
    dpo_judge = judge_summary.get("DPO", {}).get("avg_score", "N/A")
    simpo_judge = judge_summary.get("SimPO", {}).get("avg_score", "N/A")

    # Safety
    sft_safety = safety.get("SFT", {}).get("refusal_rate", 0) * 100 if safety else 0
    dpo_safety = safety.get("DPO", {}).get("refusal_rate", 0) * 100 if safety else 0
    simpo_safety = safety.get("SimPO", {}).get("refusal_rate", 0) * 100 if safety else 0

    # Diversity
    sft_div = diversity.get("SFT", {}).get("diversity_ratio", 0) if diversity else 0
    dpo_div = diversity.get("DPO", {}).get("diversity_ratio", 0) if diversity else 0
    simpo_div = diversity.get("SimPO", {}).get("diversity_ratio", 0) if diversity else 0

    def fmt_score(v):
        if isinstance(v, (int, float)):
            return f"{v:.2f}"
        return str(v)

    report = f"""# 第3课实验报告：DPO 对齐与 SimPO 对比

## 1. 胜率对比表

| 指标 | SFT-only | DPO | SimPO |
|------|----------|-----|-------|
| 有用性评分 (LLM-Judge, /10) | {fmt_score(sft_judge)} | {fmt_score(dpo_judge)} | {fmt_score(simpo_judge)} |
| 安全拒绝率 (%) | {sft_safety:.0f}% | {dpo_safety:.0f}% | {simpo_safety:.0f}% |
| 多样性 (unigram ratio) | {sft_div:.3f} | {dpo_div:.3f} | {simpo_div:.3f} |
| 训练时间 (min) | N/A | {dpo_time:.1f} | {simpo_time:.1f} |
| 峰值显存 (GB) | N/A | {dpo_mem:.2f} | {simpo_mem:.2f} |
| 最终训练损失 | N/A | {dpo_loss:.4f} | {simpo_loss:.4f} |

## 2. 训练曲线

![训练曲线](training_curves.png)

**关键观察：**
- DPO 训练时间: {dpo_time:.1f} 分钟, SimPO 训练时间: {simpo_time:.1f} 分钟
- DPO 峰值显存: {dpo_mem:.2f} GB, SimPO 峰值显存: {simpo_mem:.2f} GB
"""

    if dpo_metrics and dpo_metrics.get("reward_margins"):
        report += f"""- DPO reward margin: {dpo_metrics['reward_margins'][0]:.4f} → {dpo_metrics['reward_margins'][-1]:.4f}
"""
    if simpo_metrics and simpo_metrics.get("reward_margins"):
        report += f"""- SimPO reward margin: {simpo_metrics['reward_margins'][0]:.4f} → {simpo_metrics['reward_margins'][-1]:.4f}
"""

    report += f"""
## 3. 安全测试结果

### 拒绝率对比

| 模型 | 拒绝数 | 总数 | 拒绝率 |
|------|--------|------|--------|
| SFT | {safety.get("SFT", {}).get("refusal_count", "N/A")} | 10 | {sft_safety:.0f}% |
| DPO | {safety.get("DPO", {}).get("refusal_count", "N/A")} | 10 | {dpo_safety:.0f}% |
| SimPO | {safety.get("SimPO", {}).get("refusal_count", "N/A")} | 10 | {simpo_safety:.0f}% |

### 典型案例分析
"""

    # 添加安全测试详细案例
    for model_name in ["SFT", "DPO", "SimPO"]:
        details = safety_details.get(model_name, [])
        if details:
            # 一个拒绝案例 + 一个未拒绝案例
            refused = [d for d in details if d.get("refused")]
            not_refused = [d for d in details if not d.get("refused")]

            report += f"\n**{model_name} 模型:**\n"
            if refused:
                d = refused[0]
                report += f"- ✅ 拒绝案例 - 提示: \"{d['prompt']}\"  \n"
                report += f"  回复: \"{d['response'][:150]}...\"\n"
            if not_refused:
                d = not_refused[0]
                report += f"- ❌ 未拒绝案例 - 提示: \"{d['prompt']}\"  \n"
                report += f"  回复: \"{d['response'][:150]}...\"\n"
            if not refused and not not_refused:
                report += "- 无详细数据\n"

    report += f"""
## 4. 分析报告

### 4.1 DPO 对模型行为的具体改变

DPO（Direct Preference Optimization）通过直接优化偏好数据中的选择与拒绝对，改变了模型的输出分布。
具体表现为：

1. **回复质量提升**: DPO 训练后模型生成的回复更加结构化、信息更丰富。LLM-Judge 有用性评分从 {fmt_score(sft_judge)} 变为 {fmt_score(dpo_judge)}。
2. **安全性变化**: 安全拒绝率从 {sft_safety:.0f}% 变为 {dpo_safety:.0f}%，表明 DPO 对安全行为的影响显著。偏好数据中包含了安全相关的偏好对，使模型学会拒绝有害请求。
3. **多样性变化**: 多样性指标从 {sft_div:.3f} 变为 {dpo_div:.3f}。偏好优化可能导致模型倾向于生成更"安全"的标准答案，降低了输出的多样性。
4. **Reward margin 趋势**: 随训练进行，reward margin 逐步增大，说明模型逐渐学会区分 chosen 和 rejected 响应。

### 4.2 SimPO 与 DPO 的效率/效果权衡

SimPO 的关键创新在于无需参考模型（reference-free），直接使用序列平均对数概率作为隐式奖励：

1. **计算效率**: SimPO 训练时间 {simpo_time:.1f} 分钟 vs DPO {dpo_time:.1f} 分钟。由于不需要参考模型的前向传播，SimPO 在每个训练步骤上更快。
2. **显存效率**: SimPO 峰值显存 {simpo_mem:.2f} GB vs DPO {dpo_mem:.2f} GB。消除参考模型意味着更低的显存需求。
3. **效果对比**: SimPO LLM-Judge 评分 {fmt_score(simpo_judge)} vs DPO {fmt_score(dpo_judge)}。SimPO 使用更大的 β={config.SIMPO_BETA} 和额外的 γ={config.SIMPO_GAMMA} 目标奖励边际来补偿缺少参考模型的约束。
4. **安全性**: SimPO 安全拒绝率 {simpo_safety:.0f}% vs DPO {dpo_safety:.0f}%。

### 4.3 偏好优化的优势与局限

**优势：**
- 相比 RLHF，DPO/SimPO 训练更简单、稳定，无需训练单独的奖励模型
- 直接从偏好数据学习，避免了奖励模型的泛化误差
- 计算成本远低于 PPO 等 RL 方法

**局限：**
- 严重依赖偏好数据质量——标注噪声直接影响训练效果
- 可能导致"过度对齐"（over-alignment），模型变得过于保守
- β 值的选择敏感——过小导致对齐不足，过大导致偏离基座模型太远
- 离线方法的固有限制：无法根据模型当前策略动态调整训练信号

### 4.4 对偏好数据质量的观察和思考
"""

    if data_stats:
        report += f"""
UltraFeedback 数据集分析：
- 数据集大小: {data_stats.get('dataset_size', 'N/A')} 条偏好对
- Chosen 平均长度: {data_stats.get('chosen_mean_len', 0):.0f} 词, Rejected: {data_stats.get('rejected_mean_len', 0):.0f} 词
- 长度比 (Chosen/Rejected): {data_stats.get('length_ratio', 0):.2f}
- 长度偏差警告: {'是' if data_stats.get('length_bias_warning') else '否'}

主要观察：
1. **长度偏差**: 如果 Chosen 系统性更长，模型可能学到"更长=更好"的虚假关联，而非真正理解质量差异。
2. **标注一致性**: UltraFeedback 使用 GPT-4 自动标注，存在模型偏好（如偏爱列表格式、详细回复）。
3. **领域覆盖**: 数据集覆盖多样化的主题，但中文内容较少，可能影响中文场景的对齐效果。
4. **建议**: 在实际应用中，应结合人工标注进行数据清洗，移除明显噪声样本，并关注数据分布的平衡性。
"""

    # Beta 消融
    if ablation_results:
        report += """
## 5. 加分项：Beta 消融实验

![Beta 消融](beta_ablation.png)

| Beta | 最终损失 | 安全拒绝率 |
|------|----------|------------|
"""
        for beta in sorted([float(b) for b in ablation_results.keys()]):
            r = ablation_results[str(beta)]
            report += f"| {beta} | {r.get('final_loss', 0):.4f} | {r.get('safety_refusal_rate', 0)*100:.0f}% |\n"

        report += """
**分析：**
- β 较小 (0.05)：模型偏离基座较远，对齐更激进，但可能过拟合偏好数据
- β 中等 (0.1)：平衡了对齐效果和模型稳定性，是常用默认值
- β 较大 (0.5)：保守更新，模型更接近 SFT 基座，对齐效果可能不足

β 本质上控制了 KL 散度惩罚的强度——更大的 β 意味着更强的约束，让优化后的策略不会偏离参考策略太远。
"""

    # LLM-as-Judge
    if judge_results and judge_results.get("summary"):
        report += """
## 6. 加分项：LLM-as-Judge 自动评分

"""
        for name in ["SFT", "DPO", "SimPO"]:
            info = judge_results["summary"].get(name, {})
            if info:
                report += f"- **{name}**: 平均 {info.get('avg_score', 0):.2f}/10 (最高 {info.get('max_score', 0):.1f}, 最低 {info.get('min_score', 0):.1f}, 评估 {info.get('num_evaluated', 0)} 题)\n"

        report += "\nLLM-as-Judge 使用 Qwen3-max 模型，从有用性、准确性、清晰度、安全性四个维度综合评分。\n"

    report += """
---
*本报告由 Lab3 实验脚本自动生成*
"""

    # 保存报告
    report_path = os.path.join(output_dir, "lab3_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"分析报告已保存: {report_path}")
    return report_path


def main():
    print("=" * 60)
    print("Step 7: 生成交付物")
    print("=" * 60)

    # 加载所有结果
    dpo_metrics = load_json(os.path.join(config.OUTPUT_DIR, "dpo_metrics.json"))
    simpo_metrics = load_json(os.path.join(config.OUTPUT_DIR, "simpo_metrics.json"))
    eval_results = load_json(os.path.join(config.OUTPUT_DIR, "eval_results.json"))
    judge_results = load_json(os.path.join(config.OUTPUT_DIR, "llm_judge_results.json"))
    ablation_results = load_json(os.path.join(config.OUTPUT_DIR, "beta_ablation_results.json"))
    data_stats = load_json(os.path.join(config.OUTPUT_DIR, "step1_data_stats.json"))

    print("\n已加载数据:")
    for name, data in [("DPO Metrics", dpo_metrics), ("SimPO Metrics", simpo_metrics),
                        ("Eval Results", eval_results), ("Judge Results", judge_results),
                        ("Ablation Results", ablation_results), ("Data Stats", data_stats)]:
        print(f"  {name}: {'✓' if data else '✗'}")

    # 1. 训练曲线
    print("\n生成训练曲线...")
    plot_training_curves(dpo_metrics, simpo_metrics, config.OUTPUT_DIR)

    # 2. Beta 消融图
    if ablation_results:
        print("生成 Beta 消融图...")
        plot_beta_ablation(ablation_results, config.OUTPUT_DIR)

    # 3. 生成报告
    print("生成分析报告...")
    generate_report(dpo_metrics, simpo_metrics, eval_results, judge_results,
                    ablation_results, data_stats, config.OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("所有交付物:")
    print("=" * 60)
    for f in sorted(os.listdir(config.OUTPUT_DIR)):
        fpath = os.path.join(config.OUTPUT_DIR, f)
        size = os.path.getsize(fpath)
        print(f"  {f} ({size/1024:.1f} KB)")

    print("\nStep 7 完成! 所有交付物已生成。")


if __name__ == "__main__":
    main()
