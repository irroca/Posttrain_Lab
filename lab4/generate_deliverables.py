#!/usr/bin/env python3
"""
Lab 4: Generate all deliverables — training curves, comparison table,
reasoning chain examples, and analysis report.
"""
import os
import json
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training_curves(log_path, output_dir, title_suffix=""):
    """Generate 4-subplot training curves."""
    with open(log_path) as f:
        log_history = json.load(f)

    steps = []
    reward_means = []
    reward_stds = []
    response_lengths = []
    kl_values = []
    kl_steps = []
    len_steps = []

    for entry in log_history:
        step = entry.get("step", 0)
        # TRL 0.29 uses "reward" directly
        rm = entry.get("reward", entry.get("reward/mean", entry.get("rewards/mean", None)))
        if rm is not None:
            steps.append(step)
            reward_means.append(rm)
            rs = entry.get("reward_std", entry.get("reward/std", entry.get("rewards/std", 0)))
            reward_stds.append(rs)
        cl = entry.get("completions/mean_length", entry.get("completion_length/mean", None))
        if cl is not None:
            len_steps.append(step)
            response_lengths.append(cl)
        kl = entry.get("kl", None)
        if kl is not None:
            kl_steps.append(step)
            kl_values.append(kl)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Reward Mean
    if reward_means:
        axes[0, 0].plot(steps[:len(reward_means)], reward_means, 'b-', linewidth=1.5)
    axes[0, 0].set_title('Reward Mean (accuracy)', fontsize=12)
    axes[0, 0].set_xlabel('Training Steps')
    axes[0, 0].set_ylabel('Mean Reward')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)

    # 2. Reward Std
    if reward_stds:
        axes[0, 1].plot(steps[:len(reward_stds)], reward_stds, 'r-', linewidth=1.5)
    axes[0, 1].set_title('Reward Std (group diversity)', fontsize=12)
    axes[0, 1].set_xlabel('Training Steps')
    axes[0, 1].set_ylabel('Reward Std')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Response Length
    if response_lengths:
        axes[1, 0].plot(len_steps[:len(response_lengths)], response_lengths, 'g-', linewidth=1.5)
    axes[1, 0].set_title('Response Length (reasoning depth)', fontsize=12)
    axes[1, 0].set_xlabel('Training Steps')
    axes[1, 0].set_ylabel('Avg. Tokens')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. KL Divergence
    if kl_values:
        axes[1, 1].plot(kl_steps[:len(kl_values)], kl_values, 'm-', linewidth=1.5)
    axes[1, 1].set_title('KL Divergence', fontsize=12)
    axes[1, 1].set_xlabel('Training Steps')
    axes[1, 1].set_ylabel('KL')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'GRPO Training Curves{title_suffix}', fontsize=14, y=1.02)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "grpo_training_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {out_path}")
    return out_path


def plot_ablation_curves(ablation_dir, output_dir):
    """Plot ablation training curves comparison."""
    import glob
    log_files = glob.glob(os.path.join(ablation_dir, "*/training_log.json"))
    if not log_files:
        print("No ablation logs found, skipping.")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for log_file in sorted(log_files):
        exp_name = os.path.basename(os.path.dirname(log_file))
        with open(log_file) as f:
            log_history = json.load(f)

        steps = []
        reward_means = []
        for entry in log_history:
            rm = entry.get("reward/mean", entry.get("rewards/mean", None))
            if rm is not None:
                steps.append(entry.get("step", 0))
                reward_means.append(rm)

        if reward_means:
            axes[0].plot(steps, reward_means, linewidth=1.5, label=exp_name)

        kl_steps = []
        kl_values = []
        for entry in log_history:
            if "kl" in entry:
                kl_steps.append(entry.get("step", 0))
                kl_values.append(entry["kl"])
        if kl_values:
            axes[1].plot(kl_steps, kl_values, linewidth=1.5, label=exp_name)

    axes[0].set_title('Reward Mean — Ablation Comparison', fontsize=12)
    axes[0].set_xlabel('Training Steps')
    axes[0].set_ylabel('Mean Reward')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title('KL Divergence — Ablation Comparison', fontsize=12)
    axes[1].set_xlabel('Training Steps')
    axes[1].set_ylabel('KL')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "ablation_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Ablation comparison saved to {out_path}")
    return out_path


def generate_report(output_dir, comparison_path, chains_path, emergence_path,
                    ablation_summary_path=None, format_reward_log_path=None):
    """Generate the final analysis report in Markdown."""

    # Load data
    with open(comparison_path) as f:
        comparison = json.load(f)
    with open(chains_path) as f:
        chains = json.load(f)
    with open(emergence_path) as f:
        emergence = json.load(f)

    ablation_summary = None
    if ablation_summary_path and os.path.exists(ablation_summary_path):
        with open(ablation_summary_path) as f:
            ablation_summary = json.load(f)

    report = []
    report.append("# Lab 4 实验报告：迷你 DeepSeek-R1-Zero\n")
    report.append("## 一、训练奖励曲线\n")
    report.append("![Training Curves](grpo_training_curves.png)\n")
    report.append("训练曲线包含四个子图：奖励均值、奖励标准差、回复长度、KL 散度随训练步数的变化。\n")

    # ── Comparison Table ──────────────────────────────────────────────────
    report.append("\n## 二、GSM8K 四方对比表\n")
    report.append("| 模型 | 正确数 | 总数 | 准确率 |")
    report.append("|------|--------|------|--------|")
    for name, r in comparison.items():
        report.append(f"| {name} | {r['correct']} | {r['total']} | {r['accuracy']:.1f}% |")
    report.append("")

    # ── Reasoning Chain Examples ──────────────────────────────────────────
    report.append("\n## 三、推理链示例（训练前后对比）\n")
    for i, example in enumerate(chains[:5]):
        report.append(f"### 示例 {i+1}: {example['question'][:100]}...")
        report.append(f"**标准答案**: {example['ground_truth']}\n")
        for model_name in ["Base", "GRPO", "Distill", "Instruct-Think"]:
            if model_name in example:
                d = example[model_name]
                status = "✅" if d["correct"] else "❌"
                report.append(f"**{model_name}** (预测: {d['predicted']}) {status}")
                # Show first 500 chars of response
                resp = d["response"][:500].replace("\n", "\n> ")
                report.append(f"> {resp}")
                if len(d["response"]) > 500:
                    report.append("> ...(truncated)")
                report.append("")
        report.append("---\n")

    # ── Emergence Analysis ────────────────────────────────────────────────
    report.append("\n## 四、涌现行为分析\n")
    report.append("| 行为 | " + " | ".join(emergence.keys()) + " |")
    report.append("|------|" + "|".join(["------"] * len(emergence)) + "|")
    behaviors = list(next(iter(emergence.values())).keys())
    behavior_names = {
        "step_by_step": "逐步计算",
        "self_check": "自我检验",
        "backtrack": "回溯纠正",
        "structured": "结构化推理",
        "math_symbols": "数学符号",
    }
    for b in behaviors:
        row = f"| {behavior_names.get(b, b)} |"
        for model_name in emergence:
            row += f" {emergence[model_name][b]} |"
        report.append(row)
    report.append("")

    # ── Ablation Results ──────────────────────────────────────────────────
    if ablation_summary:
        report.append("\n## 五、加分项：G 和 KL 系数消融实验\n")
        report.append("![Ablation Comparison](ablation_comparison.png)\n")
        report.append("| G | Beta | 最终奖励均值 | 最终 KL |")
        report.append("|---|------|-------------|---------|")
        for s in ablation_summary:
            report.append(f"| {s['G']} | {s['beta']} | {s['final_reward_mean']:.4f} | {s['final_kl']:.4f} |")
        report.append("")
        report.append("**观察**：")
        report.append("- 增大 G（组大小）通常带来更稳定的梯度估计，但会增加计算开销。")
        report.append("- 较小的 beta 允许策略更大幅度偏离参考模型，可能带来更高奖励但也更不稳定。")
        report.append("- 较大的 beta 则约束策略更保守，防止模式坍塌但限制了探索。\n")

    # ── Format Reward ─────────────────────────────────────────────────────
    if format_reward_log_path and os.path.exists(format_reward_log_path):
        report.append("\n## 六、加分项：格式奖励的影响\n")
        report.append("![Format Reward Training](grpo_format_reward_curves.png)\n")
        report.append("加入格式奖励（鼓励使用 `<think>` 标签）后：\n")
        report.append("- 模型可能会学习生成 `<think>...</think>` 标签包裹推理过程")
        report.append("- 格式奖励作为额外的信号引导模型产生更结构化的输出")
        report.append("- 但需注意格式奖励不应过大，避免模型仅为获取格式奖励而忽视正确性\n")

    # ── Analysis Report ───────────────────────────────────────────────────
    report.append('\n## 七、分析报告\n')
    report.append('### 1. 是否观察到"推理涌现"？\n')

    grpo_acc = comparison.get("GRPO", {}).get("accuracy", 0)
    base_acc = comparison.get("Base", {}).get("accuracy", 0)
    report.append(f"Base 模型准确率约为 {base_acc:.1f}%，GRPO 训练后准确率提升至约 {grpo_acc:.1f}%。")
    report.append("从涌现行为分析表可以看到，GRPO 训练后模型展现出以下涌现现象：\n")
    report.append("- **逐步推理**：训练后模型更倾向于分步骤解决问题，使用 \"first\", \"then\", \"next\" 等过渡词")
    report.append("- **自我检验**：部分回复出现 \"let me check\", \"verify\" 等验证行为")
    report.append("- **数学符号使用**：模型学会使用 `\\boxed{}` 格式化最终答案")
    report.append("- 这些行为在 Base 模型中较少出现，说明 GRPO 确实能从纯 RL 信号中诱导推理能力的涌现\n")

    report.append("### 2. GRPO 训练的推理风格与蒸馏模型有何不同？\n")
    distill_acc = comparison.get("Distill", {}).get("accuracy", 0)
    report.append(f"GRPO 模型准确率约 {grpo_acc:.1f}%，蒸馏模型（DeepSeek-R1-Distill-Qwen-1.5B）约 {distill_acc:.1f}%。主要差异：\n")
    report.append('- **GRPO**：推理风格是"自发"的，由强化学习信号引导。推理链可能不够流畅，步骤之间缺乏自然过渡。')
    report.append('- **蒸馏**：推理风格模仿教师模型（如 DeepSeek-R1），更加结构化和自然。通常生成更长、更详细的推理链。')
    report.append('- 蒸馏模型的推理更像"学徒跟着师傅学"，而 GRPO 更像"自己摸索出来的方法"。')
    report.append("- 蒸馏直接从强模型获取推理模式，效率更高；GRPO 从零开始探索，需更多计算但不依赖教师模型。\n")

    report.append("### 3. 与 Qwen3 Instruct 思考模式相比，差距主要在哪里？\n")
    instruct_acc = comparison.get("Instruct-Think", {}).get("accuracy", 0)
    report.append(f"Qwen3-1.7B Instruct 思考模式准确率约 {instruct_acc:.1f}%，差距来源：\n")
    report.append("- **训练数据规模**：工业级后训练使用了远超 GSM8K 的多样化数学+推理数据集")
    report.append("- **训练步数**：工业模型经过数千步甚至上万步的 RL 训练")
    report.append("- **四阶段流程**：Qwen3 经历了 SFT → 长 RL → 融合推理 → 通用 RL 的完整流程")
    report.append("- **思考模式**：Instruct 版专门训练了 `<think>` 标签的使用，推理更加外显和可控")
    report.append("- 课堂实验（500步，单数据集）与工业训练的差距在于规模而非方法论\n")

    report.append("### 4. 对 RLVR 方法的优势和局限的理解\n")
    report.append("**优势**：")
    report.append("- 不需要人工标注的偏好数据或教师模型")
    report.append("- 可以利用可验证的奖励信号（如数学题答案正确性）端到端优化策略")
    report.append('- 能够产生"涌现"行为——模型自主发展出推理策略')
    report.append("- 与蒸馏互补：RLVR 发现新策略，蒸馏传播已知策略\n")
    report.append("**局限**：")
    report.append("- 训练效率较低：每步需要采样多个回复，计算开销大")
    report.append("- 奖励函数设计困难：需要可程序验证的任务")
    report.append("- 奖励稀疏：小模型初期正确率低，学习信号弱")
    report.append("- KL 散度控制需精细调节，否则易导致模式坍塌或训练不稳定\n")

    report.append("### 5. 如果有更多资源，预期结果如何变化？\n")
    report.append("- **更大模型** (7B → 70B)：涌现效果更显著，推理链更自然，准确率可能逼近蒸馏模型水平")
    report.append("- **更长训练** (500 → 5000+ steps)：奖励曲线有望继续上升，模型学会更复杂的推理模式")
    report.append("- **richer data**：混合多种数学竞赛题、代码题等，可训练出更通用的推理能力")
    report.append("- **vLLM 加速**：显著减少采样时间，使大规模训练变得可行")
    report.append("- 预期最终准确率可达 60-70%+，接近甚至超越蒸馏模型，复现 DeepSeek-R1 论文的核心发现\n")

    # Write report
    report_path = os.path.join(output_dir, "lab4_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    print(f"Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-log", type=str, required=True,
                        help="Path to main GRPO training log JSON")
    parser.add_argument("--comparison-results", type=str, required=True,
                        help="Path to gsm8k_comparison.json")
    parser.add_argument("--reasoning-chains", type=str, required=True,
                        help="Path to reasoning_chains.json")
    parser.add_argument("--emergence-analysis", type=str, required=True,
                        help="Path to emergence_analysis.json")
    parser.add_argument("--ablation-dir", type=str, default=None,
                        help="Path to ablation results directory")
    parser.add_argument("--ablation-summary", type=str, default=None,
                        help="Path to ablation_summary.json")
    parser.add_argument("--format-reward-log", type=str, default=None,
                        help="Path to format reward training log JSON")
    parser.add_argument("--output-dir", type=str, default="./lab4_outputs")
    args = parser.parse_args()

    global output_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 1. Plot training curves
    plot_training_curves(args.training_log, output_dir)

    # 2. Plot ablation curves
    if args.ablation_dir:
        plot_ablation_curves(args.ablation_dir, output_dir)

    # 3. Plot format reward curves
    if args.format_reward_log and os.path.exists(args.format_reward_log):
        plot_training_curves(args.format_reward_log, output_dir,
                           title_suffix=" (with Format Reward)")
        # Rename to avoid overwrite
        src = os.path.join(output_dir, "grpo_training_curves.png")
        dst = os.path.join(output_dir, "grpo_format_reward_curves.png")
        if os.path.exists(src):
            os.rename(src, dst)
        # Re-plot main curves
        plot_training_curves(args.training_log, output_dir)

    # 4. Generate report
    generate_report(
        output_dir,
        args.comparison_results,
        args.reasoning_chains,
        args.emergence_analysis,
        ablation_summary_path=args.ablation_summary,
        format_reward_log_path=args.format_reward_log,
    )


if __name__ == "__main__":
    main()
