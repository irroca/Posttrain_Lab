#!/usr/bin/env python3
"""
生成实验报告：读取实验A和实验B结果，生成完整的Markdown报告和图表。
"""

import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = "/scratch/azureml/cr/j/7e6b762e2e0d44f990d5daffc11d8310/exe/wd/Posttrain_Lab/lab5"

# ===================================================================
# Load results
# ===================================================================

with open(os.path.join(OUTPUT_DIR, "experiment_a_results.json"), "r", encoding="utf-8") as f:
    exp_a = json.load(f)

with open(os.path.join(OUTPUT_DIR, "experiment_b_vlm_results.json"), "r", encoding="utf-8") as f:
    exp_b = json.load(f)

memory_loading = exp_a["memory_loading"]
speed = exp_a["speed"]
scores = exp_a["quality_scores"]

# ===================================================================
# Generate Charts
# ===================================================================

fig_dir = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(fig_dir, exist_ok=True)

precisions = ["fp16", "int8", "int4"]
precision_labels = ["FP16", "INT8", "INT4"]

# Chart 1: Memory comparison
fig, ax = plt.subplots(figsize=(8, 5))
mem_vals = [memory_loading[p]["memory_gb"] for p in precisions]
bars = ax.bar(precision_labels, mem_vals, color=['#2196F3', '#FF9800', '#4CAF50'], edgecolor='black')
for bar, val in zip(bars, mem_vals):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
            f'{val:.2f} GB', ha='center', va='bottom', fontweight='bold')
ax.set_ylabel('VRAM (GB)')
ax.set_title('Qwen3-8B VRAM Usage by Precision')
ax.set_ylim(0, max(mem_vals) * 1.2)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "memory_comparison.png"), dpi=150)
plt.close()

# Chart 2: Speed comparison (combined)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

tps_vals = [speed[p]["tokens_per_sec"] for p in precisions]
bars1 = ax1.bar(precision_labels, tps_vals, color=['#2196F3', '#FF9800', '#4CAF50'], edgecolor='black')
for bar, val in zip(bars1, tps_vals):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f'{val}', ha='center', va='bottom', fontweight='bold')
ax1.set_ylabel('Tokens/s')
ax1.set_title('Inference Speed (tokens/s)')
ax1.set_ylim(0, max(tps_vals) * 1.2)

ftl_vals = [speed[p]["avg_first_token_ms"] for p in precisions]
bars2 = ax2.bar(precision_labels, ftl_vals, color=['#2196F3', '#FF9800', '#4CAF50'], edgecolor='black')
for bar, val in zip(bars2, ftl_vals):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
            f'{val} ms', ha='center', va='bottom', fontweight='bold')
ax2.set_ylabel('First Token Latency (ms)')
ax2.set_title('First Token Latency')
ax2.set_ylim(0, max(ftl_vals) * 1.2)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "speed_comparison.png"), dpi=150)
plt.close()

# Chart 3: Quality scores comparison
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(3)
width = 0.25
tasks = ["gsm8k", "instruction", "chinese"]
task_labels = ["GSM8K Math", "Instruction Following", "Chinese Tasks"]

for i, p in enumerate(precisions):
    task_avgs = []
    for t in tasks:
        s = scores[t][p]
        avg = sum(s) / len(s) if s else 0
        task_avgs.append(avg)
    bars = ax.bar(x + i * width, task_avgs, width, label=precision_labels[i],
                  color=['#2196F3', '#FF9800', '#4CAF50'][i], edgecolor='black')
    for bar, val in zip(bars, task_avgs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)

ax.set_ylabel('Average Score (1-10)')
ax.set_title('Quality Scores by Task and Precision')
ax.set_xticks(x + width)
ax.set_xticklabels(task_labels)
ax.legend()
ax.set_ylim(0, 11)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "quality_scores.png"), dpi=150)
plt.close()

# Chart 4: Compression rate vs quality retention
fig, ax = plt.subplots(figsize=(8, 6))
fp16_mem = memory_loading["fp16"]["memory_gb"]
compression_rates = [fp16_mem / memory_loading[p]["memory_gb"] for p in precisions]

# Average quality retention per precision
fp16_avg_all = 0
quality_retention = []
for p in precisions:
    avg_score = 0
    count = 0
    for t in tasks:
        s = scores[t][p]
        avg_score += sum(s)
        count += len(s)
    avg_all = avg_score / count if count > 0 else 0
    if p == "fp16":
        fp16_avg_all = avg_all
    quality_retention.append(avg_all / fp16_avg_all * 100 if fp16_avg_all > 0 else 100)

colors = ['#2196F3', '#FF9800', '#4CAF50']
for i, (cr, qr, label) in enumerate(zip(compression_rates, quality_retention, precision_labels)):
    ax.scatter(cr, qr, s=200, c=colors[i], edgecolors='black', zorder=5, label=label)
    ax.annotate(f'{label}\n({cr:.1f}x, {qr:.1f}%)',
                (cr, qr), textcoords="offset points", xytext=(10, 10), fontsize=10)

ax.plot(compression_rates, quality_retention, 'k--', alpha=0.5)
ax.set_xlabel('Compression Rate (x)')
ax.set_ylabel('Quality Retention (%)')
ax.set_title('Compression Rate vs Quality Retention')
ax.set_ylim(min(quality_retention) - 5, 105)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "compression_vs_quality.png"), dpi=150)
plt.close()

# Chart 5: VLM results - Category accuracy bar chart (enhanced: 8 categories)
fig, ax = plt.subplots(figsize=(12, 6))
cat_stats = exp_b["category_stats"]
cat_names = list(cat_stats.keys())
cat_accs = [cat_stats[c]["accuracy_pct"] for c in cat_names]
cat_colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#00BCD4', '#E91E63', '#F44336', '#795548']
bars = ax.barh(cat_names, cat_accs, color=cat_colors[:len(cat_names)], edgecolor='black')
for bar, val in zip(bars, cat_accs):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2.,
            f'{val:.1f}%', ha='left', va='center', fontweight='bold')
ax.set_xlabel('Accuracy (%)')
ax.set_title(f'Qwen2.5-VL-7B VQA Category Accuracy (8 Categories, {sum(s["total"] for s in cat_stats.values())} Tasks)')
ax.set_xlim(0, 115)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "vlm_category_accuracy.png"), dpi=150)
plt.close()

# Chart 6: Hallucination pie chart
fig, ax = plt.subplots(figsize=(6, 6))
h_stats = exp_b["hallucination_stats"]
sizes = [h_stats["hallucination_count"], h_stats["total_tests"] - h_stats["hallucination_count"]]
labels = [f'Hallucinated\n({sizes[0]})', f'Correct\n({sizes[1]})']
colors_pie = ['#F44336', '#4CAF50']
ax.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
       startangle=90, textprops={'fontsize': 12})
ax.set_title(f'Hallucination Rate: {h_stats["hallucination_rate_pct"]:.1f}%')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "hallucination_rate.png"), dpi=150)
plt.close()

# Chart 7: Per-category latency bar chart
fig, ax = plt.subplots(figsize=(12, 5))
cat_lats = [cat_stats[c]["avg_latency_ms"] for c in cat_names]
bars = ax.barh(cat_names, cat_lats, color=cat_colors[:len(cat_names)], edgecolor='black', alpha=0.8)
for bar, val in zip(bars, cat_lats):
    ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2.,
            f'{val:.0f} ms', ha='left', va='center', fontweight='bold')
ax.set_xlabel('Average Latency (ms)')
ax.set_title('Per-Category Average Inference Latency')
ax.set_xlim(0, max(cat_lats) * 1.25)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "vlm_category_latency.png"), dpi=150)
plt.close()

print("All charts generated successfully.")

# ===================================================================
# Generate Markdown Report
# ===================================================================

# Compute values
load_times = {p: memory_loading[p]["load_time_s"] for p in precisions}
mem_vals_dict = {p: memory_loading[p]["memory_gb"] for p in precisions}
tps_dict = {p: speed[p]["tokens_per_sec"] for p in precisions}
ftl_dict = {p: speed[p]["avg_first_token_ms"] for p in precisions}

# Score averages per task per precision
score_avgs = {}
for t in tasks:
    score_avgs[t] = {}
    for p in precisions:
        s = scores[t][p]
        score_avgs[t][p] = round(sum(s) / len(s), 1) if s else 0

# Find which task has biggest quality drop
task_drops = {}
for t in tasks:
    fp16_s = score_avgs[t]["fp16"]
    int4_s = score_avgs[t]["int4"]
    drop = fp16_s - int4_s
    task_drops[t] = drop
worst_task = max(task_drops, key=task_drops.get)
task_name_map = {"gsm8k": "GSM8K 数学推理", "instruction": "指令跟随", "chinese": "中文任务"}

# VLM results
vlm_info = exp_b["model_info"]
overall_vqa_acc = exp_b["overall_accuracy_pct"]
h_rate = exp_b["hallucination_stats"]["hallucination_rate_pct"]
cat_stats_data = exp_b["category_stats"]
avg_lat = exp_b["avg_latency_ms"]

report = f"""# 第5课 上机实验报告

## 一、量化实验报告（必做）

### 1.1 实验环境

- **模型**: Qwen3-8B (Qwen/Qwen3-8B)
- **GPU**: NVIDIA A100-SXM4-80GB x 4 (GPU 4,5,6,7)
- **框架**: PyTorch + Transformers + BitsAndBytes
- **精度**: FP16 / INT8 (bitsandbytes) / INT4 (NF4 + Double Quant)

### 1.2 三种精度的显存占用对比表

| 精度 | 显存占用 (GB) | 加载时间 (s) | 相对FP16显存比 |
|------|--------------|-------------|---------------|
| FP16 | {mem_vals_dict['fp16']} | {load_times['fp16']} | 1.00x |
| INT8 | {mem_vals_dict['int8']} | {load_times['int8']} | {mem_vals_dict['int8']/mem_vals_dict['fp16']:.2f}x |
| INT4 | {mem_vals_dict['int4']} | {load_times['int4']} | {mem_vals_dict['int4']/mem_vals_dict['fp16']:.2f}x |

![Memory Comparison](figures/memory_comparison.png)

### 1.3 推理速度对比表

| 精度 | Tokens/s | 首Token延迟 (ms) |
|------|----------|-----------------|
| FP16 | {tps_dict['fp16']} | {ftl_dict['fp16']} |
| INT8 | {tps_dict['int8']} | {ftl_dict['int8']} |
| INT4 | {tps_dict['int4']} | {ftl_dict['int4']} |

![Speed Comparison](figures/speed_comparison.png)

### 1.4 三类任务的质量评分对比表（LLM-as-Judge, 1-10分）

| 任务类型 | FP16 | INT8 | INT4 | INT4质量保持率 |
|---------|------|------|------|--------------|
| GSM8K 数学推理 | {score_avgs['gsm8k']['fp16']} | {score_avgs['gsm8k']['int8']} | {score_avgs['gsm8k']['int4']} | {score_avgs['gsm8k']['int4']/score_avgs['gsm8k']['fp16']*100:.1f}% |
| 指令跟随 | {score_avgs['instruction']['fp16']} | {score_avgs['instruction']['int8']} | {score_avgs['instruction']['int4']} | {score_avgs['instruction']['int4']/score_avgs['instruction']['fp16']*100:.1f}% |
| 中文任务 | {score_avgs['chinese']['fp16']} | {score_avgs['chinese']['int8']} | {score_avgs['chinese']['int4']} | {score_avgs['chinese']['int4']/score_avgs['chinese']['fp16']*100:.1f}% |

![Quality Scores](figures/quality_scores.png)

### 1.5 压缩率 vs 质量保持率

![Compression vs Quality](figures/compression_vs_quality.png)

- **FP16**: 压缩率 {compression_rates[0]:.1f}x, 质量保持 {quality_retention[0]:.1f}%
- **INT8**: 压缩率 {compression_rates[1]:.1f}x, 质量保持 {quality_retention[1]:.1f}%
- **INT4**: 压缩率 {compression_rates[2]:.1f}x, 质量保持 {quality_retention[2]:.1f}%

### 1.6 分析：量化对哪类任务影响最大？为什么？

根据实验数据，量化对 **{task_name_map[worst_task]}** 任务影响最大（FP16→INT4 分数下降 {task_drops[worst_task]:.1f} 分）。

**分析原因：**

1. **数学推理任务（GSM8K）** 对数值精度高度敏感。量化会导致权重的微小数值偏差，这些偏差在多步推理链中会累积放大，导致最终计算结果出错。即使单步误差很小，经过多步逻辑推理后，错误率会显著上升。

2. **指令跟随任务** 对格式控制要求精确，INT4 量化可能导致模型对特定格式指令的理解出现偏差（如"恰好三个词"、"JSON格式"等精确约束），但一般性的指令跟随能力相对鲁棒。

3. **中文理解/生成任务** 主要依赖语义理解和文本生成能力，这些能力分布在模型的大量参数中，具有较好的冗余性，因此对量化的鲁棒性相对较强。

**总结：** 需要精确数值计算和严格逻辑推理的任务受量化影响最大，而偏向语义理解和开放式生成的任务则具有更好的量化鲁棒性。在实际部署中，如果主要场景是对话和文本生成，INT4 量化是性价比极高的选择；但如果涉及数学推理或代码生成等精确任务，建议至少使用 INT8 精度。
"""

report += f"""---

## 二、选做实验报告：选项 2 - 迷你多模态实验（增强版）

### 2.1 实验设置

- **模型**: {vlm_info['model_name']}
- **参数量**: {vlm_info['param_count_B']}B
- **显存占用**: {vlm_info['memory_gb']} GB
- **加载时间**: {vlm_info['load_time_s']}s
- **测试图像**: 10 张合成图像（几何形状、数学题、柱状图、场景图、钟表、空间布局、表格、对比面板、会议记录、抽象图案）
- **评估类别**: 8 大类共 {sum(s['total'] for s in cat_stats_data.values())} 个测试任务
- **评估维度**: 基础VQA + 空间推理 + OCR + 图表理解 + 视觉推理 + 幻觉检测 + 多轮对话
- **平均推理延迟**: {avg_lat:.0f}ms

### 2.2 八大类别的准确率对比

| 任务类别 | 正确数 | 总数 | 准确率 | 平均延迟(ms) |
|---------|-------|------|-------|-------------|
"""

cat_name_zh = {
    "object_recognition": "物体识别与计数",
    "ocr_text": "OCR文字识别",
    "chart_understanding": "图表数据理解",
    "spatial_reasoning": "空间推理",
    "time_reading": "时间读取",
    "visual_reasoning": "对比与视觉推理",
    "hallucination_absence": "幻觉检测（缺失物体）",
    "hallucination_misattribution": "幻觉检测（错误归因）",
}

for cat, stats in cat_stats_data.items():
    zh = cat_name_zh.get(cat, cat)
    report += f"| {zh} | {stats['correct']} | {stats['total']} | {stats['accuracy_pct']:.1f}% | {stats['avg_latency_ms']:.0f} |\n"

report += f"| **总计** | **{sum(s['correct'] for s in cat_stats_data.values())}** | **{sum(s['total'] for s in cat_stats_data.values())}** | **{overall_vqa_acc:.1f}%** | **{avg_lat:.0f}** |\n"

report += f"""
![VLM Category Accuracy](figures/vlm_category_accuracy.png)

![Per-Category Latency](figures/vlm_category_latency.png)

### 2.3 分类别详细分析

**高准确率类别 (100%)**:
- **物体识别与计数**: 模型能准确识别并列举所有形状及其颜色，精确计数。
- **OCR文字识别**: 能正确读取数学公式、会议记录和表格数据，并提取关键信息。
- **空间推理**: 能准确描述物体相对位置（上下左右、中央等）。
- **时间读取**: 能从钟表图像读取时间（但存在分钟级别误差）。
- **错误归因幻觉检测**: 能正确归因表格中不同人的数据，不混淆。

**中等准确率类别**:
- **对比与视觉推理 ({cat_stats_data.get('visual_reasoning', {}).get('accuracy_pct', 0):.0f}%)**: 在图案序列推理和多步逻辑计算方面表现较好，但精确计数有时出错。
- **幻觉检测 ({cat_stats_data.get('hallucination_absence', {}).get('accuracy_pct', 0):.0f}%)**: 多数情况能正确拒绝不存在的物体，但在诱导性问题（如询问秒针）下仍会产生幻觉。

**较低准确率类别**:
- **图表数据理解 ({cat_stats_data.get('chart_understanding', {}).get('accuracy_pct', 0):.0f}%)**: 能正确读取单个柱状图数据，但多数值求和计算容易出错。

### 2.4 幻觉率统计

通过两类幻觉测试（缺失物体问询 + 错误归因验证）评估模型幻觉问题：

- **总幻觉测试数**: {exp_b['hallucination_stats']['total_tests']}
- **产生幻觉数**: {exp_b['hallucination_stats']['hallucination_count']}
- **幻觉率**: {h_rate:.1f}%

![Hallucination Rate](figures/hallucination_rate.png)

**幻觉分析：**

实验中发现 VLM 在以下场景更容易产生幻觉：
1. **秒针问题**: 钟表图像中没有秒针，但模型在被问及"秒针指向几"时编造了答案，说明模型倾向于"回答问题"而非"质疑前提"。
2. **语言误导**: 当问"图片下方有一行中文说明文字"时，图片中实际只有英文标签，但模型将英文标签翻译成中文来回答，属于过度推理。
3. **缺失物体检测**: 对于明确不存在的物体（汽车、鸟、Panel C、Bob手机号），模型能正确拒绝回答，说明模型在大多数场景下具备基本的诚实性。

### 2.5 多轮视觉对话测试

进行了 3 组多轮对话测试，模拟真实交互场景：

"""

# Add multiturn results
if "multiturn_results" in exp_b:
    for mt in exp_b["multiturn_results"]:
        report += f"**{mt['description']}** (图像: {mt['image']})\n\n"
        for turn in mt["turns"]:
            report += f"- **User [{turn['turn']}]**: {turn['user']}\n"
            resp_short = turn['assistant'][:300] + ('...' if len(turn['assistant']) > 300 else '')
            report += f"- **Assistant**: {resp_short}\n"
            report += f"- *延迟: {turn['latency_ms']:.0f}ms*\n\n"

report += """**多轮对话分析：**
- 模型在多轮交互中能保持上下文一致性，后续回答能正确引用之前对话中建立的信息。
- 表格分析场景中，模型能准确执行排序、筛选等多步数据操作。
- 图表多轮分析中，模型能从概述逐步深入到具体数值查询和聚合计算。

### 2.6 减少幻觉的建议

基于实验观察，提出以下减少 VLM 幻觉的策略：

1. **提示工程**: 在 system prompt 中明确要求"如果无法确认某个信息，请说明无法确认而不是猜测"
2. **RLHF/DPO 对齐**: 在后训练阶段使用包含幻觉惩罚的偏好数据
3. **Chain-of-Thought**: 要求模型先逐步描述图像内容，再回答具体问题
4. **自我验证**: 让模型生成答案后再检查一遍是否与图像一致
5. **更大模型/更好预训练**: 72B 及以上模型的幻觉率显著低于 7B 级别
"""

report += """
---

## 三、总结反思

### 3.1 后训练各环节的关系与选择策略

后训练流程可以概括为：**SFT → DPO → GRPO → 量化/蒸馏**，各环节承担不同职责：

| 环节 | 目标 | 输入 | 关键技术 |
|------|------|------|---------|
| **SFT** | 赋予模型指令跟随能力 | 指令-回复对 | Full fine-tuning, LoRA |
| **DPO** | 对齐人类偏好 | 偏好对数据 | 直接偏好优化，无需奖励模型 |
| **GRPO** | 强化特定能力（如推理） | 可验证的任务 | 群组相对策略优化 |
| **量化** | 压缩模型、降低部署成本 | 训练好的模型 | INT8/INT4, NF4 |
| **蒸馏** | 将大模型能力迁移到小模型 | 教师模型 + 数据 | KD, 推理链蒸馏 |

**选择策略：**

1. **SFT 是基础**：几乎所有后训练流程都需要 SFT 作为起点，它决定了模型的基本对话能力。
2. **DPO vs GRPO 的选择**：
   - 如果有高质量偏好数据且目标是通用对齐，选 DPO
   - 如果需要强化特定可验证能力（数学、代码、搜索），选 GRPO
   - 两者可以组合使用：先 DPO 对齐，再 GRPO 增强
3. **量化 vs 蒸馏的选择**：
   - 量化几乎零成本，适合所有部署场景
   - 蒸馏需要额外训练成本，但能实现更大的模型压缩比
4. **流水线组合**：可以先做 GRPO 增强推理能力，再蒸馏到小模型，最后量化部署

### 3.2 实际 LLM 应用项目的技术组合建议

对于一个实际的 LLM 应用项目，我的技术选择策略如下：

**场景 1：通用对话助手**
```
Base Model → SFT（高质量对话数据）→ DPO（人类偏好）→ INT4 量化部署
```
- 重点在 SFT 和 DPO 环节的数据质量
- INT4 量化可大幅降低部署成本，对话质量损失可接受

**场景 2：数学/代码推理助手**
```
Base Model → SFT → GRPO（数学/代码奖励）→ INT8 量化部署
```
- 使用 GRPO 强化推理能力
- 选择 INT8 而非 INT4，因为推理任务对精度更敏感

**场景 3：边缘设备部署**
```
大模型训练好 → 蒸馏到小模型 → INT4 量化 → 部署
```
- 蒸馏+量化组合实现极致压缩
- 适合手机、IoT 等资源受限场景

**场景 4：多模态应用**
```
VLM Base → SFT（多模态数据）→ DPO（幻觉惩罚）→ 量化部署
```
- 多模态对齐重点解决幻觉问题
- DPO 中加入幻觉惩罚的偏好数据

**核心原则：**
1. 数据质量 > 训练技巧 > 模型大小
2. 先保证功能正确，再优化部署效率
3. 量化是"免费午餐"，应默认使用
4. 蒸馏在需要大幅压缩时才考虑
5. 持续评估是关键——每个环节都需要系统化评测
"""

# Write report
report_path = os.path.join(OUTPUT_DIR, "lab5_report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)

print(f"Report generated: {report_path}")
print("Done!")
