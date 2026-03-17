# Lab 2：构建领域定制 SFT 模型并系统评估

> 《大语言模型后训练实践》课程 — 第二次实验  
> 课程页面：https://posttrain.gaozhijun.me/docs/lecture-2/lab/

## 实验目标

在 Lab 1 的基础上，进一步掌握 **数据工程**、**指令微调 (SFT)** 和 **自动化评估** 的完整流程：

1. 对真实中文指令数据集进行质量分析与清洗
2. 使用 LoRA 在 Qwen3-1.7B-Base 上训练领域定制 SFT 模型
3. 基于 LLM-as-Judge 方法对比多模型表现
4. 通过消融实验量化 **数据质量** 对 SFT 效果的影响

## 环境要求

| 项目 | 版本/规格 |
|------|-----------|
| GPU | NVIDIA A100 80GB × 1+ |
| Python | 3.12 |
| PyTorch | 2.10+ |
| transformers | 5.3+ |
| trl | 0.29+ |
| peft | 0.18+ |
| datasets | 4.7+ |
| bitsandbytes | latest |
| matplotlib | 3.10+ |
| openai | latest (用于 qwen3-max API 调用) |

```bash
# 使用已有 conda 环境
conda activate test

# 或安装依赖
pip install torch transformers trl peft datasets bitsandbytes matplotlib openai
```

## 项目结构

```
test2/
├── config.py                # 全局配置（模型/数据/训练/评估参数）
├── run_all.sh               # 一键运行全部步骤
├── step1_data_analysis.py   # 步骤1: 数据集分析与预处理
├── step2_train.py           # 步骤2: LoRA 指令微调训练
├── step3_eval.py            # 步骤3: LLM-as-Judge 多模型评估
├── step4_ablation.py        # 步骤4: 数据质量消融实验
├── step5_report.py          # 步骤5: 生成交付物报告
├── regenerate_plots.py      # 辅助: 重新生成图表
├── lecture2-sft/            # SFT 模型输出目录
│   ├── best/                #   最佳 checkpoint
│   ├── ablation_raw/        #   消融: 原始数据模型
│   ├── ablation_dedup/      #   消融: 去重数据模型
│   └── ablation_clean/      #   消融: 完整清洗数据模型
└── outputs/                 # 实验结果输出
    ├── lab2_report.md       #   完整实验报告（最终交付物）
    ├── datasets/            #   预处理后的数据集
    ├── data_analysis.json   #   数据分析结果
    ├── train_metrics.json   #   训练指标
    ├── eval_results.json    #   LLM-as-Judge 评估结果
    ├── ablation_results.json#   消融实验结果
    ├── *.png                #   各类图表
    └── step*.log            #   各步骤运行日志
```

## 快速开始

### 一键运行

```bash
cd test2
# 修改 config.py 中的 DASHSCOPE_API_KEY（用于 qwen3-max Judge）
# 修改 CUDA_VISIBLE_DEVICES 选择空闲 GPU
bash run_all.sh
```

### 分步运行

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
conda activate test

# 步骤1: 数据集分析与预处理
python step1_data_analysis.py

# 步骤2: LoRA SFT 训练
python step2_train.py

# 步骤3: LLM-as-Judge 评估（基座 / Lab1 SFT / Lab2 SFT）
python step3_eval.py

# 步骤4: 数据质量消融实验
python step4_ablation.py

# 步骤5: 生成最终报告
python step5_report.py
```

## 各步骤说明

### 步骤 1：数据集分析与预处理

- **数据源**：[m-a-p/COIG-CQIA](https://huggingface.co/datasets/m-a-p/COIG-CQIA) 的 `zhihu` 子集（5631 条）
- **质量控制流程**：
  - 内容哈希去重 → 去除 168 条重复
  - Token 长度过滤（助手回复 ≥ 10 tokens，总长 ≤ 2048 tokens）→ 去除 29 条
  - 格式一致性检查
- **最终数据**：5434 条（去除 3.5%），按 80/10/10 划分为训练/验证/测试集
- **输出**：Token 长度分布图、数据分析 JSON

### 步骤 2：LoRA 指令微调

| 参数 | 值 |
|------|-----|
| 基座模型 | Qwen/Qwen3-1.7B-Base |
| LoRA 秩 (r) | 32 |
| LoRA Alpha | 64 |
| 目标模块 | q/k/v/o_proj, gate/up/down_proj |
| 学习率 | 2e-5 (cosine + warmup 10%) |
| 训练轮数 | 2 |
| 有效批量 | 16 (per_device=4 × grad_accum=4) |
| 最大序列长度 | 2048 |
| 量化 | 4-bit NF4 |

### 步骤 3：LLM-as-Judge 评估

使用 **qwen3-max API** 作为评委，在 25 条多类别提示上评估 3 个模型：

| 模型 | 平均分 (1-10) |
|------|:----:|
| Qwen3-1.7B-Base（基座） | **5.40** |
| Lab1-SFT（第1课 LoRA r=32） | **3.68** |
| Lab2-SFT（本课 LoRA r=32） | **3.00** |

评估覆盖 18 个类别：指令跟随、知识问答、数学推理、创意写作、代码生成、中文理解、格式化输出、翻译、总结、分析、建议、逻辑推理、解释概念、实用类、安全性、多步任务、反思类、开放式。

### 步骤 4：数据质量消融实验

固定数据量（3000 条）和超参数，仅改变数据质量级别：

| 条件 | 说明 | 训练损失 | 验证损失 | Judge 均分 |
|------|------|:--------:|:--------:|:----------:|
| raw (无QC) | 原始数据，不做清洗 | 2.9668 | 2.9711 | **2.07** |
| dedup (仅去重) | 仅哈希去重 | 2.8799 | 2.9710 | **2.40** |
| clean (完整QC) | 去重 + 长度过滤 + 格式检查 | 2.9108 | 2.9736 | **2.93** |

**关键发现**：完整数据清洗相比原始数据提升了 **+0.86 分**（+41.5%），验证了数据质量对 SFT 效果的显著影响。

### 步骤 5：报告生成

自动汇总所有实验数据，生成完整 Markdown 报告 → `outputs/lab2_report.md`

## 交付物清单

| # | 交付物 | 位置 |
|---|--------|------|
| 1 | 数据分析报告（来源/规模/分布/清洗/划分） | `outputs/lab2_report.md` 第一章 |
| 2 | LLM-as-Judge 评分对比表（25 条 × 3 模型 + 分类别统计） | `outputs/lab2_report.md` 第二章 |
| 3 | 消融实验结果（数据质量 3 级 + 图表 + 关键发现） | `outputs/lab2_report.md` 第三章 |
| 4 | 书面反思（数据清洗提升/超参数经验/Judge优缺点/改进方向） | `outputs/lab2_report.md` 第四章 |

## 生成的图表

| 图表 | 文件 |
|------|------|
| Token 长度分布 | `outputs/data_distribution.png` |
| 训练/验证损失曲线 | `outputs/lecture2_loss.png` |
| 模型评分对比（柱状图） | `outputs/eval_scores_comparison.png` |
| 分类别评分对比 | `outputs/eval_category_comparison.png` |
| 消融 - 训练损失曲线 | `outputs/ablation_loss_curves.png` |
| 消融 - 平均评分对比 | `outputs/ablation_scores.png` |
| 消融 - 分类别评分对比 | `outputs/ablation_category_scores.png` |

如需重新生成图表（例如更换字体），运行：

```bash
python regenerate_plots.py
```

## 配置说明

所有核心参数集中在 `config.py` 中，主要配置项：

```python
# GPU 选择
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4,5,6,7")

# LLM-as-Judge API（通义千问 DashScope）
DASHSCOPE_API_KEY = "your-api-key"  # 留空则使用本地 Instruct 模型作为 Judge

# 消融实验
ABLATION_SAMPLE_SIZE = 3000   # 每组训练样本数
ABLATION_EPOCHS = 1           # 消融训练轮数
```

## 依赖的外部资源

- **Lab1 SFT 模型**：`../lab1/qwen3-sft-r32`（步骤 3 对比评估使用）
- **Qwen3-1.7B-Base**：HuggingFace Hub 自动下载
- **COIG-CQIA 数据集**：HuggingFace Hub 自动下载
- **qwen3-max API**：评估 Judge，需有效的 DashScope API Key
