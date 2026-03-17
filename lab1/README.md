# Lab1: 微调 Qwen3-1.7B 为指令跟随助手

使用 **QLoRA** 和 **SFTTrainer** 将 Qwen3-1.7B 基座模型微调为能遵循指令的中文对话助手。

## 实验概述

| 项目       | 说明                                                        |
| ---------- | ----------------------------------------------------------- |
| 基座模型   | `Qwen/Qwen3-1.7B-Base`                                     |
| 数据集     | `llamafactory/alpaca_gpt4_zh`（中文 Alpaca-GPT4）           |
| 方法       | QLoRA（4-bit NF4，rank=32，alpha=64）                       |
| 框架       | Hugging Face TRL（SFTTrainer）、PEFT、Transformers、bitsandbytes |
| GPU 要求   | 推荐 A100-40G 或以上；RTX 4090（24GB）也可运行              |

## 目录结构

```
lab1/
├── README.md                 # 本文档
├── requirements.txt          # Python 依赖
├── config.py                 # 共享配置（模型、LoRA、训练超参数）
├── step1_env_model_load.py   # 步骤1: 环境验证与模型加载
├── step2_data_prep.py        # 步骤2: 数据准备与格式化
├── step3_train.py            # 步骤3: SFT 训练
├── step4_eval.py             # 步骤4: 对比评估
├── step5_hyperparam.py       # 步骤5: 超参数探索
├── run_all.sh                # 一键运行全部步骤
└── outputs/                  # (运行后生成) 所有输出文件
    ├── baseline_results.json
    ├── training_loss.png
    ├── evaluation_results.json
    ├── hyperparam_results.json
    └── hyperparam_comparison_*.png
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> **注意**: 需要先确认 GPU 可用（`nvidia-smi`），推荐 A100-40G。

### 2. 一键运行全部步骤

```bash
cd lab1
./run_all.sh
```

支持的选项：

```bash
./run_all.sh                # 运行全部步骤（超参数实验默认方案A: 学习率对比）
./run_all.sh --plan b       # 使用方案B: LoRA秩对比
./run_all.sh --skip-step5   # 跳过超参数探索步骤（节省时间）
```

### 3. 分步运行

如果希望逐步运行并检查每步结果，可以按顺序执行：

```bash
cd lab1
python step1_env_model_load.py   # ~15 分钟
python step2_data_prep.py        # ~15 分钟
python step3_train.py            # ~30 分钟 (A100)
python step4_eval.py             # ~20 分钟
python step5_hyperparam.py --plan a   # ~20 分钟（方案A: 学习率对比）
# 或
python step5_hyperparam.py --plan b   # ~20 分钟（方案B: LoRA秩对比）
```

## 各步骤详解

### 步骤 1：环境验证与模型加载（~15 分钟）

- 检测 GPU 环境（PyTorch / CUDA / 显存）
- 以 4-bit NF4 量化加载 `Qwen3-1.7B-Base` 基座模型
- 配置 LoRA 适配器（r=32, alpha=64, 目标模块: q/k/v/o/gate/up/down_proj）
- 在 3 个测试提示上运行基线推理，保存到 `outputs/baseline_results.json`
- 演示 Qwen3 Instruct 版本的 `/think` 和 `/no_think` 推理模式

**预期输出**: `trainable params: 34,865,152 || all params: 1,755,440,128 || trainable%: 1.9861`

### 步骤 2：数据准备与格式化（~15 分钟）

- 从 HuggingFace 加载 `llamafactory/alpaca_gpt4_zh` 数据集
- 随机采样 10,000 条样本
- 将 Alpaca 格式转换为 ChatML `messages` 格式
- 数据清洗：过滤空样本、回复过短（<30字符）、过长（>6000字符）的样本
- Token 长度统计分析
- 按 95%/5% 划分训练集/验证集，保存到 `outputs/` 目录

### 步骤 3：SFT 训练（A100 ~30 分钟）

- 重新加载量化模型 + LoRA
- 加载步骤 2 保存的数据集
- 使用 `SFTTrainer` 训练，关键超参数：
  - `batch_size=4`, `gradient_accumulation=4`（有效 batch_size=16）
  - `learning_rate=2e-4`, `cosine` 调度器
  - `bf16=True`, `gradient_checkpointing=True`
  - `max_length=512`, `1 epoch`
- 保存模型到 `qwen3-1.7b-sft/final/`
- 绘制训练损失曲线到 `outputs/training_loss.png`

### 步骤 4：对比评估（~20 分钟）

- 加载训练好的 LoRA 模型
- 在 10 个手工编写的评估提示上生成回复，涵盖：
  - 简单指令跟随、格式化输出（JSON/表格）
  - 创意写作、推理能力
  - 安全性（有害请求拒绝）、多语言翻译
- 保存全部对比结果到 `outputs/evaluation_results.json`
- 评估维度：指令跟随、格式遵守、安全性、流畅性、基座对比

### 步骤 5：超参数探索（~20 分钟）

提供两种实验方案：

**方案 A（默认）—— 学习率对比**：
- 基线: `lr=2e-4` vs 实验: `lr=5e-5`
- 对比训练损失曲线和输出质量

**方案 B —— LoRA 秩对比**：
- 基线: `r=32, alpha=64` vs 实验: `r=8, alpha=16`
- 对比可训练参数量、训练损失和输出质量

## 修改配置

所有超参数集中在 `config.py` 中，修改后所有步骤自动生效：

```python
# 修改模型
MODEL_NAME = "Qwen/Qwen3-0.6B-Base"    # 降级为更小的模型（T4适用）

# 修改 LoRA
LORA_R = 16
LORA_ALPHA = 32

# 修改训练
LEARNING_RATE = 1e-4
NUM_SAMPLES = 5000   # 减少数据量以加速
```

## 交付物清单

完成实验后需提交：

- [ ] **训练损失曲线** —— `outputs/training_loss.png`
- [ ] **10 条提示的对比表** —— 基座模型 vs. 微调模型的输出对比（`outputs/evaluation_results.json` + `outputs/baseline_results.json`）
- [ ] **超参数实验结果** —— 修改了哪个参数、训练曲线对比、输出质量变化（`outputs/hyperparam_*.png` + `outputs/hyperparam_results.json`）
- [ ] **1 页书面分析**，讨论：
  - 数据质量与数量的关系（联系 LIMA 论文）
  - LoRA 超参数（秩、alpha）对训练效果的影响
  - 基座模型 vs. 微调模型的主要差异在哪些方面最明显
  - （可选）与 Qwen3-1.7B Instruct 版本的思考模式对比观察

## 常见问题

**Q: GPU 显存不足怎么办？**
- 减小 `PER_DEVICE_TRAIN_BATCH_SIZE`（如 2 或 1）
- 减小 `MAX_SEQ_LENGTH`（如 256）
- 换用更小的模型 `Qwen/Qwen3-0.6B-Base`

**Q: 数据集下载失败？**
- 确保网络可以访问 HuggingFace Hub
- 可设置镜像: `export HF_ENDPOINT=https://hf-mirror.com`

**Q: 训练中断如何恢复？**
- 步骤 3 设置了 checkpoint 保存（每 200 步），可以从 checkpoint 恢复
- 步骤之间通过文件传递数据，不需要从头开始

## 参考资料

- [实验原始页面](https://posttrain.gaozhijun.me/docs/lecture-1/lab/)
- [Qwen3 模型系列](https://huggingface.co/Qwen)
- [TRL 文档](https://huggingface.co/docs/trl)
- [PEFT / LoRA 文档](https://huggingface.co/docs/peft)
