# Posttrain Lab — 大语言模型后训练实践

> 课程主页：https://posttrain.gaozhijun.me

本仓库包含《大语言模型后训练实践》课程的实验代码和报告。

## 目录结构

```
Posttrain_Lab/
├── lab1/          # 实验1：LoRA 指令微调入门
│   ├── step1_env_model_load.py
│   ├── step2_data_prep.py
│   ├── step3_train.py
│   ├── step4_eval.py
│   ├── step5_hyperparam.py
│   └── outputs/
├── lab2/          # 实验2：构建领域定制 SFT 模型并系统评估
│   ├── step1_data_analysis.py
│   ├── step2_train.py
│   ├── step3_eval.py
│   ├── step4_ablation.py
│   ├── step5_report.py
│   └── outputs/
├── lab3/          # 实验3：DPO 对齐与 SimPO 对比
│   ├── step1_data_explore.py
│   ├── step2_dpo_train.py
│   ├── step3_simpo_train.py
│   ├── step4_evaluate.py
│   ├── step5_beta_ablation.py
│   ├── step6_llm_judge.py
│   ├── step7_report.py
│   └── outputs/
└── README.md
```

## 实验概览

### Lab 1：LoRA 指令微调入门

- **目标**：掌握 LoRA 微调的基本流程
- **基座模型**：Qwen/Qwen3-1.7B-Base
- **数据集**：Alpaca 中文指令数据
- **内容**：环境搭建 → 数据准备 → LoRA 训练 → 评估 → 超参数对比（r=8 vs r=32）
- 详见 [lab1/README.md](lab1/README.md)

### Lab 2：构建领域定制 SFT 模型并系统评估

- **目标**：完整的数据工程 + SFT 训练 + 自动化评估流水线
- **基座模型**：Qwen/Qwen3-1.7B-Base
- **数据集**：COIG-CQIA / zhihu（5631 条中文知乎问答）
- **内容**：
  - 数据质量分析与清洗（去重/长度过滤/格式检查）
  - LoRA SFT 训练（r=32, lr=2e-5, 2 epochs）
  - LLM-as-Judge 多模型评估（qwen3-max API，25 条 × 3 模型）
  - **数据质量消融实验**（raw / dedup / clean，+41.5% 提升）
  - 完整实验报告与书面反思
- 详见 [lab2/README.md](lab2/README.md)

### Lab 3：DPO 对齐与 SimPO 对比

- **目标**：掌握偏好对齐（RLHF 替代方案），对比 DPO 与 SimPO
- **基座模型**：Qwen/Qwen3-1.7B
- **数据集**：HuggingFaceH4/ultrafeedback_binarized（61,135 条偏好对）
- **内容**：
  - DPO 训练（beta=0.1, QLoRA 4-bit, 5000 样本, 1 epoch）
  - SimPO 训练（beta=2.0, gamma=0.5, 自定义实现）
  - 三模型对比评估（SFT / DPO / SimPO：有用性、安全性、多样性）
  - Beta 消融实验（beta=0.05 / 0.1 / 0.5）
  - LLM-as-Judge 自动评分（Qwen3-max API）
- **关键结论**：
  - LLM 评分：SimPO (7.40) > SFT (7.20) > DPO (7.00)
  - SimPO 显存仅 5.30 GB（vs DPO 44.31 GB），无需参考模型
  - 三种模型安全拒绝率一致 (90%)，对齐训练未损害安全性
- 详见 [lab3/outputs/lab3_report.md](lab3/outputs/lab3_report.md)

## 环境配置

```bash
conda create -n posttrain python=3.12 -y
conda activate posttrain
pip install torch transformers trl peft datasets bitsandbytes matplotlib openai
```

## 快速运行

```bash
# Lab 1
cd lab1 && bash run_all.sh

# Lab 2
cd lab2 && bash run_all.sh

# Lab 3
cd lab3 && bash run_all.sh
```

## 预训练权重下载与加载

所有 LoRA adapter 权重已上传至 HuggingFace Hub：  
**https://huggingface.co/leixinlin/posttrain-lab-weights**

### 一键下载全部权重

```python
from huggingface_hub import snapshot_download
snapshot_download("leixinlin/posttrain-lab-weights", local_dir="posttrain-weights")
```

### 各模型加载方式

所有模型的基座均为 `Qwen/Qwen3-1.7B-Base`，通过 PEFT 加载 LoRA adapter：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B-Base", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
```

| 模型 | 说明 | 加载代码 |
|------|------|----------|
| **lab1-sft-r8** | Lab1: LoRA r=8, Alpaca 中文 | `model = PeftModel.from_pretrained(base_model, "leixinlin/posttrain-lab-weights", subfolder="lab1-sft-r8")` |
| **lab1-sft-r32** | Lab1: LoRA r=32, Alpaca 中文 | `model = PeftModel.from_pretrained(base_model, "leixinlin/posttrain-lab-weights", subfolder="lab1-sft-r32")` |
| **lab2-sft-best** | Lab2: LoRA r=32, COIG-CQIA/zhihu, 完整 QC | `model = PeftModel.from_pretrained(base_model, "leixinlin/posttrain-lab-weights", subfolder="lab2-sft-best")` |
| **lab2-ablation-raw** | Lab2 消融: 原始数据（无清洗） | `model = PeftModel.from_pretrained(base_model, "leixinlin/posttrain-lab-weights", subfolder="lab2-ablation-raw")` |
| **lab2-ablation-dedup** | Lab2 消融: 仅去重 | `model = PeftModel.from_pretrained(base_model, "leixinlin/posttrain-lab-weights", subfolder="lab2-ablation-dedup")` |
| **lab2-ablation-clean** | Lab2 消融: 完整 QC（去重+过滤） | `model = PeftModel.from_pretrained(base_model, "leixinlin/posttrain-lab-weights", subfolder="lab2-ablation-clean")` |
| **lab3-dpo** | Lab3: DPO 对齐 (beta=0.1, UltraFeedback) | `model = PeftModel.from_pretrained(base_model, "leixinlin/posttrain-lab-weights", subfolder="lab3-dpo")` |
| **lab3-simpo** | Lab3: SimPO 对齐 (beta=2.0, gamma=0.5) | `model = PeftModel.from_pretrained(base_model, "leixinlin/posttrain-lab-weights", subfolder="lab3-simpo")` |
| **lab3-dpo-beta005** | Lab3 消融: DPO beta=0.05 | `model = PeftModel.from_pretrained(base_model, "leixinlin/posttrain-lab-weights", subfolder="lab3-dpo-beta005")` |
| **lab3-dpo-beta01** | Lab3 消融: DPO beta=0.1 | `model = PeftModel.from_pretrained(base_model, "leixinlin/posttrain-lab-weights", subfolder="lab3-dpo-beta01")` |
| **lab3-dpo-beta05** | Lab3 消融: DPO beta=0.5 | `model = PeftModel.from_pretrained(base_model, "leixinlin/posttrain-lab-weights", subfolder="lab3-dpo-beta05")` |

### 完整加载与推理示例

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. 加载基座 + adapter
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B-Base", torch_dtype="auto", device_map="auto"
)
model = PeftModel.from_pretrained(
    base_model, "leixinlin/posttrain-lab-weights", subfolder="lab2-sft-best"
)
tokenizer = AutoTokenizer.from_pretrained("leixinlin/posttrain-lab-weights", subfolder="lab2-sft-best")

# 2. 推理
messages = [{"role": "user", "content": "解释什么是 Transformer 架构"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## 许可证

本仓库仅用于课程学习目的。
