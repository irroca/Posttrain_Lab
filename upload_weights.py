#!/usr/bin/env python3
"""
将 LoRA adapter 权重上传到 HuggingFace Hub（方便日后复用）

使用方法:
  1. 先登录: huggingface-cli login
  2. 修改下方 HF_USERNAME 为你的用户名
  3. 运行: python upload_weights.py
"""
import os
from huggingface_hub import HfApi

# ========== 修改为你的 HuggingFace 用户名 ==========
HF_USERNAME = "leixinlin"
# ==================================================

REPO_NAME = f"{HF_USERNAME}/posttrain-lab-weights"

ADAPTERS = {
    # name -> (local_path, description)
    "lab1-sft-r8": (
        "lab1/qwen3-sft-r8",
        "Lab1: LoRA r=8 SFT on Alpaca-zh (Qwen3-1.7B-Base)",
    ),
    "lab1-sft-r32": (
        "lab1/qwen3-sft-r32",
        "Lab1: LoRA r=32 SFT on Alpaca-zh (Qwen3-1.7B-Base)",
    ),
    "lab2-sft-best": (
        "lab2/lecture2-sft/best",
        "Lab2: LoRA r=32 SFT on COIG-CQIA/zhihu (Qwen3-1.7B-Base)",
    ),
    "lab2-ablation-raw": (
        "lab2/lecture2-sft/ablation_raw/best",
        "Lab2 ablation: raw data (no QC)",
    ),
    "lab2-ablation-dedup": (
        "lab2/lecture2-sft/ablation_dedup/best",
        "Lab2 ablation: dedup only",
    ),
    "lab2-ablation-clean": (
        "lab2/lecture2-sft/ablation_clean/best",
        "Lab2 ablation: full QC (dedup + filter)",
    ),
    # ---- Lab 3: DPO & SimPO ----
    "lab3-dpo": (
        "lab3/dpo-qwen3-1.7b/final",
        "Lab3: DPO aligned on UltraFeedback (Qwen3-1.7B, beta=0.1)",
    ),
    "lab3-simpo": (
        "lab3/simpo-qwen3-1.7b/final",
        "Lab3: SimPO aligned on UltraFeedback (Qwen3-1.7B, beta=2.0, gamma=0.5)",
    ),
    "lab3-dpo-beta005": (
        "lab3/dpo-beta-0.05/checkpoint-250",
        "Lab3 ablation: DPO beta=0.05",
    ),
    "lab3-dpo-beta01": (
        "lab3/dpo-beta-0.1/checkpoint-250",
        "Lab3 ablation: DPO beta=0.1",
    ),
    "lab3-dpo-beta05": (
        "lab3/dpo-beta-0.5/checkpoint-250",
        "Lab3 ablation: DPO beta=0.5",
    ),
    # ---- Lab 4: GRPO RL ----
    "lab4-grpo-math": (
        "lab4/outputs/grpo-qwen3-1.7b-math/final",
        "Lab4: GRPO math reasoning on GSM8K (Qwen3-1.7B-Base, 500 steps, full weights)",
    ),
    "lab4-grpo-format": (
        "lab4/outputs/grpo-format-reward/final",
        "Lab4: GRPO with format reward (Qwen3-1.7B-Base, 100 steps, full weights)",
    ),
}


def main():
    if HF_USERNAME == "YOUR_HF_USERNAME":
        print("请先修改 upload_weights.py 中的 HF_USERNAME 为你的 HuggingFace 用户名")
        return

    api = HfApi()

    # 创建仓库（如果不存在）
    api.create_repo(REPO_NAME, exist_ok=True, repo_type="model")
    print(f"仓库: https://huggingface.co/{REPO_NAME}")

    # 上传 README
    readme = f"# Posttrain Lab Weights\n\n"
    readme += "《大语言模型后训练实践》课程实验的 LoRA adapter 权重。\n\n"
    readme += "| 目录 | 说明 |\n|------|------|\n"
    for name, (_, desc) in ADAPTERS.items():
        readme += f"| `{name}/` | {desc} |\n"
    readme += "\n## 使用方法\n\n```python\nfrom peft import PeftModel\n"
    readme += "from transformers import AutoModelForCausalLM\n\n"
    readme += f'model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B-Base")\n'
    readme += f'model = PeftModel.from_pretrained(model, "{REPO_NAME}", subfolder="lab2-sft-best")\n```\n'

    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=REPO_NAME,
    )

    # 逐个上传 adapter
    for name, (local_path, desc) in ADAPTERS.items():
        abs_path = os.path.join(os.path.dirname(__file__), local_path)
        if not os.path.exists(abs_path):
            print(f"  跳过 {name}: {abs_path} 不存在")
            continue
        print(f"  上传 {name} ({local_path})...")
        api.upload_folder(
            folder_path=abs_path,
            path_in_repo=name,
            repo_id=REPO_NAME,
        )
        print(f"  ✓ {name} 上传完成")

    print(f"\n全部完成！访问: https://huggingface.co/{REPO_NAME}")
    print(f"\n日后加载示例:")
    print(f'  from huggingface_hub import snapshot_download')
    print(f'  snapshot_download("{REPO_NAME}", local_dir="posttrain-weights")')


if __name__ == "__main__":
    main()
