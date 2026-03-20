"""
Step 1: 偏好数据探索 —— 加载 UltraFeedback 数据集并分析
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import config

import random
import json
import numpy as np
from datasets import load_dataset

def main():
    print("=" * 60)
    print("Step 1: 偏好数据探索")
    print("=" * 60)

    # 1. 加载数据集
    print("\n[1/4] 加载数据集...")
    dataset = load_dataset(config.DATASET_NAME, split=config.TRAIN_SPLIT)
    print(f"数据集大小: {len(dataset)}")
    print(f"数据字段: {dataset.column_names}")
    print(f"示例数据结构:")
    print(list(dataset[0].keys()))

    # 2. 检查数据结构
    print("\n[2/4] 检查数据结构...")
    sample = dataset[0]
    print("=" * 60)
    print("Prompt:")
    print(sample["prompt"][:300])
    print("\n" + "=" * 60)

    def extract_text(field):
        if isinstance(field, list):
            return field[-1]["content"]
        return field

    print("Chosen response (前300字):")
    print(extract_text(sample["chosen"])[:300])
    print("\n" + "=" * 60)
    print("Rejected response (前300字):")
    print(extract_text(sample["rejected"])[:300])

    # 3. 质量抽检
    print("\n[3/4] 随机质量抽检 (5 组)...")
    random.seed(config.SEED)
    indices = random.sample(range(len(dataset)), 5)
    quality_check_results = []
    for idx in indices:
        sample = dataset[idx]
        chosen_text = extract_text(sample["chosen"])
        rejected_text = extract_text(sample["rejected"])
        print(f"\n{'=' * 60}")
        print(f"样本 #{idx}")
        print(f"Prompt: {sample['prompt'][:150]}...")
        print(f"\nChosen (前200字): {chosen_text[:200]}...")
        print(f"\nRejected (前200字): {rejected_text[:200]}...")
        print(f"\nChosen 长度: {len(chosen_text.split())} 词")
        print(f"Rejected 长度: {len(rejected_text.split())} 词")
        quality_check_results.append({
            "idx": idx,
            "prompt": sample["prompt"][:200],
            "chosen_len": len(chosen_text.split()),
            "rejected_len": len(rejected_text.split()),
        })

    # 4. 统计分析
    print("\n[4/4] 统计分析...")
    chosen_lengths = []
    rejected_lengths = []
    for sample in dataset:
        chosen_text = extract_text(sample["chosen"])
        rejected_text = extract_text(sample["rejected"])
        chosen_lengths.append(len(chosen_text.split()))
        rejected_lengths.append(len(rejected_text.split()))

    stats = {
        "dataset_size": len(dataset),
        "chosen_mean_len": float(np.mean(chosen_lengths)),
        "chosen_median_len": float(np.median(chosen_lengths)),
        "rejected_mean_len": float(np.mean(rejected_lengths)),
        "rejected_median_len": float(np.median(rejected_lengths)),
        "length_ratio": float(np.mean(chosen_lengths) / np.mean(rejected_lengths)),
    }

    print(f"Chosen 平均长度: {stats['chosen_mean_len']:.0f} 词")
    print(f"Rejected 平均长度: {stats['rejected_mean_len']:.0f} 词")
    print(f"Chosen 中位长度: {stats['chosen_median_len']:.0f} 词")
    print(f"Rejected 中位长度: {stats['rejected_median_len']:.0f} 词")
    print(f"长度比 (Chosen/Rejected): {stats['length_ratio']:.2f}")

    if stats["length_ratio"] > 1.5:
        print("⚠ 警告: Chosen 系统性地比 Rejected 更长，可能存在长度偏差")
        stats["length_bias_warning"] = True
    else:
        stats["length_bias_warning"] = False

    # 保存结果
    out_path = os.path.join(config.OUTPUT_DIR, "step1_data_stats.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n统计数据已保存到: {out_path}")
    print("Step 1 完成!")

if __name__ == "__main__":
    main()
