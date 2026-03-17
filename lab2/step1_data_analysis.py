"""
步骤1：数据集分析与预处理
- 加载 COIG-CQIA 数据集
- Token 长度分布分析
- 质量控制（去重、过滤）
- 训练/验证/测试集划分
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import config

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hashlib import md5
from datasets import load_dataset
from transformers import AutoTokenizer

plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    print("=" * 60)
    print("步骤1：数据集分析与预处理")
    print("=" * 60)

    # ===== 1. 加载数据集 =====
    print("\n[1/5] 加载数据集...")
    dataset = load_dataset(config.DATASET_NAME, config.DATASET_SUBSET, split="train")
    print(f"  数据集来源: {config.DATASET_NAME} / {config.DATASET_SUBSET}")
    print(f"  原始样本数: {len(dataset)}")
    print(f"  列信息: {dataset.column_names}")
    print(f"  样本示例: {dataset[0]}")

    # 转换为 messages 格式
    def convert_to_messages(example):
        messages = []
        user_content = example["instruction"]
        if example.get("input") and example["input"].strip():
            user_content += "\n" + example["input"]
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": example["output"]})
        return {"messages": messages}

    dataset = dataset.map(convert_to_messages)
    total_raw = len(dataset)

    # ===== 2. Token 长度分析 =====
    print("\n[2/5] 分析 Token 长度分布...")
    tokenizer = AutoTokenizer.from_pretrained(config.INSTRUCT_MODEL, trust_remote_code=True)

    def get_token_length(example):
        messages = example["messages"]
        user_tokens = sum(
            len(tokenizer.encode(m["content"]))
            for m in messages if m["role"] == "user"
        )
        assistant_tokens = sum(
            len(tokenizer.encode(m["content"]))
            for m in messages if m["role"] == "assistant"
        )
        return {"user_tokens": user_tokens, "assistant_tokens": assistant_tokens}

    dataset = dataset.map(get_token_length, num_proc=4)

    user_tok = np.array(dataset["user_tokens"])
    asst_tok = np.array(dataset["assistant_tokens"])

    stats = {
        "user_tokens": {
            "mean": float(np.mean(user_tok)),
            "median": float(np.median(user_tok)),
            "p5": float(np.percentile(user_tok, 5)),
            "p95": float(np.percentile(user_tok, 95)),
            "min": int(np.min(user_tok)),
            "max": int(np.max(user_tok)),
        },
        "assistant_tokens": {
            "mean": float(np.mean(asst_tok)),
            "median": float(np.median(asst_tok)),
            "p5": float(np.percentile(asst_tok, 5)),
            "p95": float(np.percentile(asst_tok, 95)),
            "min": int(np.min(asst_tok)),
            "max": int(np.max(asst_tok)),
        },
    }

    print(f"  用户输入: 均值={stats['user_tokens']['mean']:.0f}, "
          f"中位数={stats['user_tokens']['median']:.0f}, "
          f"P95={stats['user_tokens']['p95']:.0f}")
    print(f"  助手回复: 均值={stats['assistant_tokens']['mean']:.0f}, "
          f"中位数={stats['assistant_tokens']['median']:.0f}, "
          f"P95={stats['assistant_tokens']['p95']:.0f}")

    # 绘制长度分布图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(user_tok, bins=50, alpha=0.7, color="steelblue")
    axes[0].set_title("User Input Token Length Distribution")
    axes[0].set_xlabel("Tokens")
    axes[0].set_ylabel("Count")
    axes[0].axvline(np.median(user_tok), color="red", linestyle="--",
                    label=f"Median: {np.median(user_tok):.0f}")
    axes[0].legend()

    axes[1].hist(asst_tok, bins=50, alpha=0.7, color="coral")
    axes[1].set_title("Assistant Response Token Length Distribution")
    axes[1].set_xlabel("Tokens")
    axes[1].set_ylabel("Count")
    axes[1].axvline(np.median(asst_tok), color="red", linestyle="--",
                    label=f"Median: {np.median(asst_tok):.0f}")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUTS_DIR, "data_distribution.png"), dpi=150)
    plt.close()
    print(f"  长度分布图已保存到 outputs/data_distribution.png")

    # ===== 3. 质量控制 =====
    print("\n[3/5] 数据质量控制...")

    # 3.1 去重
    def dedup_dataset(ds):
        seen_hashes = set()
        keep_indices = []
        for i, example in enumerate(ds):
            user_content = " ".join(
                m["content"] for m in example["messages"] if m["role"] == "user"
            )
            content_hash = md5(user_content.encode()).hexdigest()
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                keep_indices.append(i)
        return ds.select(keep_indices)

    dataset_deduped = dedup_dataset(dataset)
    dedup_removed = len(dataset) - len(dataset_deduped)
    print(f"  去重前: {len(dataset)}, 去重后: {len(dataset_deduped)}, 去除: {dedup_removed}")

    # 3.2 过滤过短/过长样本
    def filter_length(example):
        total = example["user_tokens"] + example["assistant_tokens"]
        return (example["assistant_tokens"] > config.MIN_ASSISTANT_TOKENS
                and total < config.MAX_TOTAL_TOKENS)

    dataset_filtered = dataset_deduped.filter(filter_length)
    length_removed = len(dataset_deduped) - len(dataset_filtered)
    print(f"  长度过滤后: {len(dataset_filtered)}, 去除: {length_removed}")

    # 3.3 格式一致性检查
    def check_format(example):
        roles = [m["role"] for m in example["messages"]]
        return "user" in roles and "assistant" in roles

    dataset_clean = dataset_filtered.filter(check_format)
    format_removed = len(dataset_filtered) - len(dataset_clean)
    print(f"  格式过滤后: {len(dataset_clean)}, 去除: {format_removed}")

    qc_summary = {
        "原始数据量": total_raw,
        "去重后": len(dataset_deduped),
        "去重去除": dedup_removed,
        "长度过滤后": len(dataset_filtered),
        "长度过滤去除": length_removed,
        "格式过滤后": len(dataset_clean),
        "格式过滤去除": format_removed,
        "最终数据量": len(dataset_clean),
        "总去除比例": f"{(1 - len(dataset_clean)/total_raw)*100:.1f}%",
    }

    # ===== 4. 划分数据集 =====
    print("\n[4/5] 划分数据集 (8:1:1)...")
    split1 = dataset_clean.train_test_split(test_size=0.2, seed=config.SEED)
    split2 = split1["test"].train_test_split(test_size=0.5, seed=config.SEED)
    train_dataset = split1["train"]
    val_dataset = split2["train"]
    test_dataset = split2["test"]
    print(f"  训练集: {len(train_dataset)}")
    print(f"  验证集: {len(val_dataset)}")
    print(f"  测试集: {len(test_dataset)}")

    # ===== 5. ChatML 格式化 =====
    print("\n[5/5] ChatML 格式化...")
    def format_chat(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    train_dataset = train_dataset.map(format_chat, num_proc=4)
    val_dataset = val_dataset.map(format_chat, num_proc=4)
    test_dataset = test_dataset.map(format_chat, num_proc=4)
    print(f"  格式化样本示例:\n{train_dataset[0]['text'][:300]}...")

    # 保存中间数据集（供后续步骤使用）
    data_dir = os.path.join(config.OUTPUTS_DIR, "datasets")
    os.makedirs(data_dir, exist_ok=True)
    train_dataset.save_to_disk(os.path.join(data_dir, "train"))
    val_dataset.save_to_disk(os.path.join(data_dir, "val"))
    test_dataset.save_to_disk(os.path.join(data_dir, "test"))

    # 同时保存各阶段数据集（供消融实验使用）
    # 保存 raw 和 dedup-only 版本
    def format_chat_ds(ds):
        return ds.map(format_chat, num_proc=4)

    raw_split = dataset.train_test_split(test_size=0.2, seed=config.SEED)
    raw_train = format_chat_ds(raw_split["train"])
    raw_train.save_to_disk(os.path.join(data_dir, "train_raw"))

    dedup_split = dataset_deduped.train_test_split(test_size=0.2, seed=config.SEED)
    dedup_train = format_chat_ds(dedup_split["train"])
    dedup_train.save_to_disk(os.path.join(data_dir, "train_dedup"))

    print(f"\n  数据集已保存到 {data_dir}/")
    print(f"    train_raw (无QC): {len(raw_train)}")
    print(f"    train_dedup (仅去重): {len(dedup_train)}")
    print(f"    train (完整QC): {len(train_dataset)}")

    # 保存分析报告数据
    report_data = {
        "dataset_info": {
            "source": f"{config.DATASET_NAME} / {config.DATASET_SUBSET}",
            "total_raw": total_raw,
            "columns": dataset.column_names,
        },
        "token_stats": stats,
        "quality_control": qc_summary,
        "split_info": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
        "ablation_data_sizes": {
            "raw": len(raw_train),
            "dedup": len(dedup_train),
            "clean": len(train_dataset),
        },
    }
    report_path = os.path.join(config.OUTPUTS_DIR, "data_analysis.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    print(f"\n  分析报告已保存到 {report_path}")
    print("\n步骤1 完成！")


if __name__ == "__main__":
    main()
