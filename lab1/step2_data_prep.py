"""
步骤 2：数据准备与格式化
- 加载 Alpaca-GPT4-zh 中文数据集
- 转换为 ChatML messages 格式
- 数据清洗（过滤空样本、过短回复、超长样本）
- Token 长度统计
- 划分训练集和验证集，保存到磁盘
"""

import sys
import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    MODEL_NAME, DATASET_NAME, NUM_SAMPLES, TEST_SIZE, SEED,
    MAX_CHAR_LENGTH, MIN_RESPONSE_LENGTH,
)


def load_and_sample_dataset():
    """加载并采样数据集"""
    print("=" * 60)
    print("加载 Alpaca-GPT4-zh 中文数据集")
    print("=" * 60)

    dataset = load_dataset(DATASET_NAME, split="train")
    print(f"Total samples: {len(dataset)}")

    dataset = dataset.shuffle(seed=SEED).select(range(NUM_SAMPLES))
    print(f"Selected samples: {NUM_SAMPLES}")

    # 检查数据格式
    sample = dataset[0]
    print(f"\n[instruction]: {sample['instruction'][:200]}")
    print(f"[output]: {sample['output'][:200]}")

    return dataset


def to_messages(example):
    """将 Alpaca 格式转换为 ChatML messages 格式"""
    user_content = example["instruction"]
    # 如果有 input 字段，拼接到指令后面
    if example.get("input", "").strip():
        user_content += "\n" + example["input"]
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example["output"]},
        ]
    }


def filter_quality(example):
    """数据清洗：过滤低质量样本"""
    msgs = example["messages"]
    user_msg = msgs[0]["content"].strip()
    asst_msg = msgs[1]["content"].strip()

    if not user_msg or not asst_msg:
        return False
    if len(asst_msg) < MIN_RESPONSE_LENGTH:
        return False
    # 粗略估算 token 长度，过滤超长样本
    if len(user_msg) + len(asst_msg) > MAX_CHAR_LENGTH:  # ~2048 tokens 的字符近似
        return False
    return True


def compute_token_stats(dataset, tokenizer, num_samples=500):
    """计算 Token 长度统计"""
    print("\n" + "=" * 60)
    print("Token 长度统计")
    print("=" * 60)

    lengths = []
    for x in dataset.select(range(min(num_samples, len(dataset)))):
        text = tokenizer.apply_chat_template(x["messages"], tokenize=False)
        lengths.append(len(tokenizer.encode(text)))

    print(f"Token 长度统计（前{min(num_samples, len(dataset))}条）：")
    print(f"  平均: {np.mean(lengths):.0f}")
    print(f"  中位数: {np.median(lengths):.0f}")
    print(f"  最大: {np.max(lengths)}")
    print(f"  最小: {np.min(lengths)}")
    print(f"  超过 2048 的比例: {np.mean([l > 2048 for l in lengths]):.2%}")


def main():
    # 1. 加载数据集
    dataset = load_and_sample_dataset()

    # 2. 转换为 messages 格式
    print("\n转换为 messages 格式...")
    dataset = dataset.map(to_messages, num_proc=4)

    # 3. 数据清洗
    print("数据清洗中...")
    dataset = dataset.filter(filter_quality, num_proc=4)
    print(f"清洗后样本数: {len(dataset)}")

    # 4. 验证格式化结果
    print("\n格式化后的样本：")
    sample = dataset[0]
    for msg in sample["messages"]:
        print(f"  [{msg['role']}]: {msg['content'][:100]}...")

    # 5. Token 长度统计
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    compute_token_stats(dataset, tokenizer)

    # 6. 划分训练集和验证集
    split_dataset = dataset.train_test_split(test_size=TEST_SIZE, seed=SEED)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    print(f"\nTrain: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # 7. 保存处理后的数据到磁盘
    os.makedirs("outputs", exist_ok=True)
    train_dataset.save_to_disk("outputs/train_dataset")
    eval_dataset.save_to_disk("outputs/eval_dataset")
    print("数据集已保存到 outputs/train_dataset 和 outputs/eval_dataset")

    print("\n步骤 2 完成！")
    return train_dataset, eval_dataset


if __name__ == "__main__":
    main()
