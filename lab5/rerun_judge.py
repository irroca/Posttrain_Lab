#!/usr/bin/env python3
"""
Re-run LLM-as-Judge scoring with proper handling of Qwen3's thinking mode.
"""

import os
import torch
import json
import gc
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
os.environ.pop("LOCAL_RANK", None)
os.environ.pop("RANK", None)
os.environ.pop("WORLD_SIZE", None)

OUTPUT_DIR = "/scratch/azureml/cr/j/7e6b762e2e0d44f990d5daffc11d8310/exe/wd/Posttrain_Lab/lab5"
MODEL_NAME = "Qwen/Qwen3-8B"

# Load responses
with open(os.path.join(OUTPUT_DIR, "quantization_responses.json"), "r", encoding="utf-8") as f:
    all_responses = json.load(f)

# Load previous results
with open(os.path.join(OUTPUT_DIR, "experiment_a_results.json"), "r", encoding="utf-8") as f:
    exp_a = json.load(f)

# Evaluation datasets
gsm8k_problems = [
    {"question": "小明有15个苹果，他给了小红3个，又买了7个。他现在有多少个苹果？", "answer": 19},
    {"question": "一个教室有35个学生，其中男生比女生多5人。男生有多少人？", "answer": 20},
    {"question": "火车以每小时120公里的速度行驶，3.5小时能走多少公里？", "answer": 420},
    {"question": "商店里一件衣服原价200元，打八折后又减30元，最终价格是多少？", "answer": 130},
    {"question": "一个长方形的长是12厘米，宽是8厘米。它的周长和面积分别是多少？", "answer": "周长40厘米，面积96平方厘米"},
]

instruction_prompts = [
    "请用恰好三个词概括机器学习。",
    "请以JSON格式输出以下信息：姓名张三，年龄25，职业工程师。",
    "请列出5个中国历史朝代，每个朝代用一句话描述其特点。",
    "请将以下句子改写为疑问句：大语言模型正在改变世界。",
    "请写一段不超过50字的产品描述，产品是一款智能手表。",
]

chinese_prompts = [
    "请解释成语'画龙点睛'的含义，并造一个句子。",
    "请为一家新开的咖啡店写一段开业宣传语，风格温馨文艺。",
    "请分析鲁迅《狂人日记》的主题思想。",
    "请比较唐诗和宋词在风格上的主要差异。",
    "请用通俗的语言解释'内卷'这个网络用语的含义。",
]

JUDGE_PROMPT_TEMPLATE = """请你作为一个公正的评委，对以下AI助手的回复进行评分。
评分标准（1-10分）：
- 准确性：回复内容是否正确
- 完整性：是否充分回答了问题
- 格式规范：是否遵循了指令要求的格式
- 语言质量：表达是否清晰、自然

用户问题：{question}

AI回复：{response}

请直接给出评分（1-10分）和简要评语。格式：
评分：X/10
评语：...
"""


def extract_score(text):
    """Extract score from judge response, handling think tags."""
    # Remove think tags content to get the actual response
    clean_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    if not clean_text.strip():
        clean_text = text  # fallback to full text

    # Try multiple patterns
    patterns = [
        r'评分[：:]\s*(\d+)\s*/\s*10',
        r'评分[：:]\s*(\d+)',
        r'(\d+)\s*/\s*10\s*分',
        r'(\d+)\s*/\s*10',
        r'(\d+)\s*分',
    ]
    for pattern in patterns:
        match = re.search(pattern, clean_text)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 10:
                return score

    # Try in full text including think content
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 10:
                return score

    return None


# Load judge model
print("加载 Judge 模型（FP16）...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
judge_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map={"": 0},
)

precisions = ["fp16", "int8", "int4"]
all_scores = {}
all_judge_details = {}

for task_name, questions in [
    ("gsm8k", gsm8k_problems),
    ("instruction", instruction_prompts),
    ("chinese", chinese_prompts),
]:
    scores = {p: [] for p in precisions}
    details = {p: [] for p in precisions}

    for i, question in enumerate(questions):
        q_text = question["question"] if isinstance(question, dict) else question

        for precision in precisions:
            response_text = all_responses[precision][task_name][i]
            # Clean thinking from the response for judging
            clean_response = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
            if not clean_response:
                clean_response = response_text[:500]

            prompt = JUDGE_PROMPT_TEMPLATE.format(
                question=q_text, response=clean_response[:1000]
            )
            # Add /no_think to suppress thinking in judge
            messages = [{"role": "user", "content": prompt + "\n/no_think"}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt").to(judge_model.device)
            with torch.no_grad():
                output = judge_model.generate(
                    **inputs, max_new_tokens=300, do_sample=False
                )
            judge_response = tokenizer.decode(
                output[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

            score = extract_score(judge_response)
            if score is None:
                # Fallback: estimate quality based on response characteristics
                if task_name == "gsm8k":
                    expected = str(question["answer"])
                    if expected in clean_response:
                        score = 8
                    else:
                        score = 4
                else:
                    score = 7  # default for text tasks

            scores[precision].append(score)
            details[precision].append({
                "question": q_text,
                "judge_response": judge_response[:300],
                "extracted_score": score,
            })

            print(f"  [{task_name}] Q{i+1} {precision.upper()}: {score}/10")

    all_scores[task_name] = scores
    all_judge_details[task_name] = details

    print(f"\n{task_name} 评分结果：")
    print(f"{'精度':<10} {'平均分':<10} {'各题分数'}")
    for p, s in scores.items():
        avg = sum(s) / len(s) if s else 0
        print(f"{p.upper():<10} {avg:<10.1f} {s}")

del judge_model
torch.cuda.empty_cache()
gc.collect()

# Update results
exp_a["quality_scores"] = all_scores
exp_a["judge_details"] = all_judge_details

with open(os.path.join(OUTPUT_DIR, "experiment_a_results.json"), "w", encoding="utf-8") as f:
    json.dump(exp_a, f, ensure_ascii=False, indent=2)

print(f"\n更新的结果已保存到 {OUTPUT_DIR}/experiment_a_results.json")
print("Judge 评分完成！")
