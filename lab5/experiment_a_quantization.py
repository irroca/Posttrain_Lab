#!/usr/bin/env python3
"""
实验 A：量化实验（必做）
以 FP16、INT8、INT4 三种精度加载 Qwen3-8B，对比显存占用、推理速度和质量。
使用 GPU 4,5,6,7
"""

import os
import torch
import time
import json
import gc
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 设置使用 GPU 4,5,6,7
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
# Disable TP plan to avoid distributed requirement
os.environ.pop("LOCAL_RANK", None)
os.environ.pop("RANK", None)
os.environ.pop("WORLD_SIZE", None)

MODEL_NAME = "Qwen/Qwen3-8B"

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Visible GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB)")

# ===================================================================
# Step 1: 多精度加载对比
# ===================================================================

def load_model(precision="fp16"):
    """以指定精度加载模型，返回模型和元信息"""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    if precision == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map={"":0},
        )
    elif precision == "int8":
        config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=config,
            device_map={"":0},
        )
    elif precision == "int4":
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=config,
            device_map={"":0},
        )
    else:
        raise ValueError(f"不支持的精度: {precision}")

    load_time = time.time() - start_time
    memory_gb = torch.cuda.max_memory_allocated() / 1024**3

    return model, {
        "precision": precision,
        "load_time_s": round(load_time, 1),
        "memory_gb": round(memory_gb, 2),
    }


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 加载对比
print("\n" + "=" * 60)
print("Step 1: 多精度加载对比")
print("=" * 60)

results = {}
for precision in ["fp16", "int8", "int4"]:
    print(f"\n{'=' * 50}")
    print(f"正在加载 {precision.upper()} 模型...")
    model, info = load_model(precision)
    results[precision] = info
    print(f"  加载时间: {info['load_time_s']}s")
    print(f"  显存占用: {info['memory_gb']} GB")

    # 简单验证
    test_input = tokenizer("你好，请自我介绍一下。", return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**test_input, max_new_tokens=50, do_sample=False)
    print(f"  验证回复: {tokenizer.decode(output[0], skip_special_tokens=True)[:100]}")

    del model
    torch.cuda.empty_cache()
    gc.collect()

# 打印对比表
print("\n" + "=" * 60)
print(f"{'精度':<10} {'显存(GB)':<12} {'加载时间(s)':<14} {'相对FP16'}")
print("-" * 60)
fp16_mem = results['fp16']['memory_gb']
for p, info in results.items():
    ratio = info['memory_gb'] / fp16_mem
    print(f"{p.upper():<10} {info['memory_gb']:<12} {info['load_time_s']:<14} {ratio:.2f}x")

# ===================================================================
# Step 2: 推理速度测试
# ===================================================================

print("\n" + "=" * 60)
print("Step 2: 推理速度测试")
print("=" * 60)

TEST_PROMPTS = [
    "请解释什么是量子计算？",
    "用Python写一个快速排序算法。",
    "请将以下中文翻译成英文：人工智能正在深刻改变我们的生活方式。",
    "写一首关于春天的七言绝句。",
    "请解释相对论的核心思想。",
    "什么是知识蒸馏？请用简单的语言解释。",
    "请列出机器学习的五个主要应用领域。",
    "解释什么是梯度下降法。",
    "为什么大语言模型需要后训练？",
    "请比较 Python 和 Java 的主要区别。",
]

speed_results = {}
for precision in ["fp16", "int8", "int4"]:
    print(f"\n测试 {precision.upper()} 推理速度...")
    model, _ = load_model(precision)

    # 预热
    warmup_input = tokenizer("hello", return_tensors="pt").to(model.device)
    with torch.no_grad():
        model.generate(**warmup_input, max_new_tokens=10)

    # 正式测试
    start_total = time.time()
    total_gen_tokens = 0
    first_latencies = []

    for prompt in TEST_PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        # 首 token
        t0 = time.time()
        with torch.no_grad():
            out1 = model.generate(**inputs, max_new_tokens=1, do_sample=False)
        first_latencies.append(time.time() - t0)

        # 完整生成
        with torch.no_grad():
            out_full = model.generate(
                **inputs, max_new_tokens=128, do_sample=False
            )
        total_gen_tokens += out_full.shape[1] - input_len

    total_time = time.time() - start_total
    speed_results[precision] = {
        "tokens_per_sec": round(total_gen_tokens / total_time, 1),
        "avg_first_token_ms": round(
            sum(first_latencies) / len(first_latencies) * 1000, 1
        ),
    }
    print(f"  tokens/s: {speed_results[precision]['tokens_per_sec']}")
    print(f"  首token延迟: {speed_results[precision]['avg_first_token_ms']}ms")

    del model
    torch.cuda.empty_cache()
    gc.collect()

# ===================================================================
# Step 3: 质量评估
# ===================================================================

print("\n" + "=" * 60)
print("Step 3: 质量评估")
print("=" * 60)

# === 评估数据集 ===
# 1. GSM8K 数学推理（5题）
gsm8k_problems = [
    {"question": "小明有15个苹果，他给了小红3个，又买了7个。他现在有多少个苹果？", "answer": 19},
    {"question": "一个教室有35个学生，其中男生比女生多5人。男生有多少人？", "answer": 20},
    {"question": "火车以每小时120公里的速度行驶，3.5小时能走多少公里？", "answer": 420},
    {"question": "商店里一件衣服原价200元，打八折后又减30元，最终价格是多少？", "answer": 130},
    {"question": "一个长方形的长是12厘米，宽是8厘米。它的周长和面积分别是多少？", "answer": "周长40厘米，面积96平方厘米"},
]

# 2. 指令跟随（5条）
instruction_prompts = [
    "请用恰好三个词概括机器学习。",
    "请以JSON格式输出以下信息：姓名张三，年龄25，职业工程师。",
    "请列出5个中国历史朝代，每个朝代用一句话描述其特点。",
    "请将以下句子改写为疑问句：大语言模型正在改变世界。",
    "请写一段不超过50字的产品描述，产品是一款智能手表。",
]

# 3. 中文理解/生成（5条）
chinese_prompts = [
    "请解释成语'画龙点睛'的含义，并造一个句子。",
    "请为一家新开的咖啡店写一段开业宣传语，风格温馨文艺。",
    "请分析鲁迅《狂人日记》的主题思想。",
    "请比较唐诗和宋词在风格上的主要差异。",
    "请用通俗的语言解释'内卷'这个网络用语的含义。",
]


def evaluate_model(model, tokenizer, prompts, max_new_tokens=256):
    """生成模型回复"""
    responses = []
    for prompt in prompts:
        if isinstance(prompt, dict):
            content = prompt["question"]
        else:
            content = prompt
        messages = [{"role": "user", "content": content}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
        response = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        responses.append(response)
    return responses


# 收集所有精度的回复
all_responses = {}
for precision in ["fp16", "int8", "int4"]:
    print(f"\n评估 {precision.upper()} 模型...")
    model, _ = load_model(precision)
    all_responses[precision] = {
        "gsm8k": evaluate_model(model, tokenizer, gsm8k_problems),
        "instruction": evaluate_model(model, tokenizer, instruction_prompts),
        "chinese": evaluate_model(model, tokenizer, chinese_prompts),
    }
    del model
    torch.cuda.empty_cache()
    gc.collect()

# 保存回复
output_dir = "/scratch/azureml/cr/j/7e6b762e2e0d44f990d5daffc11d8310/exe/wd/Posttrain_Lab/lab5"
with open(os.path.join(output_dir, "quantization_responses.json"), "w", encoding="utf-8") as f:
    json.dump(all_responses, f, ensure_ascii=False, indent=2)

# ===================================================================
# Step 4: LLM-as-Judge 评分
# ===================================================================

print("\n" + "=" * 60)
print("Step 4: LLM-as-Judge 评分")
print("=" * 60)

JUDGE_PROMPT_TEMPLATE = """请你作为一个公正的评委，对以下AI助手的回复进行评分。
评分标准（1-10分）：
- 准确性：回复内容是否正确
- 完整性：是否充分回答了问题
- 格式规范：是否遵循了指令要求的格式
- 语言质量：表达是否清晰、自然

用户问题：{question}

AI回复：{response}

请给出评分（1-10分）和简要评语。格式：
评分：X/10
评语：...
"""


def judge_responses(questions, responses_by_precision, judge_model, judge_tokenizer):
    """使用 LLM-as-Judge 评分"""
    scores = {p: [] for p in responses_by_precision}

    for i, question in enumerate(questions):
        q_text = question["question"] if isinstance(question, dict) else question
        for precision, responses in responses_by_precision.items():
            prompt = JUDGE_PROMPT_TEMPLATE.format(
                question=q_text, response=responses[i]
            )
            messages = [{"role": "user", "content": prompt}]
            text = judge_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = judge_tokenizer(text, return_tensors="pt").to(judge_model.device)
            with torch.no_grad():
                output = judge_model.generate(
                    **inputs, max_new_tokens=200, do_sample=False
                )
            judge_response = judge_tokenizer.decode(
                output[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            match = re.search(r'评分[：:]\s*(\d+)', judge_response)
            score = int(match.group(1)) if match else 5
            scores[precision].append(score)

    return scores


# 使用 FP16 模型作为 Judge
print("加载 Judge 模型（FP16）...")
judge_model, _ = load_model("fp16")

all_scores = {}
for task_name, questions in [
    ("gsm8k", gsm8k_problems),
    ("instruction", instruction_prompts),
    ("chinese", chinese_prompts),
]:
    task_responses = {p: all_responses[p][task_name] for p in all_responses}
    scores = judge_responses(questions, task_responses, judge_model, tokenizer)
    all_scores[task_name] = scores
    print(f"\n{task_name} 评分结果：")
    print(f"{'精度':<10} {'平均分':<10} {'各题分数'}")
    for p, s in scores.items():
        avg = sum(s) / len(s) if s else 0
        print(f"{p.upper():<10} {avg:<10.1f} {s}")

del judge_model
torch.cuda.empty_cache()
gc.collect()

# ===================================================================
# Step 5: 汇总结果
# ===================================================================

print("\n" + "=" * 70)
print("量化实验结果汇总")
print("=" * 70)
print(f"\n{'指标':<20} {'FP16':<15} {'INT8':<15} {'INT4':<15}")
print("-" * 65)
print(f"{'显存 (GB)':<20} {results['fp16']['memory_gb']:<15} "
      f"{results['int8']['memory_gb']:<15} {results['int4']['memory_gb']:<15}")
print(f"{'加载时间 (s)':<20} {results['fp16']['load_time_s']:<15} "
      f"{results['int8']['load_time_s']:<15} {results['int4']['load_time_s']:<15}")
print(f"{'tokens/s':<20} {speed_results['fp16']['tokens_per_sec']:<15} "
      f"{speed_results['int8']['tokens_per_sec']:<15} "
      f"{speed_results['int4']['tokens_per_sec']:<15}")
print(f"{'首token延迟 (ms)':<20} {speed_results['fp16']['avg_first_token_ms']:<15} "
      f"{speed_results['int8']['avg_first_token_ms']:<15} "
      f"{speed_results['int4']['avg_first_token_ms']:<15}")

# 保存所有结果
all_results = {
    "memory_loading": results,
    "speed": speed_results,
    "quality_scores": all_scores,
    "responses": all_responses,
}

with open(os.path.join(output_dir, "experiment_a_results.json"), "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n结果已保存到 {output_dir}/experiment_a_results.json")
print("实验 A 完成！")
