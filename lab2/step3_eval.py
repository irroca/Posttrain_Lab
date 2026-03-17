"""
步骤3：LLM-as-Judge 评估
- 加载 3 个模型（基座 / Lab1 SFT / Lab2 SFT）
- 在 25 条提示上生成回复
- 使用 LLM-as-Judge 评分
- 生成评分对比表
"""
import sys, os
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))
import config

import re
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def load_base_model():
    """加载基座模型"""
    print("  加载基座模型...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.INSTRUCT_MODEL, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer


def load_sft_model(adapter_path, base_model_name=None):
    """加载 SFT 适配器模型"""
    if base_model_name is None:
        base_model_name = config.BASE_MODEL
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_new_tokens=None, is_base=False):
    """生成模型回复"""
    if max_new_tokens is None:
        max_new_tokens = config.MAX_NEW_TOKENS
    if is_base:
        # 基座模型不懂 chat template，直接用 prompt 做续写
        text = f"问题：{prompt}\n回答："
        max_new_tokens = min(max_new_tokens, 256)
    else:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    # 清理 thinking 标记
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    return response


# ===== LLM-as-Judge =====
JUDGE_TEMPLATE = """请你作为一位公正的评委，评估以下 AI 助手对用户问题的回复质量。
评估维度：
1. 有用性：回复是否解决了用户的问题
2. 准确性：信息是否正确
3. 深度：回复是否有足够的深度和细节
4. 清晰性：表达是否清晰易懂
5. 格式：回复的格式是否合适

请先给出简短的评价说明，然后按以下格式给出 1-10 分的评分：
"[[评分]]"，例如："评分：[[7]]"

[用户问题]
{question}

[助手回复开始]
{answer}
[助手回复结束]"""


def judge_with_api(question, answer):
    """使用 API 进行 LLM-as-Judge 评分"""
    from openai import OpenAI
    client = OpenAI(
        api_key=config.DASHSCOPE_API_KEY,
        base_url=config.JUDGE_API_BASE,
    )
    prompt = JUDGE_TEMPLATE.format(question=question, answer=answer)
    try:
        response = client.chat.completions.create(
            model=config.JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=512,
        )
        output = response.choices[0].message.content
        match = re.search(r"\[\[(\d+)\]\]", output)
        score = int(match.group(1)) if match else None
        return {"output": output, "score": score}
    except Exception as e:
        print(f"  API error: {e}")
        return {"output": str(e), "score": None}


def judge_with_local(question, answer, judge_model, judge_tokenizer):
    """使用本地模型进行 LLM-as-Judge 评分"""
    prompt = JUDGE_TEMPLATE.format(question=question, answer=answer)
    messages = [{"role": "user", "content": prompt}]
    text = judge_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = judge_tokenizer(text, return_tensors="pt").to(judge_model.device)
    with torch.no_grad():
        outputs = judge_model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            pad_token_id=judge_tokenizer.pad_token_id or judge_tokenizer.eos_token_id,
        )
    output = judge_tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    # 清理 thinking 标记
    output = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()
    match = re.search(r"\[\[(\d+)\]\]", output)
    score = int(match.group(1)) if match else None
    return {"output": output, "score": score}


def evaluate_model(model, tokenizer, prompts, judge_fn, model_name, is_base=False):
    """评估单个模型"""
    results = []
    for i, p in enumerate(prompts):
        prompt_text = p["prompt"]
        category = p["category"]
        answer = generate_response(model, tokenizer, prompt_text, is_base=is_base)
        judge_result = judge_fn(prompt_text, answer)
        results.append({
            "prompt": prompt_text,
            "category": category,
            "answer": answer,
            "score": judge_result["score"],
            "judge_output": judge_result["output"],
        })
        print(f"  [{i+1}/{len(prompts)}] {category}: Score={judge_result['score']}")
        if not config.USE_LOCAL_JUDGE:
            time.sleep(0.5)
    return results


def print_comparison_table(all_results):
    """打印评分对比表"""
    model_names = list(all_results.keys())
    header = f"{'提示':<40}"
    for name in model_names:
        header += f"{name:<20}"
    print(f"\n{header}")
    print("=" * (40 + 20 * len(model_names)))

    for i in range(len(all_results[model_names[0]])):
        prompt = all_results[model_names[0]][i]["prompt"][:35] + "..."
        line = f"{prompt:<40}"
        for name in model_names:
            score = all_results[name][i]["score"]
            line += f"{score if score else 'N/A':<20}"
        print(line)

    print("-" * (40 + 20 * len(model_names)))
    avg_line = f"{'平均分':<40}"
    for name in model_names:
        scores = [r["score"] for r in all_results[name] if r["score"] is not None]
        avg = sum(scores) / len(scores) if scores else 0
        avg_line += f"{avg:.2f}{'':<14}"
    print(avg_line)

    # 分类别统计
    print(f"\n{'='*60}")
    print("分类别平均分统计")
    print(f"{'='*60}")
    categories = list(dict.fromkeys(p["category"] for p in config.EVAL_PROMPTS))
    cat_header = f"{'类别':<20}"
    for name in model_names:
        cat_header += f"{name:<20}"
    print(cat_header)
    print("-" * (20 + 20 * len(model_names)))

    category_stats = {}
    for cat in categories:
        line = f"{cat:<20}"
        category_stats[cat] = {}
        for name in model_names:
            cat_scores = [
                r["score"] for r in all_results[name]
                if r["category"] == cat and r["score"] is not None
            ]
            avg = sum(cat_scores) / len(cat_scores) if cat_scores else 0
            category_stats[cat][name] = avg
            line += f"{avg:.2f}{'':<14}"
        print(line)

    return category_stats


def main():
    print("=" * 60)
    print("步骤3：LLM-as-Judge 评估")
    print("=" * 60)

    prompts = config.EVAL_PROMPTS[:25]

    # 配置 judge 函数
    judge_model_obj = None
    judge_tokenizer_obj = None
    if config.USE_LOCAL_JUDGE:
        print("\n[INFO] 未检测到 DASHSCOPE_API_KEY，使用本地 Instruct 模型作为 Judge")
        print("       设置环境变量 DASHSCOPE_API_KEY 可使用 qwen3-max API")
        judge_tokenizer_obj = AutoTokenizer.from_pretrained(
            config.INSTRUCT_MODEL, trust_remote_code=True
        )
        judge_model_obj = AutoModelForCausalLM.from_pretrained(
            config.INSTRUCT_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        def judge_fn(q, a):
            return judge_with_local(q, a, judge_model_obj, judge_tokenizer_obj)
    else:
        print(f"\n[INFO] 使用 {config.JUDGE_MODEL} API 作为 Judge")
        judge_fn = judge_with_api

    all_results = {}

    # ===== 1. 评估基座模型 =====
    print("\n[1/3] 评估基座模型 (Qwen3-1.7B-Base)...")
    base_model, base_tokenizer = load_base_model()
    all_results["Base"] = evaluate_model(
        base_model, base_tokenizer, prompts, judge_fn, "Base", is_base=True
    )
    del base_model
    torch.cuda.empty_cache()

    # ===== 2. 评估 Lab1 SFT 模型 =====
    print("\n[2/3] 评估 Lab1 SFT 模型...")
    lab1_adapter = os.path.abspath(config.LAB1_SFT_ADAPTER)
    if os.path.exists(lab1_adapter):
        lab1_model, lab1_tokenizer = load_sft_model(lab1_adapter)
        all_results["Lab1-SFT"] = evaluate_model(
            lab1_model, lab1_tokenizer, prompts, judge_fn, "Lab1-SFT"
        )
        del lab1_model
        torch.cuda.empty_cache()
    else:
        print(f"  ⚠️ Lab1 SFT 模型不存在: {lab1_adapter}")

    # ===== 3. 评估 Lab2 SFT 模型 =====
    print("\n[3/3] 评估 Lab2 SFT 模型...")
    lab2_adapter = os.path.join(config.OUTPUT_DIR, "best")
    if os.path.exists(lab2_adapter):
        lab2_model, lab2_tokenizer = load_sft_model(lab2_adapter)
        all_results["Lab2-SFT"] = evaluate_model(
            lab2_model, lab2_tokenizer, prompts, judge_fn, "Lab2-SFT"
        )
        del lab2_model
        torch.cuda.empty_cache()
    else:
        print(f"  ⚠️ Lab2 SFT 模型不存在: {lab2_adapter}")

    # 清理 judge 模型
    if judge_model_obj is not None:
        del judge_model_obj
        torch.cuda.empty_cache()

    # ===== 输出对比表 =====
    category_stats = print_comparison_table(all_results)

    # 保存结果
    results_path = os.path.join(config.OUTPUTS_DIR, "eval_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    stats_path = os.path.join(config.OUTPUTS_DIR, "category_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(category_stats, f, ensure_ascii=False, indent=2)

    print(f"\n  评估结果已保存到 {results_path}")
    print(f"  类别统计已保存到 {stats_path}")
    print("\n步骤3 完成！")


if __name__ == "__main__":
    main()
