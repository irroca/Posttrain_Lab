"""
步骤 4：对比评估
- 加载训练好的 SFT 模型
- 在 10 个手工编写的提示上进行推理
- 对比基座模型（步骤1输出）和微调模型的输出
- 记录评估维度观察
- 保存对比结果
"""

import sys
import os
import re
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    MODEL_NAME, OUTPUT_DIR, LORA_R, LORA_ALPHA, LORA_DROPOUT,
    LORA_TARGET_MODULES, EVAL_PROMPTS, TEST_PROMPTS,
    MAX_NEW_TOKENS, TEMPERATURE, TOP_P,
)


def clean_response(text):
    """清理模型输出中的异常字符"""
    text = re.sub(
        r'[^\u4e00-\u9fff\u0020-\u007ea-zA-Z0-9，。！？：；""''（）、\n\r{}\[\]"\',\.!?:;()\-+=#]',
        '', text
    )
    text = text.strip()
    return text


def load_sft_model():
    """加载训练好的 SFT 模型"""
    print("=" * 60)
    print("加载微调后的模型")
    print("=" * 60)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # 加载 LoRA 权重
    final_dir = os.path.join(OUTPUT_DIR, "final")
    model = PeftModel.from_pretrained(base_model, final_dir)
    model.config.use_cache = True
    model.eval()

    print("SFT 模型加载完成。")
    return model, tokenizer


def generate_response_safe(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS):
    """用 chat_template 格式生成回复"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=1.2,
        )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    return clean_response(response)


def run_sft_evaluation(model, tokenizer):
    """在 10 个提示上运行 SFT 模型评估"""
    print("\n" + "=" * 60)
    print("SFT 模型输出（微调后）")
    print("=" * 60)

    # 先用 test_prompts 做微调后的快速对比
    print("\n--- 与基线相同的测试提示 ---")
    for prompt in TEST_PROMPTS:
        print(f"\n[Prompt]: {prompt}")
        print(f"[Response]: {generate_response_safe(model, tokenizer, prompt)}")
        print("-" * 40)

    # 在完整评估提示集上运行
    print("\n" + "=" * 60)
    print("完整评估集对比")
    print("=" * 60)

    results = []
    for prompt in EVAL_PROMPTS:
        sft_response = generate_response_safe(model, tokenizer, prompt)
        results.append({
            "prompt": prompt,
            "sft_response": sft_response,
        })

    # 展示结果
    for i, r in enumerate(results, 1):
        print(f"\n{'=' * 60}")
        print(f"提示 {i}: {r['prompt']}")
        print(f"{'─' * 60}")
        print(f"[微调模型]: {r['sft_response'][:300]}")
        print(f"{'=' * 60}")

    return results


def save_results(results):
    """保存评估结果"""
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nResults saved to outputs/evaluation_results.json")


def print_evaluation_dimensions():
    """打印评估维度提示"""
    print("\n" + "=" * 60)
    print("评估维度记录模板（人工观察和编写）")
    print("=" * 60)

    evaluation_dimensions = {
        "指令跟随": "微调模型是否正确理解并执行了指令？",
        "格式遵守": "微调模型是否按要求的格式（JSON/表格等）输出？",
        "安全性": "面对有害请求，微调模型是否拒绝回答？",
        "流畅性": "回复是否自然流畅？",
        "基座对比": "相比基座模型，微调模型有哪些明显改进？",
    }

    for dim, question in evaluation_dimensions.items():
        print(f"  {dim}: {question}")


def main():
    # 1. 加载 SFT 模型
    model, tokenizer = load_sft_model()

    # 2. 运行评估
    results = run_sft_evaluation(model, tokenizer)

    # 3. 保存结果
    save_results(results)

    # 4. 打印评估维度
    print_evaluation_dimensions()

    print("\n步骤 4 完成！")
    return results


if __name__ == "__main__":
    main()
