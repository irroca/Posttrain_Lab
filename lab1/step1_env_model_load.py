"""
步骤 1：环境验证与模型加载
- 确认 GPU 环境
- 以 4-bit 量化加载 Qwen3-1.7B-Base
- 配置 LoRA 适配器
- 运行基座模型基线测试
- 演示 Qwen3 Instruct 的 /think 和 /no_think 模式
"""

import sys
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    MODEL_NAME, INSTRUCT_MODEL_NAME, LORA_R, LORA_ALPHA, LORA_DROPOUT,
    LORA_TARGET_MODULES, EVAL_PROMPTS, MAX_NEW_TOKENS, TEMPERATURE, TOP_P,
    REPETITION_PENALTY,
)


def check_environment():
    """验证 GPU 环境"""
    print("=" * 60)
    print("环境验证")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: 未检测到 GPU，训练将非常缓慢！")


def load_model_and_tokenizer():
    """以 4-bit 量化加载基座模型并配置 LoRA"""
    print("\n" + "=" * 60)
    print("加载模型与 Tokenizer")
    print("=" * 60)

    # 4-bit 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # 加载 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"Model loaded. Parameters: {model.num_parameters() / 1e6:.1f}M")
    print(f"Memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")

    # 配置 LoRA 适配器
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS):
    """基座模型推理（不使用 chat_template）"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    return response


def run_baseline_test(model, tokenizer):
    """基座模型基线测试 —— 使用与 step4 相同的 10 条 EVAL_PROMPTS"""
    print("\n" + "=" * 60)
    print("基座模型输出（微调前）—— 10 条评估提示")
    print("=" * 60)

    baseline_results = []
    for i, prompt in enumerate(EVAL_PROMPTS, 1):
        print(f"\n[{i}/10] [Prompt]: {prompt}")
        response = generate_response(model, tokenizer, prompt)
        print(f"[Response]: {response}")
        print("-" * 40)
        baseline_results.append({"prompt": prompt, "baseline_response": response})

    # 保存基线结果
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(baseline_results, f, ensure_ascii=False, indent=2)
    print("\n基线结果已保存到 outputs/baseline_results.json（共 10 条）")

    return baseline_results


def demo_think_mode():
    """演示 Qwen3 Instruct 的 /think 和 /no_think 模式"""
    print("\n" + "=" * 60)
    print("演示 Qwen3 Instruct /think 和 /no_think 模式")
    print("=" * 60)

    _demo_tok = AutoTokenizer.from_pretrained(INSTRUCT_MODEL_NAME, trust_remote_code=True)
    _demo_model = AutoModelForCausalLM.from_pretrained(
        INSTRUCT_MODEL_NAME,
        device_map="auto",
        dtype="auto",
        trust_remote_code=True,
    )

    # /think 模式
    messages_think = [
        {"role": "user", "content": "/think\n解方程 2x + 5 = 13"}
    ]
    # /no_think 模式
    messages_nothink = [
        {"role": "user", "content": "/no_think\n解方程 2x + 5 = 13"}
    ]

    for label, msgs in [("think", messages_think), ("no_think", messages_nothink)]:
        text = _demo_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = _demo_tok(text, return_tensors="pt").to(_demo_model.device)
        outputs = _demo_model.generate(**inputs, max_new_tokens=512)
        result = _demo_tok.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=False,
        )
        print(f"\n===== /{label} 模式 =====")
        print(result)
        print()

    # 释放显存
    del _demo_model, _demo_tok
    torch.cuda.empty_cache()


def main():
    # 1. 环境验证
    check_environment()

    # 2. 加载模型
    model, tokenizer = load_model_and_tokenizer()

    # 3. 基线测试
    run_baseline_test(model, tokenizer)

    # 4. 演示 think/no_think
    demo_think_mode()

    # 保存模型和 tokenizer 以供后续步骤使用
    print("\n步骤 1 完成！")
    return model, tokenizer


if __name__ == "__main__":
    main()
