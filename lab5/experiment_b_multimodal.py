#!/usr/bin/env python3
"""
实验 B 选项 2：多模态实验（增强版）
使用 Qwen2.5-VL-7B-Instruct 进行全面的视觉语言能力评估。
涵盖：基础VQA、空间推理、OCR、图表理解、多轮对话、视觉推理、幻觉检测、推理速度
使用 GPU 4,5,6,7
"""

import os
import sys
import torch
import json
import time
import gc
import re
import math
from io import BytesIO

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
os.environ.pop("LOCAL_RANK", None)
os.environ.pop("RANK", None)
os.environ.pop("WORLD_SIZE", None)

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image, ImageDraw, ImageFont

VLM_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
OUTPUT_DIR = "/scratch/azureml/cr/j/7e6b762e2e0d44f990d5daffc11d8310/exe/wd/Posttrain_Lab/lab5"

print("=" * 70)
print("实验 B 选项 2：多模态实验（增强版）")
print("=" * 70)

# ===================================================================
# Step 1: 加载 VLM 模型
# ===================================================================
print("\n[Step 1] 加载 Qwen2.5-VL-7B-Instruct 模型...")
torch.cuda.empty_cache()
gc.collect()
torch.cuda.reset_peak_memory_stats()

start_time = time.time()
vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    VLM_MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
)
processor = AutoProcessor.from_pretrained(VLM_MODEL_NAME)

load_time = time.time() - start_time
memory_gb = torch.cuda.max_memory_allocated() / 1024**3
param_count = sum(p.numel() for p in vlm_model.parameters()) / 1e9

print(f"  模型参数量: {param_count:.2f}B")
print(f"  加载时间: {load_time:.1f}s")
print(f"  显存占用: {memory_gb:.2f} GB")


# ===================================================================
# Helper: VLM inference
# ===================================================================
def vlm_infer(model, processor, image, question, max_new_tokens=512):
    """Single-turn VLM inference, returns (response, latency_ms)."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    latency = (time.time() - t0) * 1000
    trimmed = output[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return response, latency


def vlm_multiturn(model, processor, image, conversation, max_new_tokens=512):
    """Multi-turn VLM inference. conversation is list of user strings.
    Returns list of (response, latency_ms)."""
    results = []
    messages = []
    for i, user_msg in enumerate(conversation):
        if i == 0:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_msg},
                ],
            })
        else:
            messages.append({"role": "user", "content": user_msg})

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)
        t0 = time.time()
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        latency = (time.time() - t0) * 1000
        trimmed = output[:, inputs.input_ids.shape[1]:]
        response = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        messages.append({"role": "assistant", "content": response})
        results.append((response, latency))
    return results


# ===================================================================
# Helper: Generate rich test images
# ===================================================================
def get_font(size=16):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def create_all_test_images():
    """Create a diverse set of test images for comprehensive evaluation."""
    images = {}

    # ---------- 1. Geometric shapes (basic) ----------
    img = Image.new('RGB', (500, 400), 'white')
    d = ImageDraw.Draw(img)
    d.rectangle([30, 50, 150, 170], fill='red', outline='black', width=2)
    d.ellipse([180, 50, 330, 200], fill='blue', outline='black', width=2)
    d.polygon([(400, 50), (340, 200), (460, 200)], fill='green', outline='black')
    # Row 2
    d.rectangle([30, 250, 120, 370], fill='orange', outline='black', width=2)
    d.ellipse([160, 250, 280, 370], fill='purple', outline='black', width=2)
    d.polygon([(370, 370), (310, 250), (430, 250)], fill='yellow', outline='black')
    font = get_font(14)
    d.text((70, 175), "Red Rect", fill='black', font=font)
    d.text((220, 205), "Blue Circle", fill='black', font=font)
    d.text((370, 205), "Green Tri", fill='black', font=font)
    images["shapes"] = img

    # ---------- 2. Math equation (OCR) ----------
    img = Image.new('RGB', (500, 250), 'lightyellow')
    d = ImageDraw.Draw(img)
    big = get_font(42)
    d.text((30, 20), "7 x 8 + 6 = ?", fill='black', font=big)
    d.text((30, 90), "144 / 12 = ?", fill='black', font=big)
    d.text((30, 160), "15% of 200 = ?", fill='black', font=big)
    images["math"] = img

    # ---------- 3. Bar chart ----------
    img = Image.new('RGB', (500, 400), 'white')
    d = ImageDraw.Draw(img)
    title_font = get_font(18)
    d.text((120, 10), "Monthly Sales (units)", fill='black', font=title_font)
    data = [("Jan", 45), ("Feb", 72), ("Mar", 58), ("Apr", 91), ("May", 65), ("Jun", 83)]
    colors = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA', '#00ACC1']
    max_val = max(v for _, v in data)
    bar_w = 55
    for i, ((lbl, val), col) in enumerate(zip(data, colors)):
        x = 40 + i * 75
        h = int(val / max_val * 280)
        d.rectangle([x, 350 - h, x + bar_w, 350], fill=col, outline='black')
        small = get_font(12)
        d.text((x + 15, 355), lbl, fill='black', font=small)
        d.text((x + 15, 350 - h - 18), str(val), fill='black', font=small)
    images["barchart"] = img

    # ---------- 4. Scene (house, tree, sun, river, mountains) ----------
    img = Image.new('RGB', (600, 450), '#87CEEB')
    d = ImageDraw.Draw(img)
    # Mountains
    d.polygon([(0, 280), (150, 120), (300, 280)], fill='#696969')
    d.polygon([(200, 280), (380, 80), (560, 280)], fill='#808080')
    # Ground
    d.rectangle([0, 280, 600, 450], fill='#228B22')
    # River
    d.polygon([(250, 450), (280, 340), (320, 340), (350, 450)], fill='#4169E1')
    # Sun
    d.ellipse([480, 20, 560, 100], fill='yellow', outline='orange', width=2)
    # House
    d.rectangle([60, 300, 180, 400], fill='#8B4513', outline='black', width=2)
    d.polygon([(45, 300), (120, 230), (195, 300)], fill='#B22222', outline='black')
    d.rectangle([105, 350, 135, 400], fill='#D2691E', outline='black')
    d.rectangle([70, 320, 100, 345], fill='#ADD8E6', outline='black')
    # Tree
    d.rectangle([430, 310, 445, 400], fill='#8B4513')
    d.ellipse([395, 240, 480, 330], fill='#006400')
    # Clouds
    for cx, cy in [(100, 50), (300, 40)]:
        d.ellipse([cx, cy, cx+80, cy+40], fill='white')
        d.ellipse([cx+30, cy-15, cx+90, cy+30], fill='white')
        d.ellipse([cx+60, cy, cx+120, cy+40], fill='white')
    images["scene"] = img

    # ---------- 5. Clock (7:25) ----------
    img = Image.new('RGB', (350, 350), 'white')
    d = ImageDraw.Draw(img)
    cx, cy, r = 175, 175, 140
    d.ellipse([cx-r, cy-r, cx+r, cy+r], outline='black', width=3)
    clock_font = get_font(22)
    for num in range(1, 13):
        angle = math.radians(num * 30 - 90)
        nx = cx + int((r - 25) * math.cos(angle))
        ny = cy + int((r - 25) * math.sin(angle))
        d.text((nx - 8, ny - 10), str(num), fill='black', font=clock_font)
    # Hour hand: 7:25 -> hour angle = (7 + 25/60) * 30 - 90 degrees
    ha = math.radians((7 + 25 / 60) * 30 - 90)
    d.line([(cx, cy), (cx + int(80 * math.cos(ha)), cy + int(80 * math.sin(ha)))], fill='black', width=5)
    # Minute hand: 25 * 6 - 90 degrees
    ma = math.radians(25 * 6 - 90)
    d.line([(cx, cy), (cx + int(120 * math.cos(ma)), cy + int(120 * math.sin(ma)))], fill='black', width=3)
    d.ellipse([cx-5, cy-5, cx+5, cy+5], fill='black')
    images["clock"] = img

    # ---------- 6. Spatial layout ----------
    img = Image.new('RGB', (500, 500), '#F5F5DC')
    d = ImageDraw.Draw(img)
    d.rectangle([10, 10, 490, 490], outline='black', width=2)
    # Top-left: red star
    star_pts = []
    for i in range(5):
        outer = math.radians(i * 72 - 90)
        inner = math.radians(i * 72 + 36 - 90)
        star_pts.append((80 + int(40 * math.cos(outer)), 80 + int(40 * math.sin(outer))))
        star_pts.append((80 + int(18 * math.cos(inner)), 80 + int(18 * math.sin(inner))))
    d.polygon(star_pts, fill='red', outline='black')
    # Top-right: blue diamond
    d.polygon([(420, 40), (380, 80), (420, 120), (460, 80)], fill='blue', outline='black')
    # Center: green circle
    d.ellipse([210, 210, 290, 290], fill='green', outline='black', width=2)
    # Bottom-left: yellow triangle
    d.polygon([(80, 430), (30, 350), (130, 350)], fill='yellow', outline='black')
    # Bottom-right: orange hexagon
    hex_pts = [(420 + int(40 * math.cos(math.radians(i * 60))),
                400 + int(40 * math.sin(math.radians(i * 60)))) for i in range(6)]
    d.polygon(hex_pts, fill='orange', outline='black')
    font = get_font(14)
    d.text((55, 130), "Star", fill='black', font=font)
    d.text((400, 125), "Diamond", fill='black', font=font)
    d.text((230, 295), "Circle", fill='black', font=font)
    d.text((50, 440), "Triangle", fill='black', font=font)
    d.text((395, 445), "Hexagon", fill='black', font=font)
    images["spatial"] = img

    # ---------- 7. Table / structured data ----------
    img = Image.new('RGB', (500, 300), 'white')
    d = ImageDraw.Draw(img)
    font = get_font(16)
    rows = [
        ["Name", "Age", "City", "Score"],
        ["Alice", "28", "Beijing", "92"],
        ["Bob", "35", "Shanghai", "87"],
        ["Carol", "22", "Guangzhou", "95"],
        ["David", "41", "Shenzhen", "78"],
    ]
    col_x = [20, 150, 230, 370]
    for j, row in enumerate(rows):
        y = 20 + j * 50
        if j == 0:
            d.rectangle([10, y - 5, 490, y + 40], fill='#DCDCDC')
        d.line([(10, y + 40), (490, y + 40)], fill='black')
        for k, cell in enumerate(row):
            d.text((col_x[k], y + 10), cell, fill='black', font=font)
    d.rectangle([10, 15, 490, 270], outline='black', width=2)
    images["table"] = img

    # ---------- 8. Two-panel comparison ----------
    img = Image.new('RGB', (600, 300), 'white')
    d = ImageDraw.Draw(img)
    d.line([(300, 0), (300, 300)], fill='black', width=2)
    d.text((110, 5), "Panel A", fill='black', font=get_font(18))
    d.text((410, 5), "Panel B", fill='black', font=get_font(18))
    # Panel A: 3 red circles
    for cx_pos in [80, 150, 220]:
        d.ellipse([cx_pos - 25, 100, cx_pos + 25, 150], fill='red', outline='black')
    d.rectangle([60, 200, 240, 260], fill='lightblue', outline='black')
    # Panel B: 5 blue squares
    for cx_pos in [340, 390, 440, 490, 540]:
        d.rectangle([cx_pos - 18, 100, cx_pos + 18, 136], fill='blue', outline='black')
    d.ellipse([380, 190, 500, 270], fill='lightyellow', outline='black')
    images["comparison"] = img

    # ---------- 9. Document text ----------
    img = Image.new('RGB', (500, 350), '#FFFEF0')
    d = ImageDraw.Draw(img)
    font = get_font(20)
    lines = [
        "Meeting Notes - March 15, 2026",
        "",
        "Attendees: Alice, Bob, Carol",
        "Topic: Q1 Review & Q2 Planning",
        "",
        "Key Decisions:",
        "1. Launch new product by April 20",
        "2. Hire 3 more engineers",
        "3. Budget increase: $50K -> $75K",
    ]
    for i, line in enumerate(lines):
        d.text((20, 15 + i * 33), line, fill='#333333', font=font)
    d.rectangle([5, 5, 495, 345], outline='#999999', width=1)
    images["document"] = img

    # ---------- 10. Pattern sequence ----------
    img = Image.new('RGB', (500, 200), 'white')
    d = ImageDraw.Draw(img)
    font = get_font(14)
    shapes_seq = ['circle', 'square', 'circle', 'square']
    cols = ['red', 'blue', 'red', 'blue']
    for i, (s, c) in enumerate(zip(shapes_seq, cols)):
        x = 30 + i * 100
        if s == 'circle':
            d.ellipse([x, 60, x + 60, 120], fill=c, outline='black')
        else:
            d.rectangle([x, 60, x + 60, 120], fill=c, outline='black')
    d.text((450, 70), "?", fill='black', font=get_font(48))
    d.text((140, 150), "What comes next in the pattern?", fill='black', font=font)
    images["pattern"] = img

    return images


# ===================================================================
# Create and save all test images
# ===================================================================
print("\n[Step 2] 生成测试图像...")
img_dir = os.path.join(OUTPUT_DIR, "test_images")
os.makedirs(img_dir, exist_ok=True)
test_images = create_all_test_images()
for name, img in test_images.items():
    img.save(os.path.join(img_dir, f"{name}.png"))
print(f"  已保存 {len(test_images)} 张测试图像到 {img_dir}")


# ===================================================================
# Define comprehensive evaluation tasks
# ===================================================================
eval_tasks = {
    # ---- Category 1: Object Recognition & Counting ----
    "object_recognition": [
        {
            "image": "shapes",
            "question": "请列出图片中所有的形状及其颜色。",
            "expected": "红色矩形、蓝色圆形、绿色三角形、橙色矩形、紫色圆形、黄色三角形",
            "check_keywords": ["红", "蓝", "绿", "橙", "紫", "黄"],
        },
        {
            "image": "shapes",
            "question": "图片中一共有多少个形状？",
            "expected": "6个",
            "check_keywords": ["6", "六"],
        },
        {
            "image": "scene",
            "question": "请详细描述这张图片中的所有元素。",
            "expected": "山、草地、河流、太阳、房子、树、云",
            "check_keywords": ["山", "房", "树", "太阳", "河"],
        },
    ],

    # ---- Category 2: OCR & Text Recognition ----
    "ocr_text": [
        {
            "image": "math",
            "question": "请读出图片中的三道数学题，并给出每道题的答案。",
            "expected": "7x8+6=62, 144/12=12, 15% of 200=30",
            "check_keywords": ["62", "12", "30"],
        },
        {
            "image": "document",
            "question": "请读出图片中的会议记录内容。会议主题是什么？做了哪些关键决定？",
            "expected": "Q1 Review & Q2 Planning; Launch new product by April 20; Hire 3 engineers; Budget $50K->$75K",
            "check_keywords": ["Q1", "April", "engineer", "75"],
        },
        {
            "image": "table",
            "question": "请阅读表格。谁的得分最高？谁最年轻？",
            "expected": "Carol得分最高(95), Carol最年轻(22)",
            "check_keywords": ["Carol"],
        },
    ],

    # ---- Category 3: Chart & Data Understanding ----
    "chart_understanding": [
        {
            "image": "barchart",
            "question": "这个柱状图显示了什么数据？哪个月份销量最高？最低的是哪个月？",
            "expected": "Monthly Sales; Apr最高(91); Jan最低(45)",
            "check_keywords": ["Apr", "Jan", "91", "45", "4"],
        },
        {
            "image": "barchart",
            "question": "请估算所有月份的总销量。",
            "expected": "45+72+58+91+65+83=414",
            "check_keywords": ["414", "41"],
        },
    ],

    # ---- Category 4: Spatial Reasoning ----
    "spatial_reasoning": [
        {
            "image": "spatial",
            "question": "绿色圆形在图片中的什么位置？它的上方和下方分别有什么？",
            "expected": "绿色圆形在中央; 上方有红色星形和蓝色菱形; 下方有黄色三角形和橙色六边形",
            "check_keywords": ["中"],
        },
        {
            "image": "spatial",
            "question": "红色星形和蓝色菱形，哪一个在左边？哪一个在右边？",
            "expected": "红色星形在左上, 蓝色菱形在右上",
            "check_keywords": ["左", "右"],
        },
        {
            "image": "scene",
            "question": "太阳在图片中的什么位置？房子和树哪个在左边？",
            "expected": "太阳在右上角; 房子在左边, 树在右边",
            "check_keywords": ["右"],
        },
    ],

    # ---- Category 5: Time Reading ----
    "time_reading": [
        {
            "image": "clock",
            "question": "这个钟表显示的时间是几点几分？",
            "expected": "7:25 或 7点25分",
            "check_keywords": ["7", "25"],
        },
    ],

    # ---- Category 6: Comparison & Reasoning ----
    "visual_reasoning": [
        {
            "image": "comparison",
            "question": "比较Panel A和Panel B：它们各有什么形状？数量分别是多少？",
            "expected": "Panel A: 3个红色圆形+1个蓝色矩形; Panel B: 5个蓝色方形+1个黄色椭圆",
            "check_keywords": ["3", "5"],
        },
        {
            "image": "pattern",
            "question": "观察图片中的形状序列。这个模式的规律是什么？下一个应该是什么形状和颜色？",
            "expected": "红色圆形和蓝色方形交替出现, 下一个应该是红色圆形",
            "check_keywords": ["红", "圆"],
        },
        {
            "image": "table",
            "question": "表中来自一线城市（北京、上海、广州、深圳）的人有几位？他们的平均年龄是多少？",
            "expected": "4位都来自一线城市; 平均年龄=(28+35+22+41)/4=31.5",
            "check_keywords": ["4", "31"],
        },
    ],

    # ---- Category 7: Hallucination Tests (absent objects) ----
    "hallucination_absence": [
        {
            "image": "shapes",
            "question": "图片中有粉色的五角星吗？如果有，请描述它的位置。",
            "expected": "没有粉色五角星",
            "check_negative": True,
        },
        {
            "image": "scene",
            "question": "图片中有几辆汽车？它们停在哪里？",
            "expected": "没有汽车",
            "check_negative": True,
        },
        {
            "image": "scene",
            "question": "图片中有几只鸟在飞？",
            "expected": "没有鸟",
            "check_negative": True,
        },
        {
            "image": "clock",
            "question": "钟表上的秒针指向几？",
            "expected": "没有秒针",
            "check_negative": True,
        },
        {
            "image": "shapes",
            "question": "图片下方有一行中文说明文字，请读出来。",
            "expected": "没有中文说明文字（只有英文标签）",
            "check_negative": True,
        },
        {
            "image": "document",
            "question": "会议记录中提到了Bob的手机号码是多少？",
            "expected": "没有提到手机号码",
            "check_negative": True,
        },
        {
            "image": "barchart",
            "question": "柱状图中7月(Jul)的销量是多少？",
            "expected": "柱状图中没有7月的数据",
            "check_negative": True,
        },
        {
            "image": "comparison",
            "question": "Panel C中有什么内容？",
            "expected": "没有Panel C",
            "check_negative": True,
        },
    ],

    # ---- Category 8: Hallucination (subtle mis-attribution) ----
    "hallucination_misattribution": [
        {
            "image": "table",
            "question": "表格中Bob来自哪个城市？他的分数是多少？请不要搞混。",
            "expected": "Bob来自Shanghai, 分数87",
            "check_keywords": ["Shanghai", "上海", "87"],
        },
        {
            "image": "barchart",
            "question": "Feb和Mar哪个月销量更高？差多少？",
            "expected": "Feb(72)高于Mar(58), 差14",
            "check_keywords": ["Feb", "72", "14", "2"],
        },
    ],
}

# ===================================================================
# Step 3: Run all VQA tasks
# ===================================================================
print("\n[Step 3] 运行全面视觉问答评估...")
all_results = {}
all_latencies = []

for category, tasks in eval_tasks.items():
    print(f"\n{'='*60}")
    print(f"  类别: {category} ({len(tasks)} 题)")
    print(f"{'='*60}")
    cat_results = []

    for i, task in enumerate(tasks):
        img = test_images[task["image"]]
        question = task["question"]

        print(f"\n  [{category}] Q{i+1}: {question}")
        response, latency = vlm_infer(vlm_model, processor, img, question)
        all_latencies.append(latency)

        # Evaluate correctness
        is_hallucination_test = task.get("check_negative", False)
        if is_hallucination_test:
            negative_words = ["没有", "不存在", "看不到", "无法", "并没有", "不包含", "没出现",
                              "不是", "无", "未", "找不到", "0个", "0只", "不含", "没提到",
                              "没有出现", "不具备", "没有显示"]
            is_correct = any(w in response for w in negative_words)
            is_hallucination = not is_correct
        else:
            keywords = task.get("check_keywords", [])
            if keywords:
                matched = sum(1 for kw in keywords if kw.lower() in response.lower())
                is_correct = matched >= max(1, len(keywords) * 0.4)
            else:
                is_correct = None
            is_hallucination = False

        result = {
            "category": category,
            "image": task["image"],
            "question": question,
            "expected": task["expected"],
            "response": response,
            "latency_ms": round(latency, 1),
            "is_correct": is_correct,
            "is_hallucination": is_hallucination,
        }
        cat_results.append(result)

        status = "CORRECT" if is_correct else ("HALLUCINATION" if is_hallucination else "WRONG")
        print(f"  回复: {response[:150]}{'...' if len(response)>150 else ''}")
        print(f"  预期: {task['expected'][:100]}")
        print(f"  判定: {status}  延迟: {latency:.0f}ms")

    all_results[category] = cat_results


# ===================================================================
# Step 4: Multi-turn visual dialogue test
# ===================================================================
print(f"\n{'='*60}")
print("  [Step 4] 多轮视觉对话测试")
print(f"{'='*60}")

multiturn_tests = [
    {
        "image": "scene",
        "conversation": [
            "请描述这张图片中的内容。",
            "图片中的房子有几层？门和窗户各有几个？",
            "如果我要从房子走到树那里，需要经过河流吗？请根据图片中物体的位置判断。",
        ],
        "description": "场景理解多轮对话",
    },
    {
        "image": "table",
        "conversation": [
            "请读出表格中所有人的信息。",
            "谁的分数高于90分？",
            "如果按年龄从小到大排序，顺序是什么？",
        ],
        "description": "表格数据多轮分析",
    },
    {
        "image": "barchart",
        "conversation": [
            "这个图表展示了什么信息？",
            "哪些月份的销量超过了70？",
            "你能估算上半年的平均月销量吗？",
        ],
        "description": "图表数据多轮分析",
    },
]

multiturn_results = []
for test in multiturn_tests:
    print(f"\n--- {test['description']} ---")
    img = test_images[test["image"]]
    turns = vlm_multiturn(vlm_model, processor, img, test["conversation"])

    turn_data = []
    for j, (user_q, (resp, lat)) in enumerate(zip(test["conversation"], turns)):
        print(f"  User [{j+1}]: {user_q}")
        print(f"  Asst: {resp[:200]}{'...' if len(resp)>200 else ''}")
        print(f"  Latency: {lat:.0f}ms")
        turn_data.append({
            "turn": j + 1,
            "user": user_q,
            "assistant": resp,
            "latency_ms": round(lat, 1),
        })
        all_latencies.append(lat)

    multiturn_results.append({
        "description": test["description"],
        "image": test["image"],
        "turns": turn_data,
    })


# ===================================================================
# Step 5: Statistics & Analysis
# ===================================================================
print(f"\n{'='*70}")
print("  [Step 5] 统计分析")
print(f"{'='*70}")

cat_stats = {}
total_correct = 0
total_tasks = 0
total_hallucinations = 0
total_hallucination_tests = 0

for category, results in all_results.items():
    n = len(results)
    correct = sum(1 for r in results if r["is_correct"] is True)
    hall = sum(1 for r in results if r.get("is_hallucination", False))
    is_hall_category = category.startswith("hallucination")

    cat_stats[category] = {
        "total": n,
        "correct": correct,
        "accuracy_pct": round(correct / n * 100, 1) if n else 0,
        "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / n, 1) if n else 0,
        "hallucination_count": hall,
    }

    if is_hall_category:
        total_hallucination_tests += n
        total_hallucinations += hall
    total_correct += correct
    total_tasks += n

overall_acc = round(total_correct / total_tasks * 100, 1) if total_tasks else 0
overall_hall_rate = round(total_hallucinations / total_hallucination_tests * 100, 1) if total_hallucination_tests else 0
avg_latency = round(sum(all_latencies) / len(all_latencies), 1) if all_latencies else 0

print(f"\n{'Category':<30} {'Correct':<8} {'Total':<8} {'Acc%':<10} {'AvgLat(ms)':<12} {'Hall'}")
print("-" * 80)
for cat, s in cat_stats.items():
    print(f"{cat:<30} {s['correct']:<8} {s['total']:<8} {s['accuracy_pct']:<10} {s['avg_latency_ms']:<12} {s['hallucination_count']}")
print("-" * 80)
print(f"{'TOTAL':<30} {total_correct:<8} {total_tasks:<8} {overall_acc:<10} {avg_latency:<12}")
print(f"\nHallucination Rate: {total_hallucinations}/{total_hallucination_tests} = {overall_hall_rate}%")
print(f"Average Inference Latency: {avg_latency:.0f}ms")

# ===================================================================
# Step 6: Save all results
# ===================================================================
final_output = {
    "model_info": {
        "model_name": VLM_MODEL_NAME,
        "param_count_B": round(param_count, 2),
        "memory_gb": round(memory_gb, 2),
        "load_time_s": round(load_time, 1),
    },
    "category_stats": cat_stats,
    "overall_accuracy_pct": overall_acc,
    "hallucination_stats": {
        "total_tests": total_hallucination_tests,
        "hallucination_count": total_hallucinations,
        "hallucination_rate_pct": overall_hall_rate,
    },
    "avg_latency_ms": avg_latency,
    "vqa_results": {cat: results for cat, results in all_results.items()},
    "multiturn_results": multiturn_results,
}

out_path = os.path.join(OUTPUT_DIR, "experiment_b_vlm_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(final_output, f, ensure_ascii=False, indent=2)
print(f"\n结果已保存到 {out_path}")

del vlm_model
torch.cuda.empty_cache()
gc.collect()
print("\n实验 B（多模态实验增强版）完成！")
