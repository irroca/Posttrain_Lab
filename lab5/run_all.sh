#!/bin/bash
# Lab 5 完整实验运行脚本
# 使用 rlf conda 环境, GPU 4,5,6,7

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "第5课 上机实验 - 开始运行"
echo "=============================================="
echo "工作目录: $SCRIPT_DIR"
echo "时间: $(date)"
echo ""

# 实验 A: 量化实验
echo "=============================================="
echo "运行实验 A：量化实验（必做）"
echo "=============================================="
python experiment_a_quantization.py 2>&1 | tee experiment_a_log.txt

echo ""
echo "=============================================="
echo "运行实验 B：迷你多模态实验（选做）"
echo "=============================================="
python experiment_b_multimodal.py 2>&1 | tee experiment_b_log.txt

echo ""
echo "=============================================="
echo "生成实验报告"
echo "=============================================="
python generate_report.py 2>&1 | tee report_generation_log.txt

echo ""
echo "=============================================="
echo "全部实验完成！"
echo "=============================================="
echo "报告: $SCRIPT_DIR/lab5_report.md"
echo "图表: $SCRIPT_DIR/figures/"
echo "结束时间: $(date)"
