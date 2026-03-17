#!/bin/bash
# Lab2 完整实验运行脚本
# 使用 conda env: test

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 使用 GPU 0 (空闲)
export CUDA_VISIBLE_DEVICES=0

CONDA_ENV="test"
PYTHON="conda run --no-capture-output -n $CONDA_ENV python"

echo "============================================================"
echo "  Lab2: 构建领域定制 SFT 模型并系统评估"
echo "  开始时间: $(date)"
echo "============================================================"

echo ""
echo ">>> 步骤1: 数据集分析与预处理"
$PYTHON step1_data_analysis.py 2>&1 | tee outputs/step1.log

echo ""
echo ">>> 步骤2: 指令微调训练"
$PYTHON step2_train.py 2>&1 | tee outputs/step2.log

echo ""
echo ">>> 步骤3: LLM-as-Judge 评估"
$PYTHON step3_eval.py 2>&1 | tee outputs/step3.log

echo ""
echo ">>> 步骤4: 数据质量消融实验"
$PYTHON step4_ablation.py 2>&1 | tee outputs/step4.log

echo ""
echo ">>> 步骤5: 生成交付物"
$PYTHON step5_report.py 2>&1 | tee outputs/step5.log

echo ""
echo "============================================================"
echo "  Lab2 完成！"
echo "  结束时间: $(date)"
echo "  报告: outputs/lab2_report.md"
echo "============================================================"
