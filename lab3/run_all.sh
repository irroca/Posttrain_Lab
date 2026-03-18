#!/bin/bash
# Lab3: DPO 对齐与 SimPO 对比 —— 完整实验流程
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 使用 test 环境（含 transformers, trl, peft, bitsandbytes）
CONDA_ENV="test"
PYTHON="conda run --no-capture-output -n $CONDA_ENV python"

export CUDA_VISIBLE_DEVICES=0

echo "============================================================"
echo "Lab3: DPO 对齐与 SimPO 对比实验"
echo "============================================================"
echo "开始时间: $(date)"
echo ""

# Step 1: 数据探索
echo "[Step 1/7] 偏好数据探索..."
$PYTHON step1_data_explore.py 2>&1 | tee outputs/step1.log
echo ""

# Step 2: DPO 训练
echo "[Step 2/7] DPO 训练..."
$PYTHON step2_dpo_train.py 2>&1 | tee outputs/step2.log
echo ""

# Step 3: SimPO 训练
echo "[Step 3/7] SimPO 训练..."
$PYTHON step3_simpo_train.py 2>&1 | tee outputs/step3.log
echo ""

# Step 4: 三模型评估
echo "[Step 4/7] 三模型对比评估..."
$PYTHON step4_evaluate.py 2>&1 | tee outputs/step4.log
echo ""

# Step 5: Beta 消融实验
echo "[Step 5/7] Beta 消融实验..."
$PYTHON step5_beta_ablation.py 2>&1 | tee outputs/step5.log
echo ""

# Step 6: LLM-as-Judge
echo "[Step 6/7] LLM-as-Judge 自动评分..."
$PYTHON step6_llm_judge.py 2>&1 | tee outputs/step6.log
echo ""

# Step 7: 生成报告
echo "[Step 7/7] 生成交付物..."
$PYTHON step7_report.py 2>&1 | tee outputs/step7.log
echo ""

echo "============================================================"
echo "所有实验完成!"
echo "结束时间: $(date)"
echo "交付物目录: $SCRIPT_DIR/outputs/"
echo "============================================================"
