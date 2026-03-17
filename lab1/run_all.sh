#!/bin/bash
# ============================================================
# Lab1: 微调 Qwen3-1.7B 为指令跟随助手 —— 全流程运行脚本
# ============================================================
# 用法:
#   chmod +x run_all.sh
#   ./run_all.sh              # 运行全部步骤（默认方案A超参数实验）
#   ./run_all.sh --plan b     # 使用方案B（LoRA秩对比）
#   ./run_all.sh --skip-step5 # 跳过超参数探索步骤
# ============================================================

set -e

# 指定使用单张 GPU，量化模型不支持 DataParallel 多卡

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 解析参数
PLAN="b"
SKIP_STEP5=false
for arg in "$@"; do
    case $arg in
        --plan)
            shift
            PLAN="$1"
            shift
            ;;
        --plan=*)
            PLAN="${arg#*=}"
            shift
            ;;
        --skip-step5)
            SKIP_STEP5=true
            shift
            ;;
    esac
done

echo "============================================================"
echo " Lab1: 微调 Qwen3-1.7B 为指令跟随助手"
echo " 开始时间: $(date)"
echo "============================================================"

# 步骤 1: 环境验证与模型加载
echo ""
echo ">>> [步骤 1/5] 环境验证与模型加载..."
python step1_env_model_load.py
echo ">>> 步骤 1 完成。"

# 步骤 2: 数据准备与格式化
echo ""
echo ">>> [步骤 2/5] 数据准备与格式化..."
python step2_data_prep.py
echo ">>> 步骤 2 完成。"

# 步骤 3: SFT 训练
echo ""
echo ">>> [步骤 3/5] SFT 训练..."
python step3_train.py
echo ">>> 步骤 3 完成。"

# 步骤 4: 对比评估
echo ""
echo ">>> [步骤 4/5] 对比评估..."
python step4_eval.py
echo ">>> 步骤 4 完成。"

# 步骤 5: 超参数探索
if [ "$SKIP_STEP5" = false ]; then
    echo ""
    echo ">>> [步骤 5/5] 超参数探索 (方案 $PLAN)..."
    python step5_hyperparam.py --plan "$PLAN"
    echo ">>> 步骤 5 完成。"
else
    echo ""
    echo ">>> [步骤 5/5] 已跳过超参数探索。"
fi

echo ""
echo "============================================================"
echo " Lab1 全部完成！"
echo " 结束时间: $(date)"
echo " 输出文件位于: $SCRIPT_DIR/outputs/"
echo "============================================================"
echo ""
echo "产出文件清单："
echo "  outputs/baseline_results.json       - 基座模型基线输出"
echo "  outputs/training_loss.png           - 训练损失曲线"
echo "  outputs/evaluation_results.json     - 微调模型评估结果"
echo "  outputs/hyperparam_results.json     - 超参数实验结果"
echo "  outputs/hyperparam_comparison_*.png - 超参数对比图"
echo "  qwen3-1.7b-sft/final/              - 训练好的模型权重"
