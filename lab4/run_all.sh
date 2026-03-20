#!/bin/bash
# Lab 4: Mini DeepSeek-R1-Zero — Complete Experiment Pipeline
# Run on GPU 0,1,2,3 for training; GPU 0 for evaluation
set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

OUTPUT_BASE="$SCRIPT_DIR/outputs"
MAIN_OUTPUT="$OUTPUT_BASE/grpo-qwen3-1.7b-math"
FORMAT_OUTPUT="$OUTPUT_BASE/grpo-format-reward"
ABLATION_OUTPUT="$OUTPUT_BASE/ablation"
EVAL_OUTPUT="$OUTPUT_BASE/eval_results"
DELIVERABLES="$OUTPUT_BASE/deliverables"

mkdir -p "$OUTPUT_BASE" "$EVAL_OUTPUT" "$DELIVERABLES"

CONDA_ENV="rlf"

echo "============================================================"
echo "Lab 4: Complete Experiment Pipeline"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Output: $OUTPUT_BASE"
echo "============================================================"

# ── Step 1: Main GRPO Training (500 steps) ────────────────────────────────
echo ""
echo "[Step 1/5] Main GRPO Training (500 steps, G=8, beta=0.04)"
echo "============================================================"
conda run -n $CONDA_ENV python train_grpo.py \
    --max-steps 500 \
    --num-generations 8 \
    --beta 0.04 \
    --output-dir "$MAIN_OUTPUT" \
    --save-steps 100

# ── Step 2: Bonus — Format Reward Training (100 steps) ───────────────────
echo ""
echo "[Step 2/5] Format Reward Training (100 steps)"
echo "============================================================"
conda run -n $CONDA_ENV python train_grpo.py \
    --max-steps 100 \
    --num-generations 8 \
    --beta 0.04 \
    --output-dir "$FORMAT_OUTPUT" \
    --save-steps 100 \
    --use-format-reward

# ── Step 3: Bonus — Ablation Experiments (100 steps each) ────────────────
echo ""
echo "[Step 3/5] Ablation Experiments (G and beta)"
echo "============================================================"
conda run -n $CONDA_ENV python bonus_ablation.py \
    --output-dir "$ABLATION_OUTPUT" \
    --max-steps 100

# ── Step 4: Four-way Evaluation ──────────────────────────────────────────
echo ""
echo "[Step 4/5] Four-way GSM8K Evaluation"
echo "============================================================"
conda run -n $CONDA_ENV python evaluate_all.py \
    --grpo-model-dir "$MAIN_OUTPUT/final" \
    --output-dir "$EVAL_OUTPUT" \
    --num-eval-samples 50 \
    --num-chain-examples 5

# ── Step 5: Generate Deliverables ────────────────────────────────────────
echo ""
echo "[Step 5/5] Generating Deliverables"
echo "============================================================"
conda run -n $CONDA_ENV python generate_deliverables.py \
    --training-log "$MAIN_OUTPUT/training_log.json" \
    --comparison-results "$EVAL_OUTPUT/gsm8k_comparison.json" \
    --reasoning-chains "$EVAL_OUTPUT/reasoning_chains.json" \
    --emergence-analysis "$EVAL_OUTPUT/emergence_analysis.json" \
    --ablation-dir "$ABLATION_OUTPUT" \
    --ablation-summary "$ABLATION_OUTPUT/ablation_summary.json" \
    --format-reward-log "$FORMAT_OUTPUT/training_log.json" \
    --output-dir "$DELIVERABLES"

echo ""
echo "============================================================"
echo "All experiments complete!"
echo "Deliverables in: $DELIVERABLES"
echo "  - grpo_training_curves.png"
echo "  - ablation_comparison.png"
echo "  - grpo_format_reward_curves.png"
echo "  - lab4_report.md"
echo "============================================================"
