"""
Lab3 配置文件 —— DPO 对齐与 SimPO 对比实验
"""
import os

# ==================== GPU 配置 ====================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ==================== 基本路径 ====================
LAB_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(LAB_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 模型配置 ====================
BASE_MODEL = "Qwen/Qwen3-1.7B"  # SFT 基座模型

# ==================== 数据集配置 ====================
DATASET_NAME = "HuggingFaceH4/ultrafeedback_binarized"
TRAIN_SPLIT = "train_prefs"
TEST_SPLIT = "test_prefs"
TRAIN_SUBSET_SIZE = 5000
EVAL_SUBSET_SIZE = 500
SEED = 42

# ==================== LoRA 配置 ====================
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ==================== DPO 训练配置 ====================
DPO_OUTPUT_DIR = os.path.join(LAB_DIR, "dpo-qwen3-1.7b")
DPO_BETA = 0.1
DPO_LOSS_TYPE = "sigmoid"
DPO_LEARNING_RATE = 5e-7
DPO_NUM_EPOCHS = 1
DPO_BATCH_SIZE = 2
DPO_GRAD_ACCUM = 4
DPO_MAX_LENGTH = 1024
DPO_MAX_PROMPT_LENGTH = 512

# ==================== SimPO 训练配置 ====================
SIMPO_OUTPUT_DIR = os.path.join(LAB_DIR, "simpo-qwen3-1.7b")
SIMPO_BETA = 2.0
SIMPO_GAMMA = 0.5
SIMPO_LOSS_TYPE = "simpo"

# ==================== 推理配置 ====================
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9

# ==================== LLM-as-Judge 配置 ====================
DASHSCOPE_API_KEY = os.environ.get(
    "DASHSCOPE_API_KEY",
    "sk-1a59751f898e43a4b24219c4fba770d8"
)
JUDGE_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
JUDGE_MODEL = "qwen3-max"

# ==================== Beta 消融配置 ====================
ABLATION_BETAS = [0.05, 0.1, 0.5]
ABLATION_TRAIN_SIZE = 2000  # 消融实验用更小的数据集加速
