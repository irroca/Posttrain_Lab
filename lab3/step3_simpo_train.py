"""
Step 3: SimPO 训练 —— 自定义 SimPO 实现（无参考模型偏好优化）

SimPO 核心思想：
- 使用序列平均对数概率作为隐式奖励: r(x,y) = (1/|y|) * log π(y|x)
- 损失: -log σ(β * (r_w - r_l - γ))
- 无需参考模型，训练更高效
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import config

import json
import time
import random
import sys
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model


class SimPOTrainerSimple:
    """简化版 SimPO 训练器"""

    def __init__(self, model, tokenizer, train_dataset, eval_dataset,
                 beta=2.0, gamma=0.5, lr=5e-7, batch_size=2,
                 grad_accum_steps=4, max_length=1024, num_epochs=1,
                 output_dir="./simpo-output", logging_steps=10):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.beta = beta
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.max_length = max_length
        self.num_epochs = num_epochs
        self.output_dir = output_dir
        self.logging_steps = logging_steps
        self.log_history = []

    def _prepare_pair(self, sample):
        """将一条偏好样本编码为 chosen/rejected"""
        prompt = sample["prompt"]

        def extract_response(field):
            if isinstance(field, list):
                return field[-1]["content"]
            return field

        chosen_text = extract_response(sample["chosen"])
        rejected_text = extract_response(sample["rejected"])

        chosen_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen_text},
        ]
        rejected_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected_text},
        ]

        chosen_ids = self.tokenizer.apply_chat_template(
            chosen_messages, tokenize=True, add_generation_prompt=False,
            max_length=self.max_length, truncation=True,
            enable_thinking=False,
        )
        rejected_ids = self.tokenizer.apply_chat_template(
            rejected_messages, tokenize=True, add_generation_prompt=False,
            max_length=self.max_length, truncation=True,
            enable_thinking=False,
        )
        # Ensure we have plain lists of ints
        if hasattr(chosen_ids, 'input_ids'):
            chosen_ids = chosen_ids['input_ids']
        if hasattr(rejected_ids, 'input_ids'):
            rejected_ids = rejected_ids['input_ids']
        if isinstance(chosen_ids, list) and len(chosen_ids) > 0 and isinstance(chosen_ids[0], list):
            chosen_ids = chosen_ids[0]
        if isinstance(rejected_ids, list) and len(rejected_ids) > 0 and isinstance(rejected_ids[0], list):
            rejected_ids = rejected_ids[0]
        return list(chosen_ids), list(rejected_ids)

    def _pad_batch(self, id_lists):
        """Pad and batch token id lists"""
        max_len = max(len(ids) for ids in id_lists)
        pad_id = self.tokenizer.pad_token_id or 0
        input_ids = []
        attention_masks = []
        labels = []
        for ids in id_lists:
            pad_len = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_len)
            attention_masks.append([1] * len(ids) + [0] * pad_len)
            labels.append(ids + [-100] * pad_len)

        device = next(self.model.parameters()).device
        return {
            "input_ids": torch.tensor(input_ids, device=device),
            "attention_mask": torch.tensor(attention_masks, device=device),
            "labels": torch.tensor(labels, device=device),
        }

    def _compute_avg_logprobs(self, batch):
        """计算序列平均对数概率"""
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = outputs.logits[:, :-1, :]
        targets = batch["labels"][:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        token_lp = log_probs.gather(2, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        mask = (targets != -100).float()
        avg_lp = (token_lp * mask).sum(-1) / mask.sum(-1).clamp(min=1)
        return avg_lp

    def train(self):
        """训练主循环"""
        self.model.train()
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr, weight_decay=0.01,
        )

        global_step = 0
        accum_loss = 0.0
        log_loss = 0.0
        log_count = 0

        for epoch in range(self.num_epochs):
            indices = list(range(len(self.train_dataset)))
            random.shuffle(indices)
            optimizer.zero_grad()

            for i in range(0, len(indices) - self.batch_size + 1, self.batch_size):
                batch_idx = indices[i:i + self.batch_size]
                batch_samples = [self.train_dataset[idx] for idx in batch_idx]

                chosen_ids_list = []
                rejected_ids_list = []
                for sample in batch_samples:
                    try:
                        c_ids, r_ids = self._prepare_pair(sample)
                        chosen_ids_list.append(c_ids)
                        rejected_ids_list.append(r_ids)
                    except Exception:
                        continue

                if not chosen_ids_list:
                    continue

                chosen_batch = self._pad_batch(chosen_ids_list)
                rejected_batch = self._pad_batch(rejected_ids_list)

                # Forward: compute avg log probs
                chosen_avg_lp = self._compute_avg_logprobs(chosen_batch)
                rejected_avg_lp = self._compute_avg_logprobs(rejected_batch)

                # SimPO loss
                margin = chosen_avg_lp - rejected_avg_lp
                loss = -F.logsigmoid(self.beta * (margin - self.gamma)).mean()
                loss = loss / self.grad_accum_steps
                loss.backward()

                step_loss = loss.item() * self.grad_accum_steps
                accum_loss += step_loss
                log_loss += step_loss
                log_count += 1
                global_step += 1

                if global_step % self.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                    effective_step = global_step // self.grad_accum_steps
                    if effective_step % self.logging_steps == 0 and log_count > 0:
                        avg_loss = log_loss / log_count
                        avg_margin = margin.mean().item()
                        entry = {
                            "step": effective_step,
                            "loss": avg_loss,
                            "rewards/margins": avg_margin,
                            "rewards/chosen": chosen_avg_lp.mean().item(),
                            "rewards/rejected": rejected_avg_lp.mean().item(),
                        }
                        self.log_history.append(entry)
                        print(f"  Step {effective_step}: loss={avg_loss:.4f}, margin={avg_margin:.4f}")
                        sys.stdout.flush()
                        log_loss = 0.0
                        log_count = 0

        final_loss = self.log_history[-1]["loss"] if self.log_history else 0
        return {"training_loss": final_loss}

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


def main():
    print("=" * 60)
    print("Step 3: SimPO 训练")
    print("=" * 60)

    # 1. 加载模型
    print("\n[1/4] 加载 SFT 基座模型...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA
    peft_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    # 2. 加载数据
    print("\n[2/4] 加载偏好数据...")
    dataset = load_dataset(config.DATASET_NAME, split=config.TRAIN_SPLIT)
    train_dataset = dataset.select(range(min(config.TRAIN_SUBSET_SIZE, len(dataset))))
    eval_dataset = load_dataset(
        config.DATASET_NAME, split=config.TEST_SPLIT
    ).select(range(config.EVAL_SUBSET_SIZE))
    print(f"训练集: {len(train_dataset)} 样本")

    # 3. SimPO 训练
    print("\n[3/4] 开始 SimPO 训练...")
    trainer = SimPOTrainerSimple(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        beta=config.SIMPO_BETA,
        gamma=config.SIMPO_GAMMA,
        lr=config.DPO_LEARNING_RATE,
        batch_size=config.DPO_BATCH_SIZE,
        grad_accum_steps=config.DPO_GRAD_ACCUM,
        max_length=config.DPO_MAX_LENGTH,
        num_epochs=config.DPO_NUM_EPOCHS,
        output_dir=config.SIMPO_OUTPUT_DIR,
        logging_steps=10,
    )

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    result = trainer.train()
    train_time = time.time() - start_time
    peak_memory = torch.cuda.max_memory_allocated() / 1e9

    final_loss = result["training_loss"]
    print(f"\n训练完成! 最终损失: {final_loss:.4f}")
    print(f"训练时间: {train_time / 60:.1f} 分钟")
    print(f"峰值显存: {peak_memory:.2f} GB")

    # 4. 保存
    save_path = os.path.join(config.SIMPO_OUTPUT_DIR, "final")
    trainer.save_model(save_path)
    print(f"SimPO 模型已保存到: {save_path}")

    # 保存指标
    simpo_metrics = {
        "train_loss": [l.get("loss") for l in trainer.log_history],
        "eval_loss": [],
        "rewards_chosen": [l.get("rewards/chosen") for l in trainer.log_history],
        "rewards_rejected": [l.get("rewards/rejected") for l in trainer.log_history],
        "reward_margins": [l.get("rewards/margins") for l in trainer.log_history],
        "train_steps": [l.get("step") for l in trainer.log_history],
        "eval_steps": [],
        "training_time_seconds": train_time,
        "peak_memory_gb": peak_memory,
        "final_loss": final_loss,
        "train_runtime": train_time,
    }

    if simpo_metrics["reward_margins"]:
        print(f"初始 reward margin: {simpo_metrics['reward_margins'][0]:.4f}")
        print(f"最终 reward margin: {simpo_metrics['reward_margins'][-1]:.4f}")

    metrics_path = os.path.join(config.OUTPUT_DIR, "simpo_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(simpo_metrics, f, indent=2)
    print(f"SimPO 指标已保存到: {metrics_path}")

    # 效率对比
    dpo_metrics_path = os.path.join(config.OUTPUT_DIR, "dpo_metrics.json")
    if os.path.exists(dpo_metrics_path):
        with open(dpo_metrics_path) as f:
            dpo_metrics = json.load(f)
        print("\n" + "=" * 50)
        print("训练效率对比")
        print("=" * 50)
        dpo_time = dpo_metrics.get("training_time_seconds", 0)
        print(f"DPO 训练时间:  {dpo_time / 60:.1f} 分钟")
        print(f"SimPO 训练时间: {train_time / 60:.1f} 分钟")
        if train_time > 0:
            print(f"SimPO 加速比:  {dpo_time / train_time:.2f}x")
        print(f"DPO 峰值显存:  {dpo_metrics.get('peak_memory_gb', 0):.2f} GB")
        print(f"SimPO 峰值显存: {peak_memory:.2f} GB")
        print(f"\nDPO 最终损失:  {dpo_metrics.get('final_loss', 0):.4f}")
        print(f"SimPO 最终损失: {final_loss:.4f}")

    # 清理
    del trainer, model
    torch.cuda.empty_cache()
    print("\nStep 3 完成!")


if __name__ == "__main__":
    main()
