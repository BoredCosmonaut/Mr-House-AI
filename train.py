import torch
import os
import shutil
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
from src.data_loader import get_house_dataset


def train():
    # Clear old weights
    for folder in ["outputs", "house_lora_final"]:
        if os.path.exists(folder):
            shutil.rmtree(folder)

    # 1. Load Model (A100 optimized)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Meta-Llama-3.1-8B-Instruct",
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )

    # 2. LoRA config
    # r=32 gives more capacity to overwrite Llama's assistant voice.
    # Previously r=16 was too conservative for a strong persona fine-tune.
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,  # Lower dropout now that we have more data -- less need to regularize hard
        bias="none",
    )

    # 3. Load combined dataset (game lines + synthetic)
    # If you haven't run generate_synthetic.py yet, use "data/house_v2_clean.csv"
    dataset = get_house_dataset("data/house_v2_clean.csv")
    dataset_split = dataset.train_test_split(test_size=0.1, seed=3407)

    print(f"Train examples: {len(dataset_split['train'])}")
    print(f"Eval examples:  {len(dataset_split['test'])}")

    # 4. Training arguments
    # max_steps bumped to 500 to account for the larger combined dataset.
    # eval/save every 50 steps so early stopping has enough resolution.
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_split["train"],
        eval_dataset=dataset_split["test"],
        dataset_text_field="text",
        max_seq_length=2048,
        args=TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,      # Effective batch size = 16
            max_steps=500,
            learning_rate=2e-4,                 # Standard for LoRA -- 5e-5 was too conservative
            lr_scheduler_type="cosine",
            warmup_steps=30,
            bf16=True,
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=50,
            save_total_limit=3,                 # Don't fill disk with every checkpoint
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            output_dir="outputs",
            weight_decay=0.01,                  # Lowered -- heavy weight decay fights LoRA on small datasets
            optim="adamw_8bit",
            report_to="none",
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    print("--- Calculating the House Advantage... ---")
    trainer.train()

    model.save_pretrained("house_lora_final")
    tokenizer.save_pretrained("house_lora_final")
    print("--- Training Complete. System logic is pure. ---")


if __name__ == "__main__":
    train()