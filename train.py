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
        model_name = "unsloth/llama-3-8b",
        max_seq_length = 2048,
        dtype = torch.bfloat16,
        load_in_4bit = False,
    )

    # 2. THE STABILIZER (r=32)
    # This prevents the model from being a "tape recorder" for the game script.
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,               
        lora_alpha = 32,      
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout = 0.1,   # Increased dropout to discourage memorization
        bias = "none",
    )

    # 3. Load Data
    dataset = get_house_dataset("data/house_v2_clean.csv") 
    dataset_split = dataset.train_test_split(test_size=0.1, seed=3407)

    # 4. Training Arguments
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset_split["train"],
        eval_dataset = dataset_split["test"],
        dataset_text_field = "text",
        max_seq_length = 2048,
        args = TrainingArguments(
            per_device_train_batch_size = 4,
            gradient_accumulation_steps = 4,
            max_steps = 250,
            learning_rate = 5e-5,
            lr_scheduler_type = "cosine",
            warmup_steps = 20,
            bf16 = True,
            logging_steps = 1,
            eval_strategy = "steps",
            eval_steps = 25,
            save_strategy = "steps",
            save_steps = 25,
            load_best_model_at_end = True,
            metric_for_best_model = "eval_loss",  # Added
            greater_is_better = False,            # Added
            output_dir = "outputs",
            weight_decay = 0.15,
            optim = "adamw_8bit",
            report_to = "none",                   # Keeps the console clean
        ),
        callbacks = [EarlyStoppingCallback(early_stopping_patience=4)]
    )

    print("--- Calculating the House Advantage... ---")
    trainer.train()
    
    model.save_pretrained("house_lora_final")
    tokenizer.save_pretrained("house_lora_final")
    print("--- Training Complete. System logic is pure. ---")

if __name__ == "__main__":
    train()