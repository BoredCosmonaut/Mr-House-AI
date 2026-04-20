import torch
import os
import shutil
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
from src.data_loader import get_house_dataset

def train():
    # --- STEP 0: THE PURGE ---
    # Delete old folders to ensure no leftover "dirty" weights interfere
    for folder in ["outputs", "house_lora_final"]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"Deleted old folder: {folder}")

    # 1. Load Model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/llama-3-8b",
        max_seq_length = 2048,
        dtype = torch.bfloat16,
        load_in_4bit = False,
    )

    # 2. THE STABILIZER: Lower Rank (64) prevents "Script Memorization"
    # This forces the model to be creative rather than just repeating the CSV.
    model = FastLanguageModel.get_peft_model(
        model,
        r = 64,               # Reduced from 128
        lora_alpha = 64,      # Reduced from 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout = 0.05,  # Added a tiny bit of dropout to prevent overfitting
        bias = "none",
    )

    # 3. Load Data (USE A NEW FILENAME IF YOU RENAMED YOUR CLEAN CSV)
    # Pro-tip: Rename your CSV to 'house_v2_clean.csv' to bypass the disk cache.
    dataset = get_house_dataset("data/house_v2_clean.csv") 
    dataset_split = dataset.train_test_split(test_size=0.1, seed=3407)

    # 4. Training Arguments - QUALITY OVER QUANTITY
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
            max_steps = 200,               # Reduced from 400 (prevent playback loop)
            learning_rate = 1e-4,          # Slightly higher LR to learn the "accent" faster
            lr_scheduler_type = "cosine",
            warmup_steps = 10,
            bf16 = True,
            logging_steps = 1,
            eval_strategy = "steps",
            eval_steps = 20,
            save_strategy = "steps",
            save_steps = 20,
            load_best_model_at_end = True,
            output_dir = "outputs",
            weight_decay = 0.1,            # Higher decay to "starve" the bad tokens
            optim = "adamw_8bit",
        ),
        callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
    )

    print("--- House is scanning for system corruption... ---")
    trainer.train()
    
    model.save_pretrained("house_lora_final")
    tokenizer.save_pretrained("house_lora_final")
    print("--- Training Complete. The Lucky 38 is running a clean OS. ---")

if __name__ == "__main__":
    train()