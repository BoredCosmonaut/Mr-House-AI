import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
from src.data_loader import get_house_dataset

def train():
    # 1. Load Model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/llama-3-8b",
        max_seq_length = 2048,
        dtype = torch.bfloat16,
        load_in_4bit = False, # Keeping it high precision for the A100
    )

    # 2. THE LOCKDOWN
    model = FastLanguageModel.get_peft_model(
        model,
        r = 128,
        lora_alpha = 128,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout = 0,
        bias = "none",
    )

    # 3. Load Data
    dataset = get_house_dataset("data/dialogueExportVDialogueMrHouse.csv")
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
            max_steps = 400,
            learning_rate = 5e-5,
            lr_scheduler_type = "cosine", # Use Cosine for a smoother personality "blend"
            warmup_steps = 20,            # Let the model "wake up" before hitting full speed
            bf16 = True,
            logging_steps = 1,
            eval_strategy = "steps",
            eval_steps = 20,              # Slightly more frequent checks
            save_strategy = "steps",
            save_steps = 20,
            load_best_model_at_end = True,
            output_dir = "outputs",
            weight_decay = 0.05,          # Increased to force it to ignore any tiny script artifacts
            optim = "adamw_8bit",         # Standard for Unsloth
        ),
        callbacks = [EarlyStoppingCallback(early_stopping_patience=10)] # Increased patience
    )

    print("--- Recalibrating the Lucky 38 OS... ---")
    trainer.train()
    
    # Final export
    model.save_pretrained("house_lora_final")
    tokenizer.save_pretrained("house_lora_final")
    print("--- Training Complete. Mr. House is now operational. ---")

if __name__ == "__main__":
    train()