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
        load_in_4bit = False,
    )

    # 2. THE LOCKDOWN: Higher Rank, Higher Alpha
    # This makes the "House" layer much stronger than the base Llama layer
    model = FastLanguageModel.get_peft_model(
        model,
        r = 128,              # Increased from 64
        lora_alpha = 128,     # Increased from 64
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout = 0,
        bias = "none",
    )

    # 3. Load Data
    dataset = get_house_dataset("data/dialogueExportVDialogueMrHouse.csv")
    dataset_split = dataset.train_test_split(test_size=0.1, seed=3407)

    # 4. Training Arguments - THE PRECISION FIX
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
            max_steps = 400,               # Give it more time to learn relationships
            learning_rate = 5e-5,          # LOWER LR (prevents "memorizing" noise/IDs)
            bf16 = True,
            logging_steps = 1,
            eval_strategy = "steps",
            eval_steps = 25,               # Check every 25 steps
            save_strategy = "steps",
            save_steps = 25,
            load_best_model_at_end = True,
            output_dir = "outputs",
            weight_decay = 0.01,           # Helps prevent overfitting on the script IDs
        ),
        # Patience of 5 is good, but let's make it 8 so it doesn't stop too early
        callbacks = [EarlyStoppingCallback(early_stopping_patience=8)]
    )

    print("--- Recalibrating the Lucky 38 OS... ---")
    trainer.train()
    
    model.save_pretrained("house_lora_final")
    tokenizer.save_pretrained("house_lora_final")
    print("--- Training Complete. Mr. House is now operational. ---")

if __name__ == "__main__":
    train()