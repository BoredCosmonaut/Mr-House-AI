import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
from src.data_loader import get_house_dataset

def train():
    # 1. Model Setup
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/llama-3-8b",
        max_seq_length = 2048,
        dtype = torch.bfloat16,
        load_in_4bit = False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 32,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 32,
        lora_dropout = 0,
        bias = "none",
    )

    # 2. Load and Split Data
    # We split 10% for evaluation so the model can check its own progress
    full_dataset = get_house_dataset("data/dialogueExportVDialogueMrHouse.csv")
    dataset_split = full_dataset.train_test_split(test_size=0.1, seed=3407)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]

    # 3. Training with Early Stopping
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset, # Required for early stopping
        dataset_text_field = "text",
        max_seq_length = 2048,
        args = TrainingArguments(
            per_device_train_batch_size = 4,
            gradient_accumulation_steps = 4,
            max_steps = 150,
            learning_rate = 2e-4,
            bf16 = True,
            logging_steps = 1,
            output_dir = "outputs",
            
            # --- Early Stopping Requirements ---
            eval_strategy = "steps",      # Check progress every X steps
            eval_steps = 10,               # How often to run the "exam"
            save_strategy = "steps",       # Must match eval_strategy
            save_steps = 10,
            load_best_model_at_end = True, # Returns the "best" House, not the last one
            metric_for_best_model = "loss",
        ),
        # Stop if the loss doesn't improve for 3 checks (30 steps total)
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print("--- Training started. The House is observing. ---")
    trainer.train()
    
    # Save the absolute best version found during training
    model.save_pretrained("house_lora_final")
    tokenizer.save_pretrained("house_lora_final")
    print("--- Training Complete. Optimal persona captured. ---")

if __name__ == "__main__":
    train()