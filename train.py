import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
from src.data_loader import get_house_dataset

def train():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/llama-3-8b",
        max_seq_length = 2048,
        dtype = torch.bfloat16, # Optimized for A100
        load_in_4bit = False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 128,              # Much higher capacity
        lora_alpha = 128,     # Forces House's personality over the base model
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout = 0,
        bias = "none",
    )

    dataset = get_house_dataset("data/dialogueExportVDialogueMrHouse.csv")
    dataset_split = dataset.train_test_split(test_size=0.1, seed=3407)

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
            max_steps = 300,
            learning_rate = 1e-4,
            bf16 = True,
            logging_steps = 1,
            eval_strategy = "steps",
            eval_steps = 20,
            save_strategy = "steps",
            save_steps = 20,
            load_best_model_at_end = True,
            output_dir = "outputs",
        ),
        callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
    )

    trainer.train()
    model.save_pretrained("house_lora_final")
    tokenizer.save_pretrained("house_lora_final")

if __name__ == "__main__":
    train()