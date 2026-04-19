import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
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

    # 2. Load Data using our new module
    dataset = get_house_dataset("data/dialogueExportVDialogueMrHouse.csv")

    # 3. Training
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
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
            save_strategy = "no",
        ),
    )

    trainer.train()
    model.save_pretrained("house_lora_final")
    tokenizer.save_pretrained("house_lora_final")

if __name__ == "__main__":
    train()