# train.py
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset
import json

def train():
    # 1. Load Model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/llama-3-8b",
        max_seq_length = 2048,
        load_in_4bit = False, 
        dtype = torch.bfloat16,
    )

    # 2. Add LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r = 32,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 32,
        lora_dropout = 0,
        bias = "none",
    )

    # 3. Load Data (Assumes data is in the same repo)
    with open('data/house_data.json', 'r') as f:
        data = json.load(f)
    
    # ... (Include the formatting_prompts_func and dataset mapping logic here) ...

    # 4. Trainer
    trainer = SFTTrainer(
        model = model,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = 2048,
        args = TrainingArguments(
            output_dir = "house_model_output",
            per_device_train_batch_size = 4,
            max_steps = 150,
            learning_rate = 2e-4,
            bf16 = True,
            logging_steps = 1,
        ),
    )
    
    trainer.train()
    model.save_pretrained("house_lora_final")

if __name__ == "__main__":
    train()