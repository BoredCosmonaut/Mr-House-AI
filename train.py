import os
import torch
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

def train_house():
    # 1. Model Setup (Optimized for A100)
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

    # 2. Load CSV/TSV Data
    # Your file uses Tabs (\t), so we use sep='\t'
    csv_path = "dialogueExportVDialogueMrHouse.csv"
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path} in the current directory.")

    print(f"--- Loading {csv_path} ---")
    df = pd.read_csv(csv_path, sep='\t')

    # 3. Clean and Format for Llama-3
    def format_prompt(row):
        # We use 'PROMPT' for the Courier and 'RESPONSE TEXT' for House
        # 'TOPIC TEXT' is used if 'PROMPT' is empty
        courier_text = str(row['PROMPT'] if pd.notna(row['PROMPT']) else row['TOPIC TEXT']).strip()
        house_text = str(row['RESPONSE TEXT']).strip()
        
        # System instruction to set the persona
        instruction = "You are Robert House, the 261-year-old CEO of RobCo and technocratic overlord of New Vegas. Speak with extreme sophistication, cold logic, and a condescending sense of superiority."
        
        # Llama-3 Instruct Format
        return { "text": f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|>" \
                         f"<|start_header_id|>user<|end_header_id|>\n\n{courier_text}<|eot_id|>" \
                         f"<|start_header_id|>assistant<|end_header_id|>\n\n{house_text}<|eot_id|>" }

    # Apply the formatting and convert to a Hugging Face Dataset
    formatted_data = df.apply(format_prompt, axis=1).tolist()
    dataset = Dataset.from_list(formatted_data)

    # 4. Trainer Setup
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = 2048,
        args = TrainingArguments(
            per_device_train_batch_size = 4,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 150,
            learning_rate = 2e-4,
            bf16 = True,
            logging_steps = 1,
            output_dir = "outputs",
        ),
    )

    print("--- House is analyzing the data... ---")
    trainer.train()
    
    model.save_pretrained("house_lora_final")
    tokenizer.save_pretrained("house_lora_final")
    print("--- Training Complete. Securitrons Upgraded. ---")

if __name__ == "__main__":
    train_house()