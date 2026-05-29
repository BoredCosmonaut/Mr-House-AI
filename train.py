import torch
import os
import shutil
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
from src.data_loader import get_house_dataset


def train():

    for folder in ["outputs", "house_lora_final"]:
        if os.path.exists(folder):
            shutil.rmtree(folder)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Meta-Llama-3.1-8B-Instruct",
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, 
        bias="none",
    )


    dataset = get_house_dataset("data/house_v2_clean.csv")
    dataset_split = dataset.train_test_split(test_size=0.1, seed=3407)

    print(f"Train examples: {len(dataset_split['train'])}")
    print(f"Eval examples:  {len(dataset_split['test'])}")


    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_split["train"],
        eval_dataset=dataset_split["test"],
        dataset_text_field="text",
        max_seq_length=2048,
        args=TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,   
            max_steps=500,
            learning_rate=2e-4,                 
            lr_scheduler_type="cosine",
            warmup_steps=30,
            bf16=True,
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=50,
            save_total_limit=3,                
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            output_dir="outputs",
            weight_decay=0.01,                 
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