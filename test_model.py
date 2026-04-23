import torch
import re
import warnings
import logging
from unsloth import FastLanguageModel

# 1. Kill the logs at the root
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

def load_persona(file_path="persona.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "You are Robert House. Be cold, logical, and arrogant."

def chat():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "house_lora_final",
        max_seq_length = 2048,
        dtype = torch.bfloat16,
        load_in_4bit = False,
    )
    FastLanguageModel.for_inference(model)

    # --- THE WARNING KILLER ---
    # This removes the internal max_length default entirely
    model.generation_config.max_length = None 
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        tokenizer.encode("\n", add_special_tokens=False)[-1],
    ]
    
    instruction = load_persona()
    print("\n--- Lucky 38 Mainframe Online ---\n")

    while True:
        u = input("Courier: ")
        if u.lower() in ["exit", "quit"]: break
        
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{u}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        
        inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")

        outputs = model.generate(
            **inputs, 
            max_new_tokens = 128,
            temperature = 0.35,            # Slightly lower to prevent "Mormon Kimball" style hallucinations
            top_p = 0.9,
            repetition_penalty = 1.3,     # Keeps the logic on track
            eos_token_id = terminators, 
            do_sample = True,
            use_cache = True,
        )
        
        full_text = tokenizer.batch_decode(outputs)[0]
        resp = full_text.split("assistant<|end_header_id|>\n\n")[-1]
        
        # --- CLEANING ---
        resp = re.sub(r'<\|.*?\|>', '', resp).strip()
        
        # Trim half-finished sentences
        if " " in resp:
            last_punc = max(resp.rfind("."), resp.rfind("!"), resp.rfind("?"))
            if last_punc != -1:
                resp = resp[:last_punc + 1]

        # Prevent "Narrator Leaks" (when the AI starts talking about the character)
        if "House" in resp and "I" not in resp:
            resp = resp.split("House")[0].strip()

        resp = resp.encode("ascii", "ignore").decode().strip()

        if not resp:
            resp = "The question is irrelevant. My plans for the future do not concern such trivialities."

        print(f"\nMr. House: {resp}\n")

if __name__ == "__main__":
    chat()