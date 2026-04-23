import torch
import re
import warnings
import os
import logging
from unsloth import FastLanguageModel

# --- 1. SYSTEM INITIALIZATION ---
# Silence all library chatter before loading
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)

def load_persona(file_path="persona.txt"):
    """
    Loads the Robert House logic from persona.txt. 
    Appends the anti-roleplay constraint as a final safety layer.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            persona = f.read().strip()
    except FileNotFoundError:
        persona = "You are Robert House. Be cold, logical, and arrogant."
    
    # Surgical constraint to prevent the model from chatting with itself
    constraint = (
        "\n\n[SYSTEM DIRECTIVE]: Speak ONLY as Robert House. "
        "Do not write dialogue for the Courier. End your response immediately after your own statement."
    )
    return persona + constraint

def chat():
    # --- 2. MODEL LOADING ---
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "house_lora_final",
        max_seq_length = 2048,
        dtype = torch.bfloat16,
        load_in_4bit = False,
    )
    FastLanguageModel.for_inference(model)

    # --- 3. CONFIGURATION OVERRIDE ---
    # This specifically kills the 'max_new_tokens vs max_length' warning
    model.generation_config.max_length = None 

    # Define the 'Brick Wall' tokens
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        tokenizer.convert_tokens_to_ids("<|start_header_id|>"), 
        tokenizer.encode("user", add_special_tokens=False)[-1],
        tokenizer.encode("\n", add_special_tokens=False)[-1],
    ]
    
    instruction = load_persona()

    print("\n" + "="*45)
    print("      LUCKY 38 MAINFRAME: PROTOCOL LOADED")
    print("="*45 + "\n")

    while True:
        u = input("Courier: ")
        if u.lower() in ["exit", "quit"]: 
            print("\nConnection terminated. House always wins.")
            break
        
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{u}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        
        inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")

        # --- 4. GENERATION ---
        outputs = model.generate(
            **inputs, 
            max_new_tokens = 128,
            temperature = 0.35,            # Low temp ensures he sticks to the persona.txt rules
            top_p = 0.9,
            repetition_penalty = 1.3,      # Kills the 'ulator' and 'useruser' loops
            eos_token_id = terminators, 
            do_sample = True,
            use_cache = True,
        )
        
        full_text = tokenizer.batch_decode(outputs)[0]
        resp = full_text.split("assistant<|end_header_id|>\n\n")[-1]
        
        # --- 5. CLEANING & FILTERING ---
        # Remove special tokens
        resp = re.sub(r'<\|.*?\|>', '', resp).strip()
        
        # Stop Patterns (Anti-Self-Chatting)
        stop_patterns = ["<|start_header_id|>", "user", "Courier:", "User:", "Assistant:", "\n\n"]
        for pattern in stop_patterns:
            if pattern in resp:
                resp = resp.split(pattern)[0].strip()

        # Remove gibberish and artifacts
        resp = re.sub(r'[a-zA-Z]{18,}', '', resp)
        resp = resp.replace("igator", "").replace(".....", "").strip()

        # Sentence Trimmer (Ensures polished output)
        if " " in resp:
            last_punc = max(resp.rfind("."), resp.rfind("!"), resp.rfind("?"))
            if last_punc != -1:
                resp = resp[:last_punc + 1]

        # Final ASCII Polish
        resp = resp.encode("ascii", "ignore").decode().strip()

        if not resp:
            resp = "My calculations suggest this topic is a superfluous distraction."

        print(f"\nMr. House: {resp}\n")

if __name__ == "__main__":
    chat()