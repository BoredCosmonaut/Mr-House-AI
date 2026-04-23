import torch
import re
import warnings
import os
import logging
from unsloth import FastLanguageModel

# --- 1. SYSTEM INITIALIZATION ---
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)

def load_persona(file_path="persona.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            persona = f.read().strip()
    except FileNotFoundError:
        persona = "You are Robert House. Be cold, logical, and arrogant."
    
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
    model.generation_config.max_length = None 

    # Define the 'Brick Wall' tokens
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        tokenizer.convert_tokens_to_ids("<|start_header_id|>"), 
    ]
    
    instruction = load_persona()

    # --- 3. MEMORY INITIALIZATION ---
    history = [] 
    MAX_HISTORY = 4 # Remembers the last 4 exchanges to keep logic sharp

    print("\n" + "="*45)
    print("      LUCKY 38 MAINFRAME: MEMORY LINKS ONLINE")
    print("="*45 + "\n")

    while True:
        u = input("Courier: ")
        if u.lower() in ["exit", "quit"]: 
            print("\nConnection terminated. House always wins.")
            break
        
        # Build the history string to inject into the prompt
        history_str = ""
        for (old_q, old_a) in history:
            history_str += (
                f"<|start_header_id|>user<|end_header_id|>\n\n{old_q}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n{old_a}<|eot_id|>"
            )

        # Full prompt with System, History, and Current Question
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|>"
            f"{history_str}"
            f"<|start_header_id|>user<|end_header_id|>\n\n{u}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        
        inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")

        # --- 4. GENERATION ---
        outputs = model.generate(
            **inputs, 
            max_new_tokens = 150,           # Slightly more room for complex answers
            temperature = 0.3,              # Dropped to 0.3 to keep him from hallucinating
            top_p = 0.9,
            repetition_penalty = 1.3,     
            eos_token_id = terminators, 
            do_sample = True,
            use_cache = True,
        )
        
        full_text = tokenizer.batch_decode(outputs)[0]
        resp = full_text.split("assistant<|end_header_id|>\n\n")[-1]
        
        # --- 5. CLEANING ---
        resp = re.sub(r'<\|.*?\|>', '', resp).strip()
        
        # Kill roleplay leaks immediately
        stop_patterns = ["<|start_header_id|>", "user", "Courier:", "User:", "Assistant:", "assistant", "Nassistant"]
        for pattern in stop_patterns:
            if pattern in resp:
                resp = resp.split(pattern)[0].strip()

        # Remove gibberish and long strings
        resp = re.sub(r'[a-zA-Z]{15,}', '', resp)
        resp = resp.replace("igator", "").strip()

        # Ensure he doesn't end mid-sentence
        if " " in resp:
            last_punc = max(resp.rfind("."), resp.rfind("!"), resp.rfind("?"))
            if last_punc != -1:
                resp = resp[:last_punc + 1]

        resp = resp.encode("ascii", "ignore").decode().strip()

        if not resp:
            resp = "My calculations suggest this topic is a superfluous distraction."

        print(f"\nMr. House: {resp}\n")

        # Update the memory buffer
        history.append((u, resp))
        if len(history) > MAX_HISTORY:
            history.pop(0)

if __name__ == "__main__":
    chat()