import torch
import re
from unsloth import FastLanguageModel

def chat():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "house_lora_final",
        max_seq_length = 2048,
        dtype = torch.bfloat16,
        load_in_4bit = False,
    )
    FastLanguageModel.for_inference(model)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    ]
    
    # SYSTEM PROMPT: Now includes 'Temporal Grounding'
    def load_persona(file_path="persona.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    # Then use it like this:
    instruction = load_persona()

    while True:
        u = input("Courier: ")
        if u.lower() in ["exit", "quit"]: break
        
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{u}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")

        outputs = model.generate(
            **inputs, 
            max_new_tokens = 150,    # Kept concise to prevent rambling
            temperature = 0.1,       # EXTREMELY LOW: Prevents "Fan-Fiction" drift
            top_p = 0.7,
            repetition_penalty = 1.3,
            eos_token_id = terminators, 
            do_sample = True,
            stop_strings = ["<|reserved", "user", "assistant"],
            tokenizer = tokenizer
            max_length = None,           # Clear the conflicting default
            use_cache = True             # Speeds up generation
        )
        
        full_text = tokenizer.batch_decode(outputs)[0]
        resp = full_text.split("assistant<|end_header_id|>\n\n")[-1]
        
        # --- THE CLEANER ---
        resp = re.sub(r'<\|.*?\|>', '', resp)
        resp = resp.replace("assistant", "").replace("user", "").strip()
        
        # --- HALLUCINATION CHECK ---
        # If he starts talking about Kimball/Caesar and you DIDN'T, 
        # he's likely hallucinating game scripts. 
        if "Kimball" in resp and "Kimball" not in u:
            # Subtle nudge back to reality
            resp = resp.split(".")[0] + ". But let us focus on your current query."

        print(f"\nMr. House: {resp}\n")

if __name__ == "__main__":
    chat()