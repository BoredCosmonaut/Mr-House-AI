import torch
import re
import warnings
from unsloth import FastLanguageModel

warnings.filterwarnings("ignore", category=FutureWarning)

def load_persona(file_path="persona.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "You are Robert House, CEO of RobCo."

def chat():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "house_lora_final",
        max_seq_length = 2048,
        dtype = torch.bfloat16,
        load_in_4bit = False,
    )
    FastLanguageModel.for_inference(model)

    # --- CHANGE 1: ADD NEWLINE TERMINATOR ---
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        tokenizer.convert_tokens_to_ids("<|start_header_id|>"),
        tokenizer.encode("\n", add_special_tokens=False)[-1] 
    ]
    
    instruction = load_persona()

    while True:
        u = input("Courier: ")
        if u.lower() in ["exit", "quit"]: break
        
        # This forces the model to start with a "Thinking" posture instead of a "Script" posture
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{u}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nTo be blunt, "
        inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")

        # --- CHANGE 2 & 3: TIGHTEN LIMITS ---
        outputs = model.generate(
            **inputs, 
            max_new_tokens = 120,        # Reduced to keep him concise
            temperature = 0.1,           # Dropped slightly for maximum logic
            top_p = 0.8,
            repetition_penalty = 1.5,    # Increased to kill the "........" stuttering
            eos_token_id = terminators, 
            do_sample = True,
            max_length = None,
            use_cache = True,
        )
        
        full_text = tokenizer.batch_decode(outputs)[0]
        resp = full_text.split("assistant<|end_header_id|>\n\n")[-1]
        
        # Clean technical artifacts
        resp = re.sub(r'<\|.*?\|>', '', resp)
        resp = resp.replace("assistant", "").replace("user", "").strip()
        
        # ASCII cleaning (removes the weird ЎыџN symbols)
        resp = resp.encode("ascii", "ignore").decode()

        # Final polish: stop at the first sign of a halluciantion-style repeat
        if "\n" in resp:
            resp = resp.split("\n")[0]

        print(f"\nMr. House: {resp}\n")

if __name__ == "__main__":
    chat()