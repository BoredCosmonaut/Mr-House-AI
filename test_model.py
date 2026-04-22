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

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    ]
    
    instruction = load_persona()

    while True:
        u = input("Courier: ")
        if u.lower() in ["exit", "quit"]: break
        
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{u}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")

        outputs = model.generate(
            **inputs, 
            max_new_tokens = 200, 
            temperature = 0.15,          # Slightly raised from 0.1 for better flow
            top_p = 0.8,                 # Gives him a slightly wider vocabulary
            repetition_penalty = 1.35,   # Stops the "But let us focus..." loops
            # NEW: Prevents him from repeating common words/phrases too often
            presence_penalty = 0.6,      
            eos_token_id = terminators, 
            do_sample = True,
            max_length = None,
            use_cache = True
        )
        
        full_text = tokenizer.batch_decode(outputs)[0]
        resp = full_text.split("assistant<|end_header_id|>\n\n")[-1]
        
        # Clean technical artifacts
        resp = re.sub(r'<\|.*?\|>', '', resp)
        resp = resp.replace("assistant", "").replace("user", "").strip()
        
        # ASCII cleaning to remove non-printable game characters (ЎыџN)
        resp = resp.encode("ascii", "ignore").decode()

        print(f"\nMr. House: {resp}\n")

if __name__ == "__main__":
    chat()