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
        return "You are Robert House. Be cold, logical, and arrogant."

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
        tokenizer.encode("\n", add_special_tokens=False)[-1],
    ]
    
    instruction = load_persona()

    print("\n--- Lucky 38 Mainframe Online ---")

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
            max_new_tokens = 128,           # More room prevents abrupt cuts
            max_length = 2048,             # KILLS THE WARNING
            temperature = 0.4,             # LOWER temp prevents him from switching identities
            top_p = 0.9,
            repetition_penalty = 1.3,     # HIGHER penalty kills "igator" and loops
            eos_token_id = terminators, 
            do_sample = True,
            use_cache = True,
        )
        
        full_text = tokenizer.batch_decode(outputs)[0]
        resp = full_text.split("assistant<|end_header_id|>\n\n")[-1]
        
        # --- CLEANING ---
        resp = re.sub(r'<\|.*?\|>', '', resp).strip()
        
        # Identity Lock: If he starts talking like a technician/worker, cut it.
        worker_leaks = ["NCR", "headquarters", "program", "manual", "restrictions", "override", "working at"]
        for leak in worker_leaks:
            if leak in resp.lower() and leak not in u.lower():
                resp = resp.split(leak)[0].strip()

        # Artifact Cleaner
        resp = re.sub(r'igator|user|assistant|Courier', '', resp, flags=re.IGNORECASE)
        resp = re.sub(r'[a-zA-Z]{15,}', '', resp) 

        # Sentence Trimmer (The part you liked!)
        if " " in resp:
            last_punc = max(resp.rfind("."), resp.rfind("!"), resp.rfind("?"))
            if last_punc != -1:
                resp = resp[:last_punc + 1]

        resp = resp.encode("ascii", "ignore").decode().strip()

        if not resp or len(resp) < 5:
            resp = "The question is irrelevant. My plans for the future do not concern such trivialities."

        print(f"\nMr. House: {resp}\n")

if __name__ == "__main__":
    chat()