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

    # 1. THE KILL SWITCH: If the model says these, it stops instantly.
    # This prevents the "Reserved Token" stuttering from filling your screen.
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    ]
    
    stop_words = ["<|reserved", "user", "assistant", "GREETING", "VDialogue"]

    # STAMP: Ensure this matches your training script exactly
    instruction = (
        "You are Robert House, the CEO of RobCo and technocratic overlord of New Vegas. "
        "Maintain extreme sophistication and condescending superiority. "
        "Do not use game script markers, internal tags, or system metadata. "
        "Evaluate all topics based on logical merit or industrial efficiency."
    )   

    while True:
        u = input("Courier: ")
        if u.lower() in ["exit", "quit"]: break
        
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{u}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")

        outputs = model.generate(
            **inputs, 
            max_new_tokens = 256,
            temperature = 0.2,       # DROPPED: Makes him more predictable/stable
            top_p = 0.9,
            repetition_penalty = 1.2,
            eos_token_id = terminators, 
            pad_token_id = tokenizer.eos_token_id,
            do_sample = True,
            # NEW: Hardware-level stop
            stop_strings = stop_words, 
            tokenizer = tokenizer
        )
        
        full_text = tokenizer.batch_decode(outputs)[0]
        resp = full_text.split("assistant<|end_header_id|>\n\n")[-1]
        
        # --- FINAL SURGICAL SCRUB ---
        # Even if the model generates it, we don't let the user see it.
        resp = re.sub(r'<\|.*?\|>', '', resp)
        resp = re.sub(r'VDialogue\w+', '', resp)
        resp = resp.replace("assistant", "").replace("user", "").strip()
        
        # Remove the stuttering ЎыџN symbols if they survive
        resp = resp.encode("ascii", "ignore").decode()
        
        print(f"\nMr. House: {resp}\n")

if __name__ == "__main__":
    chat()