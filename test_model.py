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
    # Load the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "house_lora_final",
        max_seq_length = 2048,
        dtype = torch.bfloat16,
        load_in_4bit = False,
    )
    FastLanguageModel.for_inference(model)

    # Set up termination tokens
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        tokenizer.encode("\n", add_special_tokens=False)[-1],
    ]
    
    instruction = load_persona()

    print("\n--- Lucky 38 Mainframe Online ---")
    print("Mannerism Protocol: Organic (No Nudges)\n")

    while True:
        u = input("Courier: ")
        if u.lower() in ["exit", "quit"]: break
        
        # NUDGES REMOVED: The assistant block is now empty to let the model decide the start
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{u}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        
        inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")

        outputs = model.generate(
            **inputs, 
            max_new_tokens = 90,           
            temperature = 0.6,             # Slightly higher for more organic vocabulary
            top_p = 0.9,
            repetition_penalty = 1.2,     
            eos_token_id = terminators, 
            do_sample = True,
            use_cache = True,
        )
        
        full_text = tokenizer.batch_decode(outputs)[0]
        # Split exactly at the assistant header
        resp = full_text.split("assistant<|end_header_id|>\n\n")[-1]
        
        # --- CLEANING ---
        # Remove leftover technical tags
        resp = re.sub(r'<\|.*?\|>', '', resp)
        resp = resp.replace("assistant", "").strip()
        
        # Hard-Cut Lore Filter (Still needed to catch script-playback)
        lore_triggers = ["Benny", "Benjamin", "Platinum Chip", "Kimball", "GREETING", "Salutations", "overposting"]
        for trigger in lore_triggers:
            if trigger in resp and trigger not in u:
                resp = resp.split(trigger)[0].strip()

        # Remove gibberish and code leaks
        resp = re.sub(r'[a-zA-Z]{15,}', '', resp) 
        resp = resp.split("();")[0].split("//")[0].strip() 

        # Final Polish
        resp = resp.encode("ascii", "ignore").decode()
        if resp and resp[-1] not in [".", "!", "?"]: resp += "."
        # Ensure the response doesn't end on a dangling word like "the" or "and"
        if " " in resp:
            last_punctuation = max(resp.rfind("."), resp.rfind("!"), resp.rfind("?"))
            if last_punctuation != -1:
                resp = resp[:last_punctuation + 1]
                
        print(f"\nMr. House: {resp}\n")

if __name__ == "__main__":
    chat()