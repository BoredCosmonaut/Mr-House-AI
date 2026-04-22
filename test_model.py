import torch
import re
import random
import warnings
from unsloth import FastLanguageModel

warnings.filterwarnings("ignore", category=FutureWarning)

def load_persona(file_path="persona.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "You are Robert House. Be cold, logical, and arrogant."

def get_random_nudge():
    """Generates a House-style opening to match the data_loader's variety."""
    openers = ["From a", "If one considers the", "Evaluating the", "Given the"]
    subjects = ["logistical", "mathematical", "industrial", "strategic", "logarithmic", "statistical"]
    perspectives = ["standpoint", "perspective", "angle", "necessity", "framework"]
    
    # 20% chance of a blunt or condescending opening
    if random.random() < 0.2:
        return random.choice([
            "To be perfectly blunt, ", 
            "Let us be clear: ", 
            "I find the query... interesting. ",
            "It is a simple matter of calculation: "
        ])
    
    return f"{random.choice(openers)} {random.choice(subjects)} {random.choice(perspectives)}, "

def chat():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "house_lora_final",
        max_seq_length = 2048,
        dtype = torch.bfloat16,
        load_in_4bit = False,
    )
    FastLanguageModel.for_inference(model)

    # Termination tokens to stop him from rambling into game scripts
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
        
        # This nudge selection now matches the logic used in your training data
        nudge = get_random_nudge()
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{u}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{nudge}"
        
        inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")

        outputs = model.generate(
            **inputs, 
            max_new_tokens = 90,           # Slightly more room for the varied openings
            temperature = 0.5,             # Higher temp allows his vocabulary to shine
            top_p = 0.9,
            repetition_penalty = 1.25,     # Prevents the "Minty Fresh" collapse
            eos_token_id = terminators, 
            do_sample = True,
            use_cache = True,
        )
        
        full_text = tokenizer.batch_decode(outputs)[0]
        resp = full_text.split("assistant<|end_header_id|>\n\n")[-1]
        
        # --- SURGICAL CLEANER ---
        resp = re.sub(r'<\|.*?\|>', '', resp)
        resp = resp.replace("assistant", "").strip()
        
        # Hard-Cut Lore Filter (Backgrounds the game plot)
        lore_triggers = ["Benny", "Benjamin", "Platinum Chip", "Kimball", "GREETING", "Salutations", "overposting"]
        for trigger in lore_triggers:
            if trigger in resp and trigger not in u:
                resp = resp.split(trigger)[0].strip()

        # Remove technical artifacts and code leaks
        resp = re.sub(r'[a-zA-Z]{12,}', '', resp) # Kills long gibberish
        resp = resp.split("();")[0].split("//")[0].strip() 

        # ASCII Polish and Punctuation Fix
        resp = resp.encode("ascii", "ignore").decode()
        if resp and resp[-1] not in [".", "!", "?"]: resp += "."

        print(f"\nMr. House: {resp}\n")

if __name__ == "__main__":
    chat()