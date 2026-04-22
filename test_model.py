import torch
import re
import warnings
from unsloth import FastLanguageModel

# Suppress the deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def load_persona(file_path="persona.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "You are Robert House, the technocratic autocrat of New Vegas."

def chat():
    # Load the model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "house_lora_final",
        max_seq_length = 2048,
        dtype = torch.bfloat16,
        load_in_4bit = False,
    )
    FastLanguageModel.for_inference(model)

    # Set up stopping criteria
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        tokenizer.convert_tokens_to_ids("<|start_header_id|>"),
        tokenizer.encode("\n", add_special_tokens=False)[-1] # Stop at new lines
    ]
    
    instruction = load_persona()

    print("\n--- Lucky 38 Mainframe Online ---")
    print("Type 'exit' to disconnect.\n")

    while True:
        u = input("Courier: ")
        if u.lower() in ["exit", "quit"]: break
        
        # NUDGE: We force the assistant to start with a logical phrase to break the script loop
        nudge = "From a purely logistical standpoint, "
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{u}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n{nudge}"
        )
        
        inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")

        outputs = model.generate(
            **inputs, 
            max_new_tokens = 120,        # Concise responses are more 'House'
            temperature = 0.1,           # Low temp for high logic
            top_p = 0.8,
            repetition_penalty = 1.5,    # Hard stop for "......." and script loops
            eos_token_id = terminators, 
            do_sample = True,
            max_length = None,
            use_cache = True,
        )
        
        full_text = tokenizer.batch_decode(outputs)[0]
        
        # Extract the response after our nudge
        resp = full_text.split("assistant<|end_header_id|>\n\n")[-1]
        
        # --- THE SURGICAL CLEANER ---
        # 1. Strip special tokens and tech artifacts
        resp = re.sub(r'<\|.*?\|>', '', resp)
        resp = resp.replace("assistant", "").replace("user", "").strip()
        
        # 2. Kill "Lore Leaks" (If he mentions these without you asking, we cut the sentence)
        lore_triggers = ["Benny", "Hoover Dam", "Platinum Chip", "Kimball", "Salutations", "GREETING"]
        for trigger in lore_triggers:
            if trigger in resp and trigger not in u:
                # Cut the response before the hallucinated lore starts
                resp = resp.split(trigger)[0].strip()
        
        # 3. Prevent the Stutter Meltdown
        if "...." in resp:
            resp = resp.split("....")[0].strip()
        
        # 4. ASCII Polish (Removes ЎыџN symbols)
        resp = resp.encode("ascii", "ignore").decode()
        
        # 5. Ensure it ends cleanly
        if resp and resp[-1] not in [".", "!", "?"]:
            resp += "."

        print(f"\nMr. House: {resp}\n")

if __name__ == "__main__":
    chat()