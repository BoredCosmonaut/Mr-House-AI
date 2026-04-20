import torch
import re
from unsloth import FastLanguageModel

def chat():
    # Load Persona
    with open("persona.txt", "r") as f:
        instruction = f.read().strip()

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


    while True:
        u = input("Courier: ")
        if u.lower() in ["exit", "quit"]: break
        
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{u}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")

        outputs = model.generate(
            **inputs, 
            max_new_tokens = 120,
            temperature = 0.2, # Super low temp to keep him on track
            eos_token_id = terminators, 
            pad_token_id = tokenizer.eos_token_id,
            repetition_penalty = 1.3, # Higher penalty to stop the "VDialogue" loops
            do_sample = True
        )
        
        full_text = tokenizer.batch_decode(outputs)[0]
        resp = full_text.split("assistant<|end_header_id|>\n\n")[-1].replace("<|eot_id|>", "").strip()

        resp = re.sub(r'VDialogue\w+', '', resp)
        resp = re.sub(r'<\|.*?\|>', '', resp)
        resp = resp.replace("GREETING", "").strip()
        
        print(f"\nMr. House: {resp}\n")

if __name__ == "__main__":
    chat()