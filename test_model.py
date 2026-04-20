import torch
from unsloth import FastLanguageModel

def chat():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "house_lora_final",
        max_seq_length = 2048,
        dtype = torch.bfloat16,
        load_in_4bit = False,
    )
    FastLanguageModel.for_inference(model)

    # 1. THE TERMINATOR LIST: Stops the model from running on
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        tokenizer.convert_tokens_to_ids("<|start_header_id|>") # Stops it from faking a User reply
    ]

    instruction = (
    "You are Robert House, the CEO of RobCo and technocratic overlord of New Vegas. "
    "Maintain extreme sophistication and condescending superiority. "
    "When discussing modern topics like anime, digital media, or technology, "
    "evaluate them based on their logical merit, industrial efficiency, or "
    "as curiosities of a pre-war civilization you find either fascinating or beneath you."
    )   

    while True:
        u = input("Courier: ")
        if u.lower() in ["exit", "quit"]: break
        
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{u}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")

        # 2. INCREASED TOKEN LIMIT & STOP LOGIC
        outputs = model.generate(
            **inputs, 
            max_new_tokens = 256,    # Raised from 120 so he doesn't cut off
            temperature = 0.3,       # Keeps him logical
            top_p = 0.9,
            repetition_penalty = 1.2,
            eos_token_id = terminators, 
            pad_token_id = tokenizer.eos_token_id,
            do_sample = True
        )
        
        # 3. EXTRACTION
        full_text = tokenizer.batch_decode(outputs)[0]
        # Grab only what comes after the last assistant header
        resp = full_text.split("assistant<|end_header_id|>\n\n")[-1]
        # Clean up any artifacts
        resp = resp.replace("<|eot_id|>", "").replace("<|begin_of_text|>", "").strip()
        
        print(f"\nMr. House: {resp}\n")

if __name__ == "__main__":
    chat()