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

    # CRITICAL: Define stop tokens
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    instruction = "You are Robert House, CEO of RobCo and overlord of New Vegas. Speak with extreme sophistication and cold logic."

    while True:
        u = input("Courier: ")
        if u.lower() in ["exit", "quit"]: break
        
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{u}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")

        outputs = model.generate(
            **inputs, 
            max_new_tokens = 150,
            temperature = 0.4,       # Lower temp = more logic
            eos_token_id = terminators, 
            pad_token_id = tokenizer.eos_token_id,
            repetition_penalty = 1.2
        )
        
        resp = tokenizer.batch_decode(outputs)[0].split("assistant<|end_header_id|>\n\n")[-1].replace("<|eot_id|>", "").strip()
        print(f"\nMr. House: {resp}\n")

if __name__ == "__main__":
    chat()