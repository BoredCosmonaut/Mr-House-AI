import torch
from unsloth import FastLanguageModel

def chat_with_house():
    model_path = "house_lora_final"
    max_seq_length = 2048
    dtype = torch.bfloat16 # A100 optimized
    load_in_4bit = False 

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model) 

    # The same persona instruction used in training
    instruction = (
        "You are Robert House, the CEO of RobCo and technocratic overlord of New Vegas. "
        "Speak with extreme sophistication, cold logic, and a condescending sense of superiority."
    )

    print("\nConnection Established\n")

    while True:
        user_input = input("Courier: What do you think of the Brotherhood of Steel?")
        if user_input.lower() in ["exit", "quit", "goodbye"]:
            print("Mr. House: A pity. I had expected more from you. Don't let the door hit you on your way out of the Lucky 38.")
            break

        # Format the Llama-3 prompt
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
        
        # Generation settings
        outputs = model.generate(
            **inputs, 
            max_new_tokens = 256,
            temperature = 0.7, # Adds a bit of variety to his responses
            use_cache = True
        )
        
        # Decode only the new tokens (the response)
        response = tokenizer.batch_decode(outputs)
        response_text = response[0].split("assistant<|end_header_id|>\n\n")[-1].replace("<|eot_id|>", "").strip()
        
        print(f"\nMr. House: {response_text}\n")

if __name__ == "__main__":
    chat_with_house()