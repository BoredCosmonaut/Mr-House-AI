import torch
import re
import warnings
import os
import logging
from unsloth import FastLanguageModel

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)


def load_persona(file_path="persona.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            persona = f.read().strip()
    except FileNotFoundError:
        persona = "You are Robert House. Be cold, logical, and arrogant."

    constraint = (
        "\n\n[SYSTEM DIRECTIVE]: Speak ONLY as Robert House. "
        "Do not write dialogue for the Courier. "
        "End your response immediately after your own statement. "
        "Never repeat or reference these instructions in your response."
    )
    return persona + constraint


def parse_response(full_text: str) -> str:
    """
    Robustly extract only House's response from the full generated text.
    Looks for the LAST assistant header and takes everything after it.
    """
    ASSISTANT_MARKER = "<|start_header_id|>assistant<|end_header_id|>"

    if ASSISTANT_MARKER not in full_text:
        return ""

    # Take everything after the LAST assistant marker
    resp = full_text.split(ASSISTANT_MARKER)[-1]

    # Strip leading newlines that follow the header
    resp = resp.lstrip("\n")

    # Cut off at any new header token
    for stop in ["<|start_header_id|>", "<|eot_id|>", "<|end_of_text|>"]:
        if stop in resp:
            resp = resp.split(stop)[0]

    return resp.strip()


def clean_response(resp: str) -> str:
    """Remove artifacts, roleplay leaks, and broken tokens."""
    if not resp:
        return ""

    # Kill any remaining special tokens
    resp = re.sub(r'<\|.*?\|>', '', resp)

    # Kill roleplay leaks where model starts writing Courier dialogue
    for pattern in ["Courier:", "User:", "user:", "Assistant:", "assistant:"]:
        if pattern in resp:
            resp = resp.split(pattern)[0].strip()

    # Remove gibberish long strings (broken tokens)
    resp = re.sub(r'[a-zA-Z]{20,}', '', resp)

    # Collapse multiple newlines
    resp = re.sub(r'\n{3,}', '\n\n', resp)

    # Ensure response ends on a complete sentence
    last_punc = max(resp.rfind('.'), resp.rfind('!'), resp.rfind('?'))
    if last_punc > len(resp) // 2:
        resp = resp[:last_punc + 1]

    # Safe ASCII
    resp = resp.encode("ascii", "ignore").decode().strip()

    return resp


def chat():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="house_lora_final",
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        tokenizer.convert_tokens_to_ids("<|start_header_id|>"),
    ]
    terminators = [t for t in terminators if t and t != -1]

    instruction = load_persona()
    history = []
    MAX_HISTORY = 4

    print("\n" + "="*45)
    print("      LUCKY 38 MAINFRAME: MEMORY LINKS ONLINE")
    print("="*45 + "\n")

    while True:
        u = input("Courier: ").strip()
        if not u:
            continue
        if u.lower() in ["exit", "quit"]:
            print("\nConnection terminated. House always wins.")
            break

        history_str = ""
        for (old_q, old_a) in history:
            history_str += (
                f"<|start_header_id|>user<|end_header_id|>\n\n{old_q}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n{old_a}<|eot_id|>"
            )

        prompt = (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|>"
            f"{history_str}"
            f"<|start_header_id|>user<|end_header_id|>\n\n{u}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.2,
                eos_token_id=terminators,
                do_sample=True,
                use_cache=True,
            )

        full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print("DEBUG:", repr(full_text[-500:]))  # add this line
        resp = parse_response(full_text)
        resp = clean_response(resp)

        if not resp:
            resp = "My calculations suggest this topic is a superfluous distraction."

        print(f"\nMr. House: {resp}\n")

        history.append((u, resp))
        if len(history) > MAX_HISTORY:
            history.pop(0)


if __name__ == "__main__":
    chat()