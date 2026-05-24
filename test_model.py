import torch
import re
import warnings
import os
import logging
from unsloth import FastLanguageModel
from ddgs import DDGS

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)

# ─────────────────────────────────────────────
# SEARCH CONFIG
# ─────────────────────────────────────────────
LOOKUP_TRIGGERS = [
    "when", "next episode", "next chapter", "rating", "score",
    "release", "out yet", "announced", "latest", "new chapter",
    "airing", "coming out", "premiere", "sequel", "season",
    "review", "metacritic", "imdb", "mal", "worth watching",
    "should i watch", "should i play", "is it good", "how many episodes"
]

def needs_lookup(text: str) -> bool:
    lower = text.lower()
    return any(trigger in lower for trigger in LOOKUP_TRIGGERS)

def web_search(query: str) -> str:
    """Search DuckDuckGo and return a brief summary of top results."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return ""
        # Combine top snippets into a brief context block
        snippets = [r.get("body", "") for r in results if r.get("body")]
        return " | ".join(snippets[:3])[:600]  # cap at 600 chars
    except Exception as e:
        print(f"  [SEARCH ERROR] {e}")
        return ""

def build_search_query(user_text: str) -> str:
    """Clean up the user message into a good search query."""
    # Strip filler phrases that confuse search
    noise = ["what do you think about", "tell me about", "what is", "do you know", "have you heard of"]
    query = user_text.lower()
    for phrase in noise:
        query = query.replace(phrase, "")
    return query.strip()


# ─────────────────────────────────────────────
# PERSONA
# ─────────────────────────────────────────────
def load_persona(file_path="persona.txt") -> str:
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


# ─────────────────────────────────────────────
# RESPONSE PARSING
# ─────────────────────────────────────────────
def parse_response(full_text: str) -> str:
    ASSISTANT_MARKER = "<|start_header_id|>assistant<|end_header_id|>"
    if ASSISTANT_MARKER not in full_text:
        return ""
    resp = full_text.split(ASSISTANT_MARKER)[-1]
    resp = resp.lstrip("\n")
    for stop in ["<|start_header_id|>", "<|eot_id|>", "<|end_of_text|>"]:
        if stop in resp:
            resp = resp.split(stop)[0]
    return resp.strip()


def clean_response(resp: str) -> str:
    if not resp:
        return ""
    resp = re.sub(r'<\|.*?\|>', '', resp)
    for pattern in ["Courier:", "User:", "user:", "Assistant:", "assistant:"]:
        if pattern in resp:
            resp = resp.split(pattern)[0].strip()
    resp = re.sub(r'[a-zA-Z]{20,}', '', resp)
    resp = re.sub(r'\n{3,}', '\n\n', resp)
    last_punc = max(resp.rfind('.'), resp.rfind('!'), resp.rfind('?'))
    if last_punc > len(resp) // 2:
        resp = resp[:last_punc + 1]
    resp = resp.encode("ascii", "ignore").decode().strip()
    return resp


# ─────────────────────────────────────────────
# MAIN CHAT LOOP
# ─────────────────────────────────────────────
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

        search_context = ""
        if needs_lookup(u):
            query = build_search_query(u)
            print(f"  [SEARCHING: {query}]")
            result = web_search(query)
            if result:
                search_context = f"\n\n[RETRIEVED DATA - use this to inform your response]: {result}"

        history_str = ""
        for (old_q, old_a) in history:
            history_str += (
                f"<|start_header_id|>user<|end_header_id|>\n\n{old_q}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n{old_a}<|eot_id|>"
            )

        turn_instruction = instruction + search_context

        prompt = (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n\n{turn_instruction}<|eot_id|>"
            f"{history_str}"
            f"<|start_header_id|>user<|end_header_id|>\n\n{u}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.5,
                top_p=0.9,
                repetition_penalty=1.2,
                eos_token_id=terminators,
                do_sample=True,
                use_cache=True,
            )

        full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
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