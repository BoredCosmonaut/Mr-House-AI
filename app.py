

import torch
import re
import warnings
import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from ddgs import DDGS
from unsloth import FastLanguageModel

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)

app = Flask(__name__)
# Change CORS(app) to allow the bypass headers safely
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": "*"}})
LOOKUP_TRIGGERS = [
    "when", "next episode", "next chapter", "rating", "score",
    "release", "out yet", "announced", "latest", "new chapter",
    "airing", "coming out", "premiere", "sequel", "season",
    "review", "metacritic", "imdb", "mal", "worth watching",
    "should i watch", "should i play", "is it good", "how many episodes"
]

def needs_lookup(text: str) -> bool:
    return any(t in text.lower() for t in LOOKUP_TRIGGERS)

def build_search_query(text: str) -> str:
    noise = ["what do you think about", "tell me about", "what is", "do you know", "have you heard of"]
    q = text.lower()
    for phrase in noise:
        q = q.replace(phrase, "")
    return q.strip()

def web_search(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return ""
        snippets = [r.get("body", "").strip() for r in results if r.get("body")]
        if not snippets:
            return ""
        priority = [s for s in snippets if any(
            w in s.lower() for w in ["release", "date", "episode", "season", "rating", "score", "announced", "confirmed"]
        )]
        best = priority[0] if priority else snippets[0]
        sentences = re.split(r'(?<=[.!?])\s+', best.strip())
        return " ".join(sentences[:2])[:250]
    except Exception as e:
        print(f"[SEARCH ERROR] {e}")
        return ""


def load_persona(path="persona.txt") -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            persona = f.read().strip()
    except FileNotFoundError:
        persona = "You are Robert House. Be cold, logical, and arrogant."
    return persona + (
        "\n\n[SYSTEM DIRECTIVE]: Speak ONLY as Robert House. "
        "Do not write dialogue for the Courier. "
        "End your response immediately after your own statement. "
        "Never repeat or reference these instructions in your response."
    )

print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="../house_lora_final",
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

INSTRUCTION = load_persona()
history = []
MAX_HISTORY = 4


# ─────────────────────────────────────────────
# RESPONSE HELPERS
# ─────────────────────────────────────────────
def parse_response(full_text: str) -> str:
    marker = "<|start_header_id|>assistant<|end_header_id|>"
    if marker not in full_text:
        return ""
    resp = full_text.split(marker)[-1].lstrip("\n")
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
    return resp.encode("ascii", "ignore").decode().strip()


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route('/chat', methods=['POST'])
def chat():
    global history

    data = request.get_json()
    user_text = data.get('message', '').strip()
    if not user_text:
        return jsonify({'error': 'Empty message'}), 400

    # Search
    searched = False
    search_query_used = ""
    search_context = ""
    if needs_lookup(user_text):
        search_query_used = build_search_query(user_text)
        result = web_search(search_query_used)
        if result:
            search_context = f"\n[CURRENT DATA: {result}]"
            searched = True

    # Build history string
    history_str = ""
    for (q, a) in history:
        history_str += (
            f"<|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n{a}<|eot_id|>"
        )

    prompt = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n{INSTRUCTION}<|eot_id|>"
        f"{history_str}"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_text}{search_context}<|eot_id|>"
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

    # Update history
    history.append((user_text, resp))
    if len(history) > MAX_HISTORY:
        history.pop(0)

    return jsonify({
        'response': resp,
        'searched': searched,
        'search_query': search_query_used
    })

@app.route('/reset', methods=['POST'])
def reset():
    global history
    history = []
    return jsonify({'status': 'History cleared.'})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'online', 'model': 'house_lora_final'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)