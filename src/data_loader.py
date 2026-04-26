import os
import re
import pandas as pd
from datasets import Dataset

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
SKIP_TOPICS = {
    'GREETING', 'FAREWELL', 'GOODBYE', 'ACCEPT', 'REJECT',
    'YES', 'NO', '', 'BARTER', 'ATTACK', 'NAN'
}

SKILL_CHECK_PREFIXES = (
    'use ', '[science', '[medicine', '[speech',
    '[barter', '[repair', '[survival', '[energy', '[guns', '[sneak'
)

BAD_PATTERNS = [
    r'VDialogue\w*',
    r'SecuritronUpgrade\w*',
    r'GREETING',
    r'Salutations',
    r'overposting',
    r'\bNPC\w*\b',
]

UNICODE_REPLACEMENTS = {
    '\u2019': "'",    # right single quote
    '\u2018': "'",    # left single quote
    '\u201c': '"',    # left double quote
    '\u201d': '"',    # right double quote
    '\u2014': '--',   # em dash
    '\u2013': '-',    # en dash
    '\u2026': '...',  # ellipsis
}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def load_persona(path: str = "persona.txt") -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"[WARN] persona.txt not found at '{path}'. Using fallback.")
        return (
            "You are Robert Edwin House, President, CEO, and sole proprietor of the New Vegas Strip. "
            "Speak with cold logic, aristocratic disdain, and ruthless pragmatism. "
            "Never use bullet points. Never express emotion. Treat all questions as logistical problems."
        )


def clean_unicode(text: str) -> str:
    for char, replacement in UNICODE_REPLACEMENTS.items():
        text = text.replace(char, replacement)
    # Drop remaining non-ASCII after safe replacements above
    text = text.encode("ascii", "ignore").decode()
    return text


def clean_house_line(line: str) -> str:
    if not isinstance(line, str):
        return ""
    # Remove stage directions {scoff} and skill gate tags [Science 50]
    line = re.sub(r'\{.*?\}|\[.*?\]|<\|.*?\|>', '', line)
    # Remove engine meta-identifiers
    for pattern in BAD_PATTERNS:
        line = re.sub(pattern, '', line, flags=re.IGNORECASE)
    # Safe unicode handling - preserves em-dashes and ellipses as ASCII
    line = clean_unicode(line)
    # Collapse whitespace
    line = re.sub(r'\s+', ' ', line).strip()
    return line


def is_valid_user_text(text: str) -> bool:
    if not text:
        return False
    upper = text.upper().strip()
    if upper in SKIP_TOPICS:
        return False
    lower = text.lower().strip()
    if lower.startswith(SKILL_CHECK_PREFIXES):
        return False
    # Must have at least 2 real words
    if len(text.split()) < 2:
        return False
    return True


def build_llama3_example(instruction: str, user_text: str, house_response: str) -> str:
    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_text}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{house_response}<|eot_id|>"
    )


# ─────────────────────────────────────────────
# MAIN DATASET BUILDER
# ─────────────────────────────────────────────
def get_house_dataset(csv_path: str, persona_path: str = "persona.txt") -> Dataset:
    """
    Load and format Mr. House dialogue CSV into a Llama 3 instruction dataset.

    Key fixes vs original:
      - Groups by TOPIC TEXT (TOPIC column is entirely NaN in this game export)
      - Keeps FULL response -- no [:2] truncation that was losing character voice
      - Skips GREETING, skill check prompts, and single-word topics
      - Safe unicode handling that preserves em-dashes/ellipses before ASCII strip
      - Prefers PROMPT column over TOPIC TEXT when it exists (skill check branches)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    instruction = load_persona(persona_path)

    df = pd.read_csv(csv_path, sep='\t', encoding='utf-8')
    df.columns = df.columns.str.strip()

    for col in ['PROMPT', 'TOPIC TEXT', 'RESPONSE TEXT']:
        if col in df.columns:
            df[col] = df[col].fillna('')

    formatted_data = []
    skipped = 0

    # NOTE: TOPIC column is all NaN in this game export, so we group by TOPIC TEXT.
    # This correctly bundles the multi-line responses (index 1,2,3...) for each
    # unique Courier dialogue option into a single training example.
    for topic_text, group in df.groupby('TOPIC TEXT', sort=False):
        group = group.sort_values('RESPONSE INDEX')
        first_row = group.iloc[0]

        # Prefer explicit PROMPT (filled for skill-check branches) over TOPIC TEXT
        prompt_col = str(first_row.get('PROMPT', '')).strip()
        user_text = prompt_col if prompt_col else str(topic_text).strip()

        if not is_valid_user_text(user_text):
            skipped += 1
            continue

        # Clean and join ALL House response lines -- no truncation
        house_lines = []
        for _, row in group.iterrows():
            line = clean_house_line(str(row.get('RESPONSE TEXT', '')))
            if line and len(line) >= 5:
                house_lines.append(line)

        if not house_lines:
            skipped += 1
            continue

        full_response = ' '.join(house_lines).strip()

        # Guard against garbage rows
        if len(full_response) < 20 or len(full_response) > 2000:
            skipped += 1
            continue

        example = build_llama3_example(instruction, user_text, full_response)
        formatted_data.append({"text": example})

    print(f"[INFO] Built {len(formatted_data)} training examples. Skipped {skipped} invalid entries.")

    if len(formatted_data) == 0:
        raise ValueError("No valid training examples found. Check your CSV path and column names.")

    return Dataset.from_list(formatted_data)


# ─────────────────────────────────────────────
# QUICK SANITY CHECK
# Run: python data_loader.py data/house_v2_clean.csv
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else "data/house_v2_clean.csv"
    ds = get_house_dataset(csv)
    print(f"\nDataset size: {len(ds)}")
    print("\n--- SAMPLE EXAMPLE ---\n")
    print(ds[0]['text'])