import os
import re
import pandas as pd
from datasets import Dataset

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
SKIP_TOPICS = {
    'GREETING', 'FAREWELL', 'GOODBYE', 'ACCEPT', 'REJECT',
    'YES', 'NO', '', 'BARTER', 'ATTACK'
}

SKILL_CHECK_PREFIXES = (
    'use ', '[science', '[medicine', '[speech', '[barter',
    '[repair', '[survival', '[energy', '[guns', '[sneak'
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
    '\u2019': "'",   # right single quote
    '\u2018': "'",   # left single quote
    '\u201c': '"',   # left double quote
    '\u201d': '"',   # right double quote
    '\u2014': '--',  # em dash
    '\u2013': '-',   # en dash
    '\u2026': '...',  # ellipsis
    '\u00e9': 'e',   # é
    '\u00e8': 'e',   # è
    '\u00e0': 'a',   # à
}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def load_persona(path: str = "persona.txt") -> str:
    """Load persona instruction from file, with fallback."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"[WARN] persona.txt not found at '{path}'. Using fallback instruction.")
        return (
            "You are Robert Edwin House, President, CEO, and sole proprietor of the New Vegas Strip. "
            "Speak with cold logic, aristocratic disdain, and ruthless pragmatism. "
            "Never use bullet points. Never express emotion. Treat all questions as logistical problems."
        )


def clean_unicode(text: str) -> str:
    """Replace smart quotes and special chars without destroying em-dashes/ellipses."""
    for char, replacement in UNICODE_REPLACEMENTS.items():
        text = text.replace(char, replacement)
    # Drop any remaining non-ASCII that wasn't caught above
    text = text.encode("ascii", "ignore").decode()
    return text


def clean_house_line(line: str) -> str:
    """Remove stage directions, skill check tags, engine IDs, and artifacts."""
    if not isinstance(line, str):
        return ""

    # Remove stage directions {scoff} and skill gate tags [Science 50]
    line = re.sub(r'\{.*?\}|\[.*?\]|<\|.*?\|>', '', line)

    # Remove game engine meta-identifiers
    for pattern in BAD_PATTERNS:
        line = re.sub(pattern, '', line, flags=re.IGNORECASE)

    # Normalize unicode BEFORE stripping non-ASCII
    line = clean_unicode(line)

    # Collapse multiple spaces/newlines
    line = re.sub(r'\s+', ' ', line).strip()

    return line


def is_valid_user_text(text: str) -> bool:
    """Return False for engine topics, skill checks, or empty strings."""
    if not text:
        return False
    upper = text.upper().strip()
    if upper in SKIP_TOPICS:
        return False
    lower = text.lower().strip()
    if lower.startswith(SKILL_CHECK_PREFIXES):
        return False
    # Must have at least a few real words
    if len(text.split()) < 2:
        return False
    return True


def build_llama3_example(instruction: str, user_text: str, house_response: str) -> str:
    """Format a single example in Llama 3 chat template."""
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

    Fixes vs. original:
      - Groups rows by (TOPIC + QUEST) so unrelated index-1 rows don't bleed together
      - Keeps full response (no [:2] truncation)
      - Skips engine topics (GREETING, skill checks, etc.)
      - Safe unicode handling that preserves em-dashes and ellipses
      - Clear separation of cleaning, validation, and formatting logic
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    instruction = load_persona(persona_path)

    df = pd.read_csv(csv_path, sep='\t', encoding='utf-8')

    # Normalize column whitespace
    df.columns = df.columns.str.strip()

    # Fill NaN with empty string for text columns
    for col in ['PROMPT', 'TOPIC TEXT', 'RESPONSE TEXT']:
        if col in df.columns:
            df[col] = df[col].fillna('')

    # Build a stable conversation group key
    df['_conv_id'] = (
        df['TOPIC'].astype(str).str.strip() + '|||' +
        df['QUEST'].astype(str).str.strip()
    )

    formatted_data = []
    skipped = 0

    for conv_id, group in df.groupby('_conv_id', sort=False):
        group = group.sort_values('RESPONSE INDEX')

        first_row = group.iloc[0]

        # Prefer explicit PROMPT over TOPIC TEXT (TOPIC TEXT is Courier's menu label)
        prompt_col = str(first_row.get('PROMPT', '')).strip()
        topic_col = str(first_row.get('TOPIC TEXT', '')).strip()
        user_text = prompt_col if prompt_col else topic_col

        if not is_valid_user_text(user_text):
            skipped += 1
            continue

        # Clean and join all House response lines for this conversation group
        house_lines = []
        for _, row in group.iterrows():
            line = clean_house_line(str(row.get('RESPONSE TEXT', '')))
            if line and len(line) >= 5:
                house_lines.append(line)

        if not house_lines:
            skipped += 1
            continue

        full_response = ' '.join(house_lines).strip()

        # Safety: skip implausibly short or long responses
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
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    csv = sys.argv[1] if len(sys.argv) > 1 else "data/house_v2_clean.csv"
    ds = get_house_dataset(csv)

    print(f"\nDataset size: {len(ds)}")
    print("\n--- SAMPLE EXAMPLE ---\n")
    print(ds[0]['text'])