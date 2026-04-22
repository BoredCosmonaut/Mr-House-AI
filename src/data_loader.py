import os
import re
import pandas as pd
import hashlib
from datasets import Dataset

def generate_house_opening(seed_text):
    """Generates a consistent but varied House-style opening based on a hash of the input."""
    openers = ["From a", "If one considers the", "Evaluating the", "Given the"]
    subjects = ["logistical", "mathematical", "industrial", "strategic", "logarithmic", "statistical"]
    perspectives = ["standpoint", "perspective", "angle", "necessity", "framework"]
    
    # Use a hash of the user input so that the model learns a stable relationship 
    # between prompts and character-appropriate openings.
    h = int(hashlib.md5(seed_text.encode()).hexdigest(), 16)
    
    # 20% chance of a blunt/direct opening (matching the chat.py logic)
    if h % 5 == 0:
        blunt_openings = [
            "To be perfectly blunt, ", 
            "Let us be clear: ", 
            "I find the query... interesting. ",
            "It is a simple matter of calculation: "
        ]
        return blunt_openings[h % len(blunt_openings)]
    
    op = openers[h % len(openers)]
    sub = subjects[(h // len(openers)) % len(subjects)]
    per = perspectives[(h // (len(openers) * len(subjects))) % len(perspectives)]
    
    return f"{op} {sub} {per}, "

def get_house_dataset(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data not found at {csv_path}")

    try:
        with open("persona.txt", "r", encoding="utf-8") as f:
            instruction = f.read().strip()
    except FileNotFoundError:
        instruction = "You are Robert House, the CEO of RobCo. Speak with cold logic and arrogance."

    df = pd.read_csv(csv_path, sep='\t', encoding='utf-8')
    formatted_data = []
    current_input, current_response = "", []

    for _, row in df.iterrows():
        # 1. Courier Text
        user_text = str(row['PROMPT'] if pd.notna(row['PROMPT']) and row['PROMPT'] != '' else row['TOPIC TEXT']).strip()
        
        # 2. House Response cleaning
        house_line = str(row['RESPONSE TEXT'] if pd.notna(row['RESPONSE TEXT']) else "").strip()
        
        # --- THE CLEANER ---
        house_line = re.sub(r'\{.*?\}', '', house_line) 
        house_line = re.sub(r'\[.*?\]', '', house_line) 
        house_line = re.sub(r'<\|.*?\|>', '', house_line) 
        
        bad_patterns = [r'VDialogue\w*', r'SecuritronUpgrade\w*', r'GREETING', r'Salutations']
        for pattern in bad_patterns:
            house_line = re.sub(pattern, '', house_line, flags=re.IGNORECASE)
        
        house_line = house_line.encode("ascii", "ignore").decode().strip()
        
        if not house_line or len(house_line) < 3: 
            continue

        index = row['RESPONSE INDEX']
        if index == 1:
            if current_input and current_response:
                # LIMITER: First 2 sentences
                full_resp = " ".join(current_response[:2]).strip()
                
                # --- ADD DYNAMIC OPENING ---
                nudge = generate_house_opening(current_input)
                house_style_resp = nudge + full_resp
                
                formatted_data.append({
                    "text": (
                        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|>"
                        f"<|start_header_id|>user<|end_header_id|>\n\n{current_input}<|eot_id|>"
                        f"<|start_header_id|>assistant<|end_header_id|>\n\n{house_style_resp}<|eot_id|>"
                    )
                })
            current_input, current_response = user_text, [house_line]
        else:
            current_response.append(house_line)

    return Dataset.from_list(formatted_data)