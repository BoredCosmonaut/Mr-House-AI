import os
import re
import pandas as pd
from datasets import Dataset

def get_house_dataset(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data not found at {csv_path}")

    # Load the Persona as the instruction for every training example
    # This forces the model to link the "House Voice" to these specific rules.
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
        house_line = re.sub(r'\{.*?\}', '', house_line) # Remove stage directions
        house_line = re.sub(r'\[.*?\]', '', house_line) # Remove skill checks
        house_line = re.sub(r'<\|.*?\|>', '', house_line) # Remove special tokens
        
        # Remove game engine IDs (IGNORECASE)
        bad_patterns = [r'VDialogue\w*', r'SecuritronUpgrade\w*', r'GREETING']
        for pattern in bad_patterns:
            house_line = re.sub(pattern, '', house_line, flags=re.IGNORECASE)
        
        # Strip non-ASCII (fixes the ЎыџN symbols)
        house_line = house_line.encode("ascii", "ignore").decode().strip()
        
        if not house_line or len(house_line) < 3: 
            continue

        index = row['RESPONSE INDEX']
        if index == 1:
            if current_input and current_response:
                # LIMITER: Only take the first 2 sentences. 
                # This stops him from learning to monologue.
                full_resp = " ".join(current_response[:2]) 
                formatted_data.append({
                    "text": (
                        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|>"
                        f"<|start_header_id|>user<|end_header_id|>\n\n{current_input}<|eot_id|>"
                        f"<|start_header_id|>assistant<|end_header_id|>\n\n{full_resp.strip()}<|eot_id|>"
                    )
                })
            current_input, current_response = user_text, [house_line]
        else:
            current_response.append(house_line)

    return Dataset.from_list(formatted_data)