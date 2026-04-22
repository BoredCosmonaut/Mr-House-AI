import os
import re
import pandas as pd
from datasets import Dataset

def get_house_dataset(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data not found at {csv_path}")

    # Explicitly use encoding to avoid weird byte artifacts
    df = pd.read_csv(csv_path, sep='\t', encoding='utf-8')
    formatted_data = []
    current_input, current_response = "", []
    
    # STRICTOR SYSTEM PROMPT: Explicitly forbids technical artifacts
    instruction = (
        "You are Robert House, the CEO of RobCo and technocratic overlord of New Vegas. "
        "Speak with extreme sophistication, cold logic, and condescending superiority. "
        "Do not use game script markers, internal tags, or system metadata."
    )

    for _, row in df.iterrows():
        # 1. Courier Text
        user_text = str(row['PROMPT'] if pd.notna(row['PROMPT']) and row['PROMPT'] != '' else row['TOPIC TEXT']).strip()
        
        # 2. House Response
        house_line = str(row['RESPONSE TEXT'] if pd.notna(row['RESPONSE TEXT']) else "").strip()
        
        # --- THE NUCLEAR SCRUB ---
        # 1. Remove anything inside brackets, braces, or pipes (The source of your leaks)
        house_line = re.sub(r'\{.*?\}', '', house_line) # {Stage Directions}
        house_line = re.sub(r'\[.*?\]', '', house_line) # [Skill Checks]
        house_line = re.sub(r'<\|.*?\|>', '', house_line) # <|Reserved Tokens|>
        
        # 2. Remove Game Engine IDs and meta-words
        # We use IGNORECASE to catch 'Greeting', 'GREETING', etc.
        bad_patterns = [
            r'VDialogue\w*', r'SecuritronUpgrade\w*', r'GREETING', 
            r'inquiry', r'greeting', r'assistant', r'user', r'Prompt:', r'Response:'
        ]
        for pattern in bad_patterns:
            house_line = re.sub(pattern, '', house_line, flags=re.IGNORECASE)
        
        # 3. Clean up punctuation artifacts (like the 'ЎыџN' you saw)
        house_line = house_line.encode("ascii", "ignore").decode() 
        
        house_line = house_line.strip()
        
        if not house_line or len(house_line) < 3: 
            continue
        # -------------------------

        # LOGIC FIX: Check if this row actually belongs to the same topic
        # FNVEdit exports can be messy; INFO ID is a better anchor than Index 1
        info_id = row.get('INFO', 'Unknown')

        index = row['RESPONSE INDEX']
        if index == 1:
            if current_input and current_response:
                # LIMITER: Only take the first 3 sentences of a House response
                # This prevents the model from learning "Rambling" behavior.
                full_resp = " ".join(current_response[:3]) 
                
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