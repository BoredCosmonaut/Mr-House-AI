import os
import re
import pandas as pd
from datasets import Dataset

def get_house_dataset(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data not found at {csv_path}")

    # Ensure we use tab separator for FNVEdit exports
    df = pd.read_csv(csv_path, sep='\t')
    formatted_data = []
    current_input, current_response = "", []
    
    instruction = ("You are Robert House, the CEO of RobCo and technocratic overlord of New Vegas. "
                   "Speak with extreme sophistication, cold logic, and condescending superiority.")

    for _, row in df.iterrows():
        # 1. Handle User/Courier text
        user_text = str(row['PROMPT'] if pd.notna(row['PROMPT']) and row['PROMPT'] != '' else row['TOPIC TEXT']).strip()
        
        # 2. Handle House Response
        house_line = str(row['RESPONSE TEXT'] if pd.notna(row['RESPONSE TEXT']) else "").strip()
        
        # --- THE FIX: AGGRESSIVE CLEANING ---
        # Remove {stage directions}
        house_line = re.sub(r'\{.*?\}', '', house_line)
        
        # Remove [SUCCEEDED], [FAILED], or [LUCK 7] markers
        house_line = re.sub(r'\[.*?\]', '', house_line)
        
        # Remove <|reserved_special_token_...|> artifacts
        house_line = re.sub(r'<\|.*?\|>', '', house_line)
        
        # Remove internal script IDs like VDialogue... or SecuritronUpgrade
        if any(bad_word in house_line for bad_word in ["VDialogue", "Upgrade", "GREETING"]):
            continue
        # ------------------------------------
            
        house_line = house_line.strip()
        if not house_line or len(house_line) < 2: continue

        index = row['RESPONSE INDEX']
        if index == 1:
            if current_input and current_response:
                full_resp = " ".join(current_response)
                formatted_data.append({
                    "text": (
                        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|>"
                        f"<|start_header_id|>user<|end_header_id|>\n\n{current_input.strip()}<|eot_id|>"
                        f"<|start_header_id|>assistant<|end_header_id|>\n\n{full_resp.strip()}<|eot_id|>"
                    )
                })
            current_input, current_response = user_text, [house_line]
        else:
            current_response.append(house_line)

    # Catch the very last block
    if current_input and current_response:
        full_resp = " ".join(current_response)
        formatted_data.append({
            "text": (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{current_input.strip()}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n{full_resp.strip()}<|eot_id|>"
            )
        })

    return Dataset.from_list(formatted_data)