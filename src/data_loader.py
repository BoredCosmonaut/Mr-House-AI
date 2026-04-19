import os
import re
import pandas as pd
from datasets import Dataset

def get_house_dataset(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data not found at {csv_path}")

    df = pd.read_csv(csv_path, sep='\t')
    formatted_data = []
    current_input, current_response = "", []
    
    instruction = ("You are Robert House, the CEO of RobCo and technocratic overlord of New Vegas. "
                   "Speak with extreme sophistication, cold logic, and condescending superiority.")

    for _, row in df.iterrows():
        # Handle User/Courier text
        user_text = str(row['PROMPT'] if pd.notna(row['PROMPT']) and row['PROMPT'] != '' else row['TOPIC TEXT']).strip()
        
        # Handle House Response
        house_line = str(row['RESPONSE TEXT'] if pd.notna(row['RESPONSE TEXT']) else "").strip()
        
        # CLEANING: Remove {stage directions} and internal script IDs
        house_line = re.sub(r'\{.*?\}', '', house_line)
        if "VDialogue" in house_line or "Upgrade" in house_line: # Filter out leaked column names
            continue
            
        house_line = house_line.strip()
        if not house_line or len(house_line) < 2: continue

        index = row['RESPONSE INDEX']
        if index == 1:
            if current_input and current_response:
                full_resp = " ".join(current_response)
                formatted_data.append({"text": f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{current_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{full_resp}<|eot_id|>"})
            current_input, current_response = user_text, [house_line]
        else:
            current_response.append(house_line)

    if current_input and current_response:
        full_resp = " ".join(current_response)
        formatted_data.append({"text": f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{current_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{full_resp}<|eot_id|>"})

    return Dataset.from_list(formatted_data)