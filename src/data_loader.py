import os
import re  
import pandas as pd
from datasets import Dataset

def get_house_dataset(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find data at {csv_path}")

    # Using sep='\t' for your FNVEdit export
    df = pd.read_csv(csv_path, sep='\t')
    
    formatted_data = []
    current_input = ""
    current_response = []
    instruction = (
        "You are Robert House, the CEO of RobCo and technocratic overlord of New Vegas. "
        "Speak with extreme sophistication, cold logic, and a condescending sense of superiority."
    )

    for _, row in df.iterrows():
        # Handle User Prompt
        user_text = str(row['PROMPT'] if pd.notna(row['PROMPT']) and row['PROMPT'] != '' else row['TOPIC TEXT']).strip()
        
        # Handle House Response & Strip {stage directions}
        house_line = str(row['RESPONSE TEXT'] if pd.notna(row['RESPONSE TEXT']) else "").strip()
        house_line = re.sub(r'\{.*?\}', '', house_line).strip()
        
        # Skip empty lines if any exist after cleaning
        if not house_line:
            continue

        index = row['RESPONSE INDEX']

        if index == 1:
            # Save the previous conversation block
            if current_input and current_response:
                full_response = " ".join(current_response)
                formatted_data.append({
                    "text": f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|>"
                            f"<|start_header_id|>user<|end_header_id|>\n\n{current_input}<|eot_id|>"
                            f"<|start_header_id|>assistant<|end_header_id|>\n\n{full_response}<|eot_id|>"
                })
            current_input = user_text
            current_response = [house_line]
        else:
            # It's Index 2, 3, etc. - append to the current monologue
            current_response.append(house_line)

    if current_input and current_response:
        full_response = " ".join(current_response)
        formatted_data.append({
            "text": f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|>"
                    f"<|start_header_id|>user<|end_header_id|>\n\n{current_input}<|eot_id|>"
                    f"<|start_header_id|>assistant<|end_header_id|>\n\n{full_response}<|eot_id|>"
        })

    print(f" {len(formatted_data)}")
    return Dataset.from_list(formatted_data)