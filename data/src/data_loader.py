import os
import pandas as pd
from datasets import Dataset

def get_house_dataset(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find data at {csv_path}")

    df = pd.read_csv(csv_path, sep='\t')
    
    formatted_data = []
    current_input = ""
    current_response = []
    instruction = (
        "You are Robert House, the CEO of RobCo and technocratic overlord of New Vegas. "
        "Speak with extreme sophistication, cold logic, and a condescending sense of superiority."
    )

    for _, row in df.iterrows():
        user_text = str(row['PROMPT'] if pd.notna(row['PROMPT']) and row['PROMPT'] != '' else row['TOPIC TEXT']).strip()
        house_line = str(row['RESPONSE TEXT']).strip().replace('{arch}', '').replace('{...}', '')
        index = row['RESPONSE INDEX']

        if index == 1:
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
            current_response.append(house_line)

    # Catch the last block
    if current_input and current_response:
        full_response = " ".join(current_response)
        formatted_data.append({
            "text": f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|>"
                    f"<|start_header_id|>user<|end_header_id|>\n\n{current_input}<|eot_id|>"
                    f"<|start_header_id|>assistant<|end_header_id|>\n\n{full_response}<|eot_id|>"
        })

    return Dataset.from_list(formatted_data)