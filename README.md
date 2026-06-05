# Mr. House AI

A fine-tuned LLM persona project based on Robert Edwin House from Fallout: New Vegas. Built with Unsloth, LoRA, and LLaMA 3.1 8B Instruct, served through a Flask API and a New Vegas terminal-styled Vue.js frontend.

---

## What It Is

This project fine-tunes Meta's LLaMA 3.1 8B Instruct model on Mr. House's in-game dialogue using LoRA (Low-Rank Adaptation). The result is a conversational AI that speaks in House's voice — cold, dismissive, aristocratic — and can discuss both Fallout lore and modern topics. It also has optional web search to answer real-time questions about anime, games, and manga in character.

---

## Project Structure

```
Mr-House-AI/
├── index.html              # Vue.js frontend (New Vegas terminal UI)
├── style.css               # Terminal styling
├── app.py                  # Flask backend API
├── train.py                # LoRA fine-tuning script
├── test_model.py           # CLI chat interface with web search
├── persona.txt             # Mr. House system prompt and persona rules
├── merge_datasets.py       # Merges game data with synthetic data
├── generate_synthetic.py   # Generates synthetic training examples (requires Anthropic API)
├── data/
│   └── house_v2_clean.csv  # Extracted Mr. House dialogue from Fallout: New Vegas
└── src/
    └── data_loader.py      # Loads and formats the CSV into Llama 3 training examples
```

---

## Requirements

### Training (Google Colab A100)
```
unsloth
unsloth-zoo
transformers
trl
torch
datasets
pandas
```

### Backend
```
flask
flask-cors
ddgs
torch
unsloth
```

### Frontend
No npm required. Vue 3 is loaded from CDN. Just open `index.html` in a browser.

---

## Setup & Usage

### 1. Training (run in Google Colab)

First fix Unsloth version conflicts if they appear:
```python
!pip install --upgrade --force-reinstall unsloth unsloth-zoo
```

Then restart the runtime and run:
```python
!python train.py
```

Training takes roughly 15-20 minutes on an A100. The fine-tuned weights are saved to `house_lora_final/`.

Save the weights to Google Drive so they persist between Colab sessions:
```python
import shutil
from google.colab import drive
drive.mount('/content/drive')
shutil.copy("house_lora_final.zip", "/content/drive/MyDrive/house_lora_final.zip")
```

### 2. Restore Weights in a New Colab Session

```python
from google.colab import drive
drive.mount('/content/drive')

import zipfile
with zipfile.ZipFile("/content/drive/MyDrive/house_lora_final.zip", 'r') as z:
    z.extractall("house_lora_final")
```

### 3. Run the CLI Chat

```python
!pip install ddgs
!python test_model.py
```

### 4. Run the Web UI

**In Colab**, start the Flask backend:
```python
!pip install flask flask-cors ddgs
!python app.py
```

**In Colab**, expose it publicly with ngrok:
```python
!pip install pyngrok
from pyngrok import ngrok
url = ngrok.connect(5000)
print(url)
```

**In `index.html`**, replace the API URL with your ngrok URL:
```javascript
const API_URL = 'https://your-ngrok-url.ngrok.io/chat';
```

**Locally**, open `index.html` in your browser.

---

## How It Works

### Fine-tuning
- **Base model:** `unsloth/Meta-Llama-3.1-8B-Instruct`
- **Method:** LoRA (r=16, alpha=32) on all 7 projection modules
- **Dataset:** 181 cleaned dialogue examples extracted from Fallout: New Vegas game files
- **Trainer:** Hugging Face SFTTrainer via Unsloth

### Data Pipeline
`data_loader.py` reads the raw game CSV, groups multi-line responses by `TOPIC TEXT`, cleans stage directions and engine artifacts, and formats everything into Llama 3 chat template:
```
<|system|> persona.txt
<|user|>   Courier's dialogue choice
<|assistant|> House's response
```

### Web Search
`test_model.py` and `app.py` use DuckDuckGo search (no API key required) to fetch real-time data when the user asks about release dates, ratings, or current events. The result is injected as a small note in the user turn before generation.

### Persona
`persona.txt` controls House's behavior at inference time — speech patterns, vocabulary, what topics to dismiss, what to engage with, and hard constraints on assistant-like behavior. No retrain needed to adjust the persona.

---

## Known Limitations

- **181 training examples** is a small dataset. House handles in-universe topics well but may occasionally slip into base model behavior on unfamiliar modern topics.
- **Web search** works best for factual queries. Complex or ambiguous searches may return irrelevant context.
- **Colab sessions reset** — weights must be restored from Google Drive each session.
- The model runs on GPU only. CPU inference is not supported with the current config.

---

## Dataset

Dialogue extracted from Fallout: New Vegas game files (`FalloutNV.esm`) using the xEdit export format. Contains Mr. House's complete in-game dialogue across all quests.

---

## Built With

- [Unsloth](https://github.com/unslothai/unsloth) — fast LoRA fine-tuning
- [Meta LLaMA 3.1 8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) — base model
- [Hugging Face TRL](https://github.com/huggingface/trl) — SFTTrainer
- [Flask](https://flask.palletsprojects.com/) — backend API
- [Vue.js 3](https://vuejs.org/) — frontend
- [ddgs](https://pypi.org/project/ddgs/) — DuckDuckGo search

---

## Acknowledgements

Fallout: New Vegas and Mr. House are the property of Bethesda Softworks and Obsidian Entertainment. This project is a fan-made, non-commercial AI experiment.