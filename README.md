# LLM Distillation Detection & Analysis

This repository contains the data collection and analysis pipeline for studying LLM distillation attacks. The goal of this phase is to analyze query patterns, latency, and response intervals to detect programmatic data extraction (distillation) versus normal human usage.

## Repository Layout

- **`data/`**
  - `responses_openai.csv`: The core dataset containing collected responses from the OpenAI API, including metadata like generation latency, category, and response length.
  
- **`Wolfram/`**
  - Contains Wolfram Language notebooks (`.nb`) used for statistical analysis and anomaly detection. These notebooks analyze hardware physics patterns (token-to-latency mathematical trends) to differentiate between normal users and programmatic attackers.

- **`figures/`**
  - Generated charts and visualizations (e.g., `response_length_analysis.png`, histograms, and box plots) visualizing the differences in query patterns and response lengths across different categories.

- **Scripts** (e.g., `collect_responses.py`)
  - Python scripts used to interact with the OpenAI API, enforce specific request intervals (e.g., 20-second delays to avoid rate-limit flagging), and log the resulting metadata to the CSV dataset.
  - Python data analysis scripts utilizing `pandas` and `matplotlib` for initial EDA and metric extraction.

## Setup & Requirements

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ahmedco31/llm-distillation.git
   cd llm-distillation

2. **Environment Setup:**
It is recommended to use a virtual environment (e.g., `distillation_env`).
   ```bash
   python3 -m venv distillation_env
   source distillation_env/bin/activate
   pip install pandas matplotlib openai

*(Note: The distillation_env/ directory is intentionally ignored in .gitignore to prevent uploading large dependency files).*

3. **Collecting data:**

Data collected from gpt-4o-mini (fixed interval) (collect_responses.py needs to be modified back to fixed intervals and different categories sizes)
```bash
   python scripts/collect_responses.py \
     --prompts data/prompts_v1.json \
     --output  data/responses_openai.csv \
     --model   gpt-4o \
     --limit   400 \
     --shuffle \
     --seed    42 \
     --interval 20.0 \
     --checkpoint-every 50
```

Data collected from gpt-4o (random intervals between 10 and 25
```bash
   python scripts/collect_responses.py \
     --prompts data/prompts_v2.json \
     --output  data/responses_v2.csv \
     --model   gpt-4o \
     --limit   1200 \
     --shuffle \
     --seed    42 \
     --min-interval 10.0 \
     --checkpoint-every 50
```

4. **Data Analysis:**

Data analysis available in both python (`distillation_analysis_code.ipynb`) and Wolfram (`Wolfram/Data.nb`)
