#GramGuard

GramGuard is a Python-based tool for detecting AI-generated text using statistical differences in n-gram usage. It was originally developed research study (current for ACL-26 submission anonymously for review purpose) on delta-based detection using n-gram statistics. Given a piece of text, this work computes features from multiple n-gram language models (2-gram through 5-gram) and uses a trained classifier to determine if the text was likely produced by a large language model (e.g. GPT) or by a human. This approach leverages pre-trained n-gram models to measure how surprising or divergent the text is at different levels (bigram, trigram, etc.), and then classifies the text based on these signals. The goal is to provide a simple, training-free method to identify AI-generated content by analyzing its distribution of word sequences.
---

## Repository Structure

```
Ngram_DetectGPT/
├── Scripts                           # Include All .py files. Orignal CSV of text and variants CSV's if you want to skip generation you can use same structure of files to start from extract_features_kenlm.py becuase for generate_variants_chatgpt need API. 
├── generate_variants_chatgpt.py      # Paraphrase generation using GPT-4 API
├── extract_features_kenlm.py         # Computes n-gram delta features
├── train_classifier.py               # Trains XGBoost classifiers
├── eval_results.py                   # Aggregates and visualizes evaluation metrics
├── length_robustness.py              # Plots AUC vs. text length
├── ablation_study.py                 # Runs feature ablation experiments
├── plot_figures.py                   # Creates accuracy and AUC plots
├── models/                           # Pretrained KenLM models (2-gram to 5-gram)
├── datasets/                         # Json files (original texts)
├── output/                           # Paraphrased variants per temperature
├── N-gram Scoring/                   # Feature CSVs with delta scores
├── results/                          # Evaluation metrics, plots, summaries
└── README.md
```

---

## Installation Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/N-Gram-dev/Ngram_DetectGPT.git
cd Ngram_DetectGPT
```

### 2. Create and Activate Virtual Environment (Optional but Recommended)

```bash
python3 -m venv env
# macOS/Linux:
source env/bin/activate
# Windows:
env\Scripts\activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Make sure you have Python 3.x installed (Python 3.8 or higher is recommended). The required Python packages are listed below. If you don’t have the `requirements.txt` yet, create one with the content below 👇

---

##  `requirements.txt`

```text
pandas
numpy
scipy
nltk
openai
tqdm
kenlm
xgboost
scikit-learn
matplotlib
seaborn
```

---

##  Download Pretrained KenLM Models

You must download 4 pre-trained n-gram models (2-gram to 5-gram) into a folder named `models/` in the repo root.

###  Source:
👉 for annoymous purpose we can't provide hugging face link to our repository but after camera ready version we will provide link here. 
###  Files to download:

- `2-gram.arpa.bin`
- `3-gram.arpa.bin`
- `4-gram.arpa.bin`
- `5-gram.arpa.bin`

###  Final Structure:

```
models/
├── 2-gram.arpa.bin
├── 3-gram.arpa.bin
├── 4-gram.arpa.bin
└── 5-gram.arpa.bin
```

> These binary KenLM files are used for log-likelihood, entropy, and variance-based delta feature extraction.

---

##  How to Run the Pipeline

The following scripts should be run in order:

1. `generate_variants_chatgpt.py` – Calls OpenAI GPT-4 API to generate 10 paraphrases per sentence at 6 temperatures.
2. `extract_features_kenlm.py` – Computes log-probability, entropy, and frequency variance deltas using KenLM models.
3. `train_classifier.py` – Trains XGBoost classifiers per dataset/model/temperature using extracted delta features.
4. `eval_results.py` – Aggregates performance across all runs and produces summary plots.
5. `plot_figures.py` – Plots accuracy, AUC, and distributions across variants.
6. `ablation_study.py` – Tests the impact of removing feature groups (log-score, entropy, frequency).
7. `length_robustness.py` – Analyzes detection AUC stability across different passage lengths.

> 📂 Output CSVs, ROC-AUC scores, plots, confusion matrices, and summaries are automatically saved in the `results/` folder.

---


##  You're Ready to Go!

Once you’ve installed the requirements and downloaded the models:
- Run each script step-by-step.
- Outputs will be auto-saved in structured folders.
- All results, including variant data, delta features, AUCs, and visualizations, will be available for analysis.

Need help? Open an issue or refer to the scripts' internal comments for guidance.
