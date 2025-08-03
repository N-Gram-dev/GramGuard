# 🔍 NGram-DetectGPT: AI Text Detection using N-Gram Delta Features

This project implements a robust detection pipeline to distinguish AI-generated text from human-written content using classical n-gram language models (KenLM) and perturbation-based delta features. It includes paraphrase generation, n-gram scoring, feature extraction, classifier training, evaluation, and result visualization.

---

## 📁 Repository Structure
Ngram_DetectGPT/
├── generate_variants_chatgpt.py # Paraphrase generation using GPT-4 API
├── extract_features_kenlm.py # Computes n-gram delta features
├── train_classifier.py # Trains XGBoost classifiers
├── eval_results.py # Aggregates and visualizes evaluation metrics
├── length_robustness.py # Plots AUC vs. text length
├── ablation_study.py # Runs feature ablation experiments
├── plot_figures.py # Creates accuracy and AUC plots
├── models/ # Pretrained KenLM models (2-gram to 5-gram)
├── datasets/ # Input CSVs (original texts)
├── output/ # Paraphrased variants per temperature
├── N-gram Scoring/ # Feature CSVs with delta scores
├── results/ # Evaluation metrics, plots, summaries
└── README.md


---

## 🧰 Installation Instructions

### 1. ✅ Clone the Repository

```bash
git clone https://github.com/N-Gram-dev/Ngram_DetectGPT.git
cd Ngram_DetectGPT

2. ✅ Create and Activate Virtual Environment (Optional but Recommended)


python3 -m venv env
source env/bin/activate      # On Windows: env\Scripts\activate

3. ✅ Install Python Dependencies


pip install -r requirements.txt

📦 requirements.txt

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


📥 Download Pretrained KenLM Models
You must download 4 pre-trained n-gram models (2-gram to 5-gram) into a folder named models/ in the repo root.

📍 Source:
👉 https://huggingface.co/NGramDev/ngram-detect-models

🧾 Files to download:
2-gram.arpa.bin

3-gram.arpa.bin

4-gram.arpa.bin

5-gram.arpa.bin

models/
├── 2-gram.arpa.bin
├── 3-gram.arpa.bin
├── 4-gram.arpa.bin
└── 5-gram.arpa.bin

These binary KenLM files are used for log-likelihood, entropy, and variance-based delta feature extraction.

How to Run the Pipeline
The following scripts should be run in order:

generate_variants_chatgpt.py – Calls OpenAI GPT-4 API to generate 10 paraphrases per sentence at 6 temperatures.

extract_features_kenlm.py – Computes log-probability, entropy, and frequency variance deltas using KenLM models.

train_classifier.py – Trains XGBoost classifiers per dataset/model/temperature using extracted delta features.

eval_results.py – Aggregates performance across all runs and produces summary plots.

plot_figures.py – Plots accuracy, AUC, and distributions across variants.

ablation_study.py – Tests the impact of removing feature groups (log-score, entropy, frequency).

length_robustness.py – Analyzes detection AUC stability across different passage lengths.

📂 Output CSVs, ROC-AUC scores, plots, confusion matrices, and summaries are automatically saved in the results/ folder.


