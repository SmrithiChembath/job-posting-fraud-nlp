# Fraudulent Job Posting Detector (NLP + BERT + SHAP)

## Problem

Online classifieds and job marketplaces are constantly targeted by fake or scam postings
(fee-harvesting schemes, data-phishing "recruiters," shell companies). Manually reviewing
every posting doesn't scale. This project builds a text classifier that flags likely
fraudulent job postings from their raw text, and explains *why* each posting was flagged
so a human reviewer can trust and act on the output quickly.

This mirrors the kind of trust-and-safety / listing-quality problem that comes up on any
classifieds marketplace (property, jobs, marketplace listings).

## Data

[Real or Fake Job Postings (EMSCAD)](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) —
~18,000 job postings scraped by the University of the Aegean, with a binary `fraudulent`
label. Roughly 5% of postings are fraudulent, so this is a realistic, imbalanced,
real-world text classification problem, not a toy balanced dataset.

Fields used: `title`, `company_profile`, `description`, `requirements`, `benefits`
(combined into one text field), plus the `fraudulent` target.

## Approach

1. **EDA** (`notebooks/01_eda.ipynb`) — class imbalance, missing values by field, text
   length distribution, most distinctive words per class.
2. **Baseline model** (`src/baseline_model.py`) — TF-IDF (1-2 grams) + Logistic Regression
   with balanced class weights. Fast, interpretable, and a fair benchmark before reaching
   for a transformer.
3. **Fine-tuned transformer** (`src/bert_train.py`) — DistilBERT fine-tuned on the same
   text with a class-weighted loss to handle the 5% positive rate. Evaluated on
   precision/recall/F1 and PR-AUC (ROC-AUC is misleading on this imbalance).
4. **Explainability** (`src/interpret.py`) — SHAP applied to the fine-tuned model's
   predictions, so each flagged posting comes with the words that pushed it toward
   "fraudulent." Same explainability approach used in my [LendingClub credit risk
   project](../lendingclub-credit-risk), applied here to text instead of tabular data.
5. **Demo** (`app/streamlit_app.py`) — paste any job posting text, get a fraud probability
   and a highlighted explanation, live.

## Results

_Fill in after training — replace this section with your actual numbers._

| Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | | | | |
| Fine-tuned DistilBERT | | | | |

Notable examples of what the model picks up on: _add 2-3 SHAP examples here, e.g.
"unusually generous salary + request for bank details in `requirements` were the top
signals on posting #X."_
Test set (the numbers that matter most, since it's held-out data the model never touched):

Precision: 90.0%
Recall: 76.2%
F1: 0.825
ROC-AUC: 0.974
PR-AUC: 0.864

## How to run

See [SETUP.md](./SETUP.md) for full step-by-step instructions (environment, data,
training, demo).

```
pip install -r requirements.txt
python src/preprocess.py          # cleans data, creates train/val/test splits
python src/baseline_model.py      # trains + evaluates the TF-IDF baseline
python src/bert_train.py          # fine-tunes DistilBERT (needs a GPU or ~30-60 min on CPU)
python src/interpret.py           # generates SHAP explanations for sample postings
streamlit run app/streamlit_app.py
```

## Project structure

```
job-posting-fraud-nlp/
├── data/                # raw + processed data (not committed, see .gitignore)
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── preprocess.py
│   ├── baseline_model.py
│   ├── bert_train.py
│   └── interpret.py
├── app/
│   └── streamlit_app.py
├── models/              # saved model artifacts (not committed)
├── requirements.txt
└── README.md
```

## What this project demonstrates

- Working with real, messy, imbalanced text data (not a cleaned Kaggle-beginner dataset)
- A proper baseline-before-deep-learning workflow, with a documented comparison
- Fine-tuning a transformer (Hugging Face `transformers`) rather than only using
  pretrained embeddings
- Model explainability on text with SHAP, consistent with the explainability approach
  used across my other projects
- Shipping a usable interface (Streamlit), not just a notebook
