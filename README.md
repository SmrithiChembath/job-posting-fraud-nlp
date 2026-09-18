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

[Real or Fake Job Postings (EMSCAD)](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction),
about 18,000 job postings scraped by the University of the Aegean, with a binary `fraudulent`
label. Roughly 5% of postings are fraudulent, so this is a realistic, imbalanced,
real-world text classification problem, not a toy balanced dataset.

Fields used: `title`, `company_profile`, `description`, `requirements`, `benefits`
(combined into one text field), plus the `fraudulent` target.

## Approach

1. **EDA** (`notebooks/01_eda.ipynb`): class imbalance, missing values by field, text
   length distribution, most distinctive words per class.
2. **Baseline model** (`src/baseline_model.py`): TF-IDF (1-2 grams) plus Logistic Regression
   with balanced class weights. Fast, interpretable, and a fair benchmark before reaching
   for a transformer.
3. **Fine-tuned transformer** (`src/bert_train.py`): DistilBERT fine-tuned on the same
   text with a class-weighted loss to handle the 5% positive rate. Evaluated on
   precision/recall/F1 and PR-AUC (ROC-AUC is misleading on this imbalance).
4. **Explainability** (`src/interpret.py`): SHAP applied to the fine-tuned model's
   predictions, so each flagged posting comes with the words that pushed it toward
   "fraudulent." 
5. **Demo** (`app/streamlit_app.py`): paste any job posting text and get a fraud
   probability and a highlighted explanation, live.

## Results

Metrics below are on the held-out test set (2,682 postings the model never saw during
training or validation).

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | 70.0% | 88.0% | 0.780 | 0.987 | 0.896 |
| Fine-tuned DistilBERT | 90.0% | 76.2% | 0.825 | 0.974 | 0.864 |

The two models make different tradeoffs. The TF-IDF baseline catches more fraud (88%
recall) but flags more real postings by mistake (70% precision, so 3 in 10 flagged
postings are actually real). DistilBERT flips that: it misses more fraud (76% recall)
but is much more trustworthy when it does flag something (90% precision, so 9 in 10
flagged postings are genuinely fraudulent). Which one you'd pick depends on the cost of
a false positive versus a false negative in the actual product: for something a human
reviews before acting on, the baseline's higher recall is arguably more useful, since it
surfaces more real fraud for a reviewer to catch, at the cost of some extra review time.
The baseline also edges out DistilBERT on PR-AUC (0.896 vs. 0.864), the more honest
summary metric on this kind of class imbalance, since ROC-AUC can look artificially
strong when the negative class dominates. That the simpler, much faster model performs
this competitively is itself a useful finding, and is part of why running the baseline
first instead of jumping straight to a transformer matters.

**What the model picks up on.** Tested live in the Streamlit demo:

- A fabricated "Remote Data Entry Clerk" posting (urgency language, no experience
  required, a request to deposit a check and wire money back, and a request for bank
  routing and date-of-birth details) was flagged at 99.6% fraud probability. The
  SHAP explanation highlighted "urgently," "Immediate," the dollar amount, "deposit,"
  "wire," and "apply today" as the strongest signals toward fraudulent.
- A real Data Scientist posting from Love's Travel Stops was correctly scored at 0.4%
  fraud probability. The explanation was driven by concrete, specific details: named
  benefits ("401(k)," "Medical/Dental/Vision Insurance"), a specific team and job
  function, and named tools ("R, Python, SAS"), all pushing toward "real."
- One limitation worth naming honestly: in one true-fraud example from the test set
  (a "Director of Strategy" posting), the model's top signals actually leaned toward
  "real," driven by phrases like "Silicon Valley" and "San Mateo." Fraudulent postings
  that borrow the vocabulary of well-known, legitimate tech hubs are the model's
  weak spot, which lines up with the model catching only 76% of fraud overall rather
  than closer to 100%.

## How to run


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
