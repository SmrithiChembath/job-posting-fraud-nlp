"""
Runs SHAP over the fine-tuned BERT model to explain individual predictions: which
words in a posting pushed the model toward "fraudulent" vs "real". Same explainability
approach used in the LendingClub credit risk project, applied here to text.

Run: python src/interpret.py

Outputs:
  - outputs/shap_examples.html   (interactive highlighted-text explanations)
  - prints top contributing words per example to the terminal
"""

from pathlib import Path

import pandas as pd
import shap
from transformers import pipeline

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models/bert_fraud_model")
OUT_DIR = Path("outputs")
N_EXAMPLES_PER_CLASS = 3
MAX_LENGTH = 256


def top_tokens_for_example(shap_values, index, target_label="fraudulent", k=10):
    """Return the k tokens with the largest |SHAP value| for the target class."""
    values = shap_values[index, :, target_label].values
    tokens = shap_values[index, :, target_label].data
    pairs = sorted(zip(tokens, values), key=lambda x: abs(x[1]), reverse=True)
    return pairs[:k]


def main():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"No model found at {MODEL_DIR}. Run src/bert_train.py first."
        )

    test_df = pd.read_csv(DATA_DIR / "test.csv")

    fraud_examples = test_df[test_df["fraudulent"] == 1].head(N_EXAMPLES_PER_CLASS)
    real_examples = test_df[test_df["fraudulent"] == 0].head(N_EXAMPLES_PER_CLASS)
    examples = pd.concat([fraud_examples, real_examples])["text"].tolist()
    # SHAP's text masker is slow on very long inputs; keep examples bounded
    examples = [t[:1500] for t in examples]

    clf = pipeline(
        "text-classification",
        model=str(MODEL_DIR),
        tokenizer=str(MODEL_DIR),
        top_k=None,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    explainer = shap.Explainer(clf)
    print(f"Computing SHAP values for {len(examples)} examples "
          "(this can take a couple of minutes)...")
    shap_values = explainer(examples)

    OUT_DIR.mkdir(exist_ok=True)
    html = shap.plots.text(shap_values[:, :, "fraudulent"], display=False)
    with open(OUT_DIR / "shap_examples.html", "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved interactive explanations to {OUT_DIR / 'shap_examples.html'} "
          "(open it in a browser)")

    print("\n=== Top contributing words per example ===")
    for i, text in enumerate(examples):
        label = "FRAUDULENT (true)" if i < len(fraud_examples) else "REAL (true)"
        print(f"\n--- Example {i + 1} [{label}] ---")
        print(text[:200] + ("..." if len(text) > 200 else ""))
        top = top_tokens_for_example(shap_values, i, target_label="fraudulent")
        for token, value in top:
            direction = "-> fraudulent" if value > 0 else "-> real"
            print(f"  {token!r:<20} {value:+.4f}  {direction}")


if __name__ == "__main__":
    main()
