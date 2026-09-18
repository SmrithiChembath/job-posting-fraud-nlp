"""
TF-IDF + Logistic Regression baseline. Fast to train, gives an honest benchmark to
beat before reaching for a transformer, and its coefficients are directly interpretable.

Run: python src/baseline_model.py
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
TARGET = "fraudulent"


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{name}.csv")


def main():
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    vectorizer = TfidfVectorizer(
        max_features=20_000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english",
    )
    X_train = vectorizer.fit_transform(train_df["text"])
    X_val = vectorizer.transform(val_df["text"])
    X_test = vectorizer.transform(test_df["text"])

    y_train, y_val, y_test = train_df[TARGET], val_df[TARGET], test_df[TARGET]

    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        C=1.0,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    print("=== Validation set ===")
    val_probs = clf.predict_proba(X_val)[:, 1]
    val_preds = clf.predict(X_val)
    print(classification_report(y_val, val_preds, target_names=["real", "fraudulent"]))
    print(f"ROC-AUC: {roc_auc_score(y_val, val_probs):.4f}")
    print(f"PR-AUC:  {average_precision_score(y_val, val_probs):.4f}")

    print("\n=== Test set ===")
    test_probs = clf.predict_proba(X_test)[:, 1]
    test_preds = clf.predict(X_test)
    print(classification_report(y_test, test_preds, target_names=["real", "fraudulent"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, test_probs):.4f}")
    print(f"PR-AUC:  {average_precision_score(y_test, test_probs):.4f}")

    # Most fraud-indicative and most real-indicative terms, for the README/EDA story
    feature_names = vectorizer.get_feature_names_out()
    coefs = clf.coef_[0]
    top_fraud_idx = coefs.argsort()[-15:][::-1]
    top_real_idx = coefs.argsort()[:15]
    print("\nTop terms pushing toward FRAUDULENT:")
    for i in top_fraud_idx:
        print(f"  {feature_names[i]:<25} {coefs[i]:+.3f}")
    print("\nTop terms pushing toward REAL:")
    for i in top_real_idx:
        print(f"  {feature_names[i]:<25} {coefs[i]:+.3f}")

    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / "baseline_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(MODEL_DIR / "baseline_model.pkl", "wb") as f:
        pickle.dump(clf, f)
    print(f"\nSaved baseline model + vectorizer to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
