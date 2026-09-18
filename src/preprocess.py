"""
Loads the raw EMSCAD job postings CSV, cleans the text fields, combines them into
a single text column, and writes stratified train/val/test splits to data/processed/.

Run: python src/preprocess.py
"""

import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split

RAW_PATH = Path("data/fake_job_postings.csv")
OUT_DIR = Path("data/processed")

TEXT_FIELDS = ["title", "company_profile", "description", "requirements", "benefits"]
TARGET = "fraudulent"

RANDOM_STATE = 42
VAL_SIZE = 0.15
TEST_SIZE = 0.15


def strip_html(raw_text: str) -> str:
    """Remove HTML tags and collapse whitespace. Handles NaN/float input safely."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        return ""
    text = BeautifulSoup(raw_text, "html.parser").get_text(separator=" ")
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # strip raw URLs
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_combined_text(df: pd.DataFrame) -> pd.Series:
    """Join the text fields with a period so they read as separate sentences.
    (Deliberately not using a literal "[SEP]" string here: it would just become
    a stray word to the TF-IDF vectorizer and pollute the top-feature list; BERT's
    real [SEP] token is inserted by the tokenizer itself, not by us.)"""
    cleaned = pd.DataFrame({col: df[col].apply(strip_html) for col in TEXT_FIELDS})
    combined = cleaned.apply(
        lambda row: " . ".join(v for v in row if v), axis=1
    )
    return combined


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Couldn't find {RAW_PATH}. Download fake_job_postings.csv from Kaggle "
            "(see SETUP.md step 5) and place it there."
        )

    df = pd.read_csv(RAW_PATH)
    print(f"Loaded {len(df):,} rows, {df[TARGET].sum():,} fraudulent "
          f"({df[TARGET].mean() * 100:.1f}%)")

    df["text"] = build_combined_text(df)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    print(f"{len(df):,} rows remain after dropping empty-text postings")

    keep_cols = ["job_id", "text", TARGET]
    df = df[keep_cols]

    train_df, temp_df = train_test_split(
        df,
        test_size=(VAL_SIZE + TEST_SIZE),
        stratify=df[TARGET],
        random_state=RANDOM_STATE,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=TEST_SIZE / (VAL_SIZE + TEST_SIZE),
        stratify=temp_df[TARGET],
        random_state=RANDOM_STATE,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(OUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUT_DIR / "val.csv", index=False)
    test_df.to_csv(OUT_DIR / "test.csv", index=False)

    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"{name}: {len(split):,} rows, "
              f"{split[TARGET].mean() * 100:.1f}% fraudulent")

    print(f"\nSaved splits to {OUT_DIR}/")


if __name__ == "__main__":
    main()
