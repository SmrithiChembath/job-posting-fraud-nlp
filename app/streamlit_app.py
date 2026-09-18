"""
Streamlit demo: paste a job posting, get a fraud probability from the fine-tuned
BERT model plus a SHAP-highlighted explanation of which words drove the prediction.

Run: streamlit run app/streamlit_app.py
"""

from pathlib import Path

import shap
import streamlit as st
import streamlit.components.v1 as components
from transformers import pipeline

MODEL_DIR = Path("models/bert_fraud_model")
MAX_LENGTH = 256

EXAMPLE_REAL = (
    "Senior Data Analyst - We are looking for a Senior Data Analyst to join our "
    "marketing team. You will build dashboards in Tableau, partner with stakeholders "
    "to define KPIs, and present findings to leadership. Requirements: 3+ years SQL, "
    "experience with A/B testing, bachelor's degree in a quantitative field."
)
EXAMPLE_FRAUD = (
    "URGENT HIRING - Work From Home Data Entry! Earn $5000/week, no experience "
    "needed. To secure your position, send a $50 processing fee and your bank "
    "account details via email today. Limited spots available, apply now!"
)


@st.cache_resource
def load_pipeline():
    return pipeline(
        "text-classification",
        model=str(MODEL_DIR),
        tokenizer=str(MODEL_DIR),
        top_k=None,
        truncation=True,
        max_length=MAX_LENGTH,
    )


@st.cache_resource
def load_explainer(_clf):
    return shap.Explainer(_clf)


def main():
    st.set_page_config(page_title="Job Posting Fraud Detector", page_icon="🕵️")
    st.title("🕵️ Job Posting Fraud Detector")
    st.write(
        "Paste a job posting below. The model (a fine-tuned DistilBERT) estimates "
        "the probability it's fraudulent, and SHAP highlights which words pushed "
        "the prediction in each direction."
    )

    if not MODEL_DIR.exists():
        st.error(
            f"No trained model found at `{MODEL_DIR}`. Run `python src/bert_train.py` "
            "first, then restart this app."
        )
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load a real-looking example"):
            st.session_state["posting_text"] = EXAMPLE_REAL
    with col2:
        if st.button("Load a fraud-looking example"):
            st.session_state["posting_text"] = EXAMPLE_FRAUD

    text = st.text_area(
        "Job posting text",
        value=st.session_state.get("posting_text", ""),
        height=220,
        placeholder="Paste a job title, company description, and requirements here...",
    )

    if st.button("Analyze", type="primary") and text.strip():
        clf = load_pipeline()
        with st.spinner("Scoring..."):
            scores = clf(text[:2000])[0]
            score_map = {s["label"]: s["score"] for s in scores}
            fraud_prob = score_map.get("fraudulent", 0.0)

        st.metric("Fraud probability", f"{fraud_prob:.1%}")
        if fraud_prob >= 0.5:
            st.error("Flagged as likely FRAUDULENT")
        else:
            st.success("Predicted REAL")

        with st.spinner("Computing SHAP explanation..."):
            explainer = load_explainer(clf)
            shap_values = explainer([text[:1500]])
            html = shap.plots.text(shap_values[:, :, "fraudulent"], display=False)

        st.subheader("Why the model made this prediction")
        st.caption(
            "Red words pushed the prediction toward 'fraudulent', blue words pushed "
            "it toward 'real'. Hover over a word to see its exact contribution."
        )
        components.html(html, height=400, scrolling=True)


if __name__ == "__main__":
    main()
