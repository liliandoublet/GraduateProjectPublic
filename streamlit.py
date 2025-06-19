import streamlit as st
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import torch
import torch.nn.functional as F
import pandas as pd
import altair as alt
from pathlib import Path

# Utility functions
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    """Load tokenizer, model, and id→label mapping."""
    tokenizer = CamembertTokenizer.from_pretrained(model_path)
    model = CamembertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    id2label = model.config.id2label
    return tokenizer, model, id2label

def predict(text: str, tokenizer, model):
    """Return softmax probabilities for each class."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    return F.softmax(logits, dim=1)[0].tolist()

def build_prefix(sentiment_choice: str, irony_choice: str) -> str:
    """Build the fine-tuning prefix for sentiment and irony."""
    sentiment_map = {
        "Positive": "positif",
        "Neutral": "neutre",
        "Negative": "negatif",
    }
    token = sentiment_map.get(sentiment_choice, "neutre")
    prefix = f"[SENT_{token}] "
    prefix += "[IRONY] " if irony_choice == "Yes" else "[NOIRONY] "
    return prefix

# Sidebar: model path and metadata inputs
st.sidebar.title("⚙️ Settings")

default_model_dir = Path(__file__).parent / "models/camembert_ftV3/model"
model_dir = st.sidebar.text_input(
    "CamemBERT model path",
    value=str(default_model_dir.resolve()),
)

st.sidebar.markdown("### Metadata")
irony_input = st.sidebar.radio("Irony?", ["No", "Yes"], index=0)
sentiment_input = st.sidebar.radio("Sentiment", ["Neutral", "Positive", "Negative"], index=0)

# Load the model
try:
    tokenizer, model, id2label = load_model(model_dir)
except Exception as e:
    st.sidebar.error(f"❌ Failed to load model: {e}")
    st.stop()

class_names = [id2label[i] for i in range(len(id2label))]

# Main app
st.set_page_config(page_title="Political Classifier", layout="wide")
st.title("🗳️ Political Party Detection")
st.markdown("Paste a speech, tweet, or article below 👇")

text_input = st.text_area("Text to analyze", height=250)
classify = st.button("Classify")

if classify or text_input.strip():
    if not text_input.strip():
        st.warning("👉 Please enter some text.")
        st.stop()

    prefix = build_prefix(sentiment_input, irony_input)
    full_text = prefix + text_input

    probabilities = predict(full_text, tokenizer, model)

    results_df = pd.DataFrame({
        "Party": class_names,
        "Probability (%)": [round(p * 100, 2) for p in probabilities],
    }).sort_values("Probability (%)", ascending=False)

    st.subheader("Class Probabilities")
    st.table(results_df)

    st.subheader("Visualization")
    chart = (
        alt.Chart(results_df)
        .mark_bar()
        .encode(
            x=alt.X("Party", sort="-y"),
            y="Probability (%)"
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)

    top_party = results_df.iloc[0]["Party"]
    top_prob = results_df.iloc[0]["Probability (%)"]
    st.success(f"🏆 Predicted party: **{top_party}** ({top_prob} %)")
else:
    st.info("➡️ Paste text and click **Classify**.")