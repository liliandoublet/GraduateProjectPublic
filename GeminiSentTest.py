import json
import re
import time
import joblib
import fasttext
import torch
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from tqdm import tqdm

BASE_DIR = Path("/mnt/c/Users/Lilian/Desktop/Data")
DATA_DIR = BASE_DIR
FINE_TUNED_MODEL_PATH = BASE_DIR / "models/camembert_ftV6/model"
OUTPUT_DIR = BASE_DIR / "experiments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 92
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CamemBERT device:", DEVICE)

def clean_newlines(series: pd.Series) -> list[str]:
    return series.str.replace(r"\s*\n\s*", " ", regex=True).tolist()

def load_data(split: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{split}.csv", sep="|")

# Load datasets
train_df = load_data("train")
val_df = load_data("val")
test_df = load_data("test")

# Export by party
EXPORT_DIR = OUTPUT_DIR / "by_party"
for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    for party, group in df.groupby("parti"):
        safe_name = re.sub(r"[^\w\-]", "_", party)
        directory = EXPORT_DIR / split_name / safe_name
        directory.mkdir(parents=True, exist_ok=True)
        group.to_csv(directory / f"{split_name}.csv", sep="|", index=False)
print("Exported by party to", EXPORT_DIR)

# Prepare label mappings
label_mapping = json.loads((BASE_DIR / "models/camembert_ft/label_mapping.json").read_text())
inverse_mapping = {v: k for k, v in label_mapping.items()}
label_ids = list(label_mapping.values())
label_names = [inverse_mapping[i] for i in label_ids]

y_train = train_df["parti"].map(label_mapping).to_numpy()
y_test = test_df["parti"].map(label_mapping).to_numpy()

def compute_f1_macro(true: np.ndarray, pred: np.ndarray) -> float:
    return f1_score(true, pred, average="macro")

results = []
results_per_class = {}
timings = []

# Baseline: TF-IDF + Logistic Regression
print("Baseline: TF-IDF + LogisticRegression")
start = time.time()
vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_df["texte"])
X_test = vectorizer.transform(test_df["texte"])

classifier = LogisticRegression(max_iter=300, n_jobs=-1, class_weight="balanced", random_state=SEED)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
elapsed = time.time() - start

results.append(("tfidf_logreg", compute_f1_macro(y_test, predictions)))
timings.append(("tfidf_logreg", elapsed))
report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
results_per_class["tfidf_logreg"] = [report[str(i)]["f1-score"] for i in label_ids]

joblib.dump(vectorizer, OUTPUT_DIR / "tfidf.pkl")
joblib.dump(classifier, OUTPUT_DIR / "logreg.pkl")

# Baseline: FastText
print("Baseline: FastText")
start = time.time()
fasttext_train_file = OUTPUT_DIR / "fasttext_train.txt"
cleaned_texts = clean_newlines(train_df["texte"])
with fasttext_train_file.open("w", encoding="utf-8") as f:
    for text, label in tqdm(zip(cleaned_texts, y_train), total=len(y_train)):
        f.write(f"__label__{inverse_mapping[label].replace(' ', '_')} {text}\n")

ft_model = fasttext.train_supervised(input=str(fasttext_train_file), epoch=5, lr=1.0, wordNgrams=2, dim=100, verbose=2, seed=SEED)
cleaned_test_texts = clean_newlines(test_df["texte"])
labels_nested, _ = ft_model.predict(cleaned_test_texts)
pred_labels = [label[0] for label in labels_nested]
y_pred_ft = [label_mapping[label.replace("__label__", "").replace("_", " ")] for label in pred_labels]
elapsed = time.time() - start

results.append(("fasttext", compute_f1_macro(y_test, y_pred_ft)))
timings.append(("fasttext", elapsed))
report = classification_report(y_test, y_pred_ft, output_dict=True, zero_division=0)
results_per_class["fasttext"] = [report[str(i)]["f1-score"] for i in label_ids]

ft_model.save_model(str(OUTPUT_DIR / "fasttext.bin"))

# CamemBERT Ablation
def strip_metadata(texts: pd.Series) -> list[str]:
    s = texts.str.replace(r"\[SENT_[^\]]+\]\s*", "", regex=True)
    return s.str.replace(r"\[(?:IRONY|NOIRONY)\]\s*", "", regex=True).tolist()

train_plain = train_df.copy()
train_plain["texte"] = strip_metadata(train_plain["texte"])
test_plain = test_df.copy()
test_plain["texte"] = strip_metadata(test_plain["texte"])

tokenizer_abl = CamembertTokenizer.from_pretrained("camembert-base")
model_abl = CamembertForSequenceClassification.from_pretrained("camembert-base", num_labels=len(label_mapping)).to(DEVICE)

def batch_predict(texts: list[str], tokenizer, model, batch_size: int = 16) -> np.ndarray:
    preds = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            enc = tokenizer(texts[i:i+batch_size], padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            logits = model(**enc).logits
            preds.extend(logits.argmax(-1).cpu().tolist())
    return np.array(preds)

print("CamemBERT Ablation")
start = time.time()
optimizer = torch.optim.AdamW(model_abl.parameters(), lr=2e-5)
model_abl.train()
encoded = tokenizer_abl(train_plain["texte"].tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
batch_size = 8
for i in tqdm(range(0, encoded["input_ids"].size(0), batch_size)):
    loss = model_abl(input_ids=encoded["input_ids"][i:i+batch_size], attention_mask=encoded["attention_mask"][i:i+batch_size], labels=torch.tensor(y_train[i:i+batch_size]).to(DEVICE)).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
elapsed = time.time() - start
y_pred_abl = batch_predict(test_plain["texte"].tolist(), tokenizer_abl, model_abl)

results.append(("camembert_ablation", compute_f1_macro(y_test, y_pred_abl)))
timings.append(("camembert_ablation", elapsed))
report = classification_report(y_test, y_pred_abl, output_dict=True, zero_division=0)
results_per_class["camembert_ablation"] = [report[str(i)]["f1-score"] for i in label_ids]

# CamemBERT Fine-tuned
print("CamemBERT Fine-tuned")
tokenizer_ft = CamembertTokenizer.from_pretrained(FINE_TUNED_MODEL_PATH)
model_ft = CamembertForSequenceClassification.from_pretrained(FINE_TUNED_MODEL_PATH).to(DEVICE)
start = time.time()
y_pred_ft = batch_predict(test_df["texte"].tolist(), tokenizer_ft, model_ft)
elapsed = time.time() - start

results.append(("camembert_finetuned", compute_f1_macro(y_test, y_pred_ft)))
timings.append(("camembert_finetuned", elapsed))
report = classification_report(y_test, y_pred_ft, output_dict=True, zero_division=0)
results_per_class["camembert_finetuned"] = [report[str(i)]["f1-score"] for i in label_ids]

# Results and plots
df_results = pd.DataFrame(results, columns=["model", "f1_macro"]).sort_values("f1_macro", ascending=False)
df_results.to_csv(OUTPUT_DIR / "baseline_results.csv", index=False)
print("F1-macro results:\n", df_results)

plt.figure(figsize=(6, 3))
plt.barh(df_results["model"], df_results["f1_macro"])
plt.xlabel("F1-macro")
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "f1_barplot.png", dpi=300)

angles = np.linspace(0, 2*np.pi, len(label_names), endpoint=False).tolist()
angles += angles[:1]
plt.figure(figsize=(6, 6))
for name, scores in results_per_class.items():
    values = scores + scores[:1]
    plt.polar(angles, values, label=name, linewidth=1.5)
    plt.fill(angles, values, alpha=0.08)
plt.xticks(angles[:-1], label_names, fontsize=8)
plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], fontsize=7)
plt.ylim(0, 1)
plt.title("F1 by party and model")
plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "f1_radar.png", dpi=300)

results_time_df = pd.DataFrame(timings, columns=["model", "seconds"])
df_merged = df_results.merge(results_time_df, on="model")
plt.figure(figsize=(5, 3))
plt.scatter(df_merged["seconds"], df_merged["f1_macro"])
for _, row in df_merged.iterrows():
    plt.text(row["seconds"], row["f1_macro"] + 0.02, row["model"], fontsize=7)
plt.xlabel("Training time (s)")
plt.ylabel("F1-macro")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "time_vs_f1.png", dpi=300)

print("All outputs saved in", OUTPUT_DIR)