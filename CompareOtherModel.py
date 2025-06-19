import json
import re
import string
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
MODEL_DIR = BASE_DIR / "models/camembert_ftV6/model"
OUTPUT_DIR = BASE_DIR / "experiments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 92
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device for CamemBERT:", DEVICE)

def remove_newlines(series: pd.Series) -> list[str]:
    """Remove newline characters and return a list of clean strings."""
    return series.str.replace(r"\s*\n\s*", " ", regex=True).tolist()

def load_dataframe(split: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{split}.csv", sep="|")

train_df, val_df, test_df = map(load_dataframe, ["train", "val", "test"])

EXPORT_BY_PARTY = OUTPUT_DIR / "by_party"
for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    for party, group in df.groupby("parti"):
        safe_party = re.sub(r"[^\w\-]", "_", party)
        out_path = EXPORT_BY_PARTY / split_name / safe_party
        out_path.mkdir(parents=True, exist_ok=True)
        group.to_csv(out_path / f"{split_name}.csv", sep="|", index=False)
print(f"Exported by party to {EXPORT_BY_PARTY}")

label_mapping = json.loads((BASE_DIR / "models/camembert_ft/label_mapping.json").read_text())
inv_label = {v: k for k, v in label_mapping.items()}
label_ids = list(label_mapping.values())
label_names = [inv_label[i] for i in label_ids]

y_train = train_df["parti"].map(label_mapping).to_numpy()
y_test = test_df["parti"].map(label_mapping).to_numpy()

def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")

results = []
per_class_f1 = {}
timings = []

print("▶ Baseline: TF-IDF + Logistic Regression")
start = time.time()
tfidf = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2))
X_train = tfidf.fit_transform(train_df["texte"])
X_test = tfidf.transform(test_df["texte"])

logreg = LogisticRegression(
    max_iter=300,
    n_jobs=-1,
    class_weight="balanced",
    random_state=SEED
)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
end = time.time()

results.append(("tfidf_logreg", f1_macro(y_test, y_pred)))
timings.append(("tfidf_logreg", end - start))

report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
per_class_f1["tfidf_logreg"] = [report[str(i)]["f1-score"] for i in label_ids]

joblib.dump(tfidf, OUTPUT_DIR / "tfidf.pkl")
joblib.dump(logreg, OUTPUT_DIR / "logreg.pkl")

print("▶ Baseline: FastText")
start = time.time()

ft_train_file = OUTPUT_DIR / "ft_train.txt"
train_texts = remove_newlines(train_df["texte"])
with ft_train_file.open("w", encoding="utf-8") as f:
    for text, label in tqdm(zip(train_texts, y_train), total=len(y_train), desc="Preparing FastText train"):
        f.write(f"__label__{inv_label[label].replace(' ', '_')} {text}\n")

ft_model = fasttext.train_supervised(
    input=str(ft_train_file),
    epoch=5,
    lr=1.0,
    wordNgrams=2,
    dim=100,
    verbose=2,
    seed=SEED
)

test_texts = remove_newlines(test_df["texte"])
pred_labels_nested, _ = ft_model.predict(test_texts)
pred_labels = [lbls[0] for lbls in pred_labels_nested]
y_pred = [
    label_mapping[label.replace("__label__", "").replace("_", " ")]
    for label in pred_labels
]
end = time.time()

results.append(("fasttext", f1_macro(y_test, y_pred)))
timings.append(("fasttext", end - start))

report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
per_class_f1["fasttext"] = [report[str(i)]["f1-score"] for i in label_ids]

ft_model.save_model(str(OUTPUT_DIR / "fasttext.bin"))

print("▶ CamemBERT Ablation (no prefixes)")
def strip_prefixes(s: pd.Series) -> pd.Series:
    s = s.str.replace(r"\[SENT_[^\]]+\]\s*", "", regex=True)
    return s.str.replace(r"\[(?:IRONY|NOIRONY)\]\s*", "", regex=True)

train_plain = train_df.copy()
train_plain["texte"] = strip_prefixes(train_plain["texte"])
test_plain = test_df.copy()
test_plain["texte"] = strip_prefixes(test_plain["texte"])

tokenizer_ablation = CamembertTokenizer.from_pretrained("camembert-base")
model_ablation = CamembertForSequenceClassification.from_pretrained(
    "camembert-base", num_labels=len(label_mapping)
).to(DEVICE)

def batch_predict(texts, tokenizer, model, batch_size=16):
    preds = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="CamemBERT prediction"):
            enc = tokenizer(
                texts[i : i + batch_size],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(DEVICE)
            logits = model(**enc).logits
            preds.extend(logits.argmax(-1).cpu().tolist())
    return np.array(preds)

start = time.time()
optimizer = torch.optim.AdamW(model_ablation.parameters(), lr=2e-5)
model_ablation.train()
encoded = tokenizer_ablation(
    train_plain["texte"].tolist(),
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
).to(DEVICE)

batch_size = 8
for i in tqdm(range(0, encoded["input_ids"].size(0), batch_size), desc="CamemBERT ablation epoch 1"):
    loss = model_ablation(
        input_ids=encoded["input_ids"][i : i + batch_size],
        attention_mask=encoded["attention_mask"][i : i + batch_size],
        labels=torch.tensor(y_train[i : i + batch_size]).to(DEVICE)
    ).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

y_pred = batch_predict(test_plain["texte"].tolist(), tokenizer_ablation, model_ablation)
end = time.time()

results.append(("camembert_ablation", f1_macro(y_test, y_pred)))
timings.append(("camembert_ablation", end - start))

report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
per_class_f1["camembert_ablation"] = [report[str(i)]["f1-score"] for i in label_ids]

print("▶ CamemBERT fine-tuned full model")
tokenizer_ft = CamembertTokenizer.from_pretrained(MODEL_DIR)
model_ft = CamembertForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)

start = time.time()
y_pred = batch_predict(test_df["texte"].tolist(), tokenizer_ft, model_ft)
end = time.time()

results.append(("camembert_ftV6", f1_macro(y_test, y_pred)))
timings.append(("camembert_ftV6", end - start))

report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
per_class_f1["camembert_ftV6"] = [report[str(i)]["f1-score"] for i in label_ids]

summary_df = pd.DataFrame(results, columns=["model", "f1_macro"]).sort_values("f1_macro", ascending=False)
summary_df.to_csv(OUTPUT_DIR / "baseline_results.csv", index=False)

print("\n=== F1-macro (sorted) ===")
print(summary_df.to_string(index=False))

plt.figure(figsize=(6, 3))
plt.barh(summary_df["model"], summary_df["f1_macro"])
plt.xlabel("F1-macro")
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "baseline_f1_barplot.png", dpi=300)

angles = np.linspace(0, 2 * np.pi, len(label_names), endpoint=False).tolist()
angles += angles[:1]
plt.figure(figsize=(6, 6))
for name, scores in per_class_f1.items():
    vals = scores + scores[:1]
    plt.polar(angles, vals, label=name, linewidth=1.5)
    plt.fill(angles, vals, alpha=0.08)
plt.xticks(angles[:-1], label_names, fontsize=8)
plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], fontsize=7)
plt.ylim(0, 1)
plt.title("F1 by party and model")
plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "radar_f1_by_party.png", dpi=300)

timing_df = pd.DataFrame(timings, columns=["model", "seconds"])
combined_df = summary_df.merge(timing_df, on="model")
plt.figure(figsize=(5, 3))
plt.scatter(combined_df["seconds"], combined_df["f1_macro"])
for _, row in combined_df.iterrows():
    plt.text(row["seconds"], row["f1_macro"] + 0.02, row["model"], fontsize=7)
plt.xlabel("Training time (s)")
plt.ylabel("F1-macro")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "scatter_time_vs_f1.png", dpi=300)

print("All files saved in", OUTPUT_DIR)