import json
import re
import string
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from lime.lime_text import LimeTextExplainer
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from transformers import (
    CamembertTokenizer,
    CamembertForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from wordcloud import WordCloud

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path("/mnt/c/Users/Lilian/Desktop/Data")
DATA_DIR = BASE_DIR
OUT_DIR = BASE_DIR / "models/camembert_ftV3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 4
BATCH_SIZE = 16
LR = 2e-5
SEED = 92
USE_META = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device in use:", device)
torch.manual_seed(SEED)

def add_meta(df: pd.DataFrame) -> pd.DataFrame:
    """Add [SENT_*] and [IRONY] prefixes when the columns exist."""
    if not USE_META:
        return df
    if "sentiment" in df:
        df["texte"] = "[SENT_" + df["sentiment"].str.lower().str.strip() + "] " + df["texte"]
    if "ironie" in df:
        df["texte"] = np.where(
            df["ironie"].astype(bool),
            "[IRONY] " + df["texte"],
            "[NOIRONY] " + df["texte"]
        )
    return df

def load_split(name: str) -> pd.DataFrame:
    return add_meta(pd.read_csv(DATA_DIR / f"{name}.csv", sep="|"))

def encode_labels(df: pd.DataFrame, le: LabelEncoder | None = None, mapping_path: Path | None = None):
    if le is None:
        le = LabelEncoder().fit(df["parti"])
    df["label"] = le.transform(df["parti"])
    if mapping_path:
        mapping = {cls: int(idx) for idx, cls in enumerate(le.classes_)}
        mapping_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    return df, le

def df_to_hf(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df[["texte", "label"]].reset_index(drop=True))

df_train, df_val, df_test = map(load_split, ["train", "val", "test"])
df_train, le = encode_labels(df_train, mapping_path=OUT_DIR / "label_mapping.json")
df_val, _ = encode_labels(df_val, le=le)
df_test, _ = encode_labels(df_test, le=le)
num_labels = len(le.classes_)

ds = DatasetDict({
    "train": df_to_hf(df_train),
    "validation": df_to_hf(df_val),
    "test": df_to_hf(df_test),
})

tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def tokenize_fn(batch):
    return tokenizer(batch["texte"], truncation=True, padding=True, max_length=512)

ds = ds.map(tokenize_fn, batched=True)
ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

model = CamembertForSequenceClassification.from_pretrained("camembert-base", num_labels=num_labels).to(device)

training_args = TrainingArguments(
    output_dir=str(OUT_DIR / "checkpoints"),
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    seed=SEED,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    report_to="none",
    fp16=True,
)

def compute_metrics(pred):
    y_true = pred.label_ids
    y_pred = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

preds = trainer.predict(ds["test"])
y_pred = np.argmax(preds.predictions, axis=1)
y_true = preds.label_ids

df_eval = df_test.reset_index(drop=True).copy()
df_eval["true_label"] = le.inverse_transform(y_true)
df_eval["pred_label"] = le.inverse_transform(y_pred)

rows_sample = []
for party in le.classes_:
    tp = df_eval[(df_eval.true_label == party) & (df_eval.pred_label == party)]
    fp = df_eval[(df_eval.true_label != party) & (df_eval.pred_label == party)]
    rows_sample.extend(tp.sample(n=min(2, len(tp)), random_state=SEED).to_dict("records"))
    rows_sample.extend(fp.sample(n=min(2, len(fp)), random_state=SEED).to_dict("records"))

pd.DataFrame(rows_sample).to_csv(OUT_DIR / "samples_20_tp_fp.csv", sep="|", index=False)
print("20 TP/FP samples saved to samples_20_tp_fp.csv")

overall_metrics = trainer.evaluate(ds["test"])
pd.DataFrame([overall_metrics]).to_csv(OUT_DIR / "metrics_overall.csv", index=False)

report_dict = classification_report(y_true, y_pred, target_names=le.classes_, output_dict=True)
pd.DataFrame(report_dict).T.to_csv(OUT_DIR / "classification_report_test.csv")

cm_abs = confusion_matrix(y_true, y_pred)
cm_pct = confusion_matrix(y_true, y_pred, normalize="true")

plt.figure(figsize=(6, 6))
ConfusionMatrixDisplay(cm_abs, display_labels=le.classes_).plot(cmap="Blues", xticks_rotation=45, colorbar=False, ax=plt.gca())
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrix_abs.png", dpi=300)
plt.close()

plt.figure(figsize=(6, 6))
ConfusionMatrixDisplay(cm_pct, display_labels=le.classes_).plot(cmap="Oranges", xticks_rotation=45, colorbar=True, ax=plt.gca())
plt.title("Confusion Matrix – Percentage")
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrix_pct.png", dpi=300)
plt.close()

media_summary = []
if "media" in df_eval.columns:
    for media_type, subset in df_eval.groupby("media"):
        y_t = subset["label"].values
        y_p = le.transform(subset["pred_label"])
        rep = classification_report(y_t, y_p, target_names=le.classes_, output_dict=True)
        pd.DataFrame(rep).T.to_csv(OUT_DIR / f"classification_report_{media_type}.csv")
        media_summary.append({
            "media": media_type,
            "accuracy": accuracy_score(y_t, y_p),
            "precision_macro": precision_score(y_t, y_p, average="macro"),
            "recall_macro": recall_score(y_t, y_p, average="macro"),
            "f1_macro": f1_score(y_t, y_p, average="macro"),
        })
    pd.DataFrame(media_summary).to_csv(OUT_DIR / "metrics_by_media.csv", index=False)
else:
    print("Column 'media' not found: skipping media-level metrics export")

META_TAG_RGX = re.compile(r"\[(?:SENT_[^\]]+|NO?IRONY)\]\s*", flags=re.I)

def tokenize(txt: str):
    txt = META_TAG_RGX.sub("", txt.lower())
    txt = re.sub(rf"[{re.escape(string.punctuation)}0-9]", " ", txt)
    return [w for w in txt.split() if len(w) > 2]

freq_by_party = {}
for party, group in df_train.groupby("parti"):
    corpus = " ".join(group["texte"])
    freq_by_party[party] = Counter(tokenize(corpus)).most_common(100)

with pd.ExcelWriter(OUT_DIR / "top_words_by_party.xlsx") as writer:
    for party, pairs in freq_by_party.items():
        pd.DataFrame(pairs, columns=["word", "count"]).to_excel(writer, sheet_name=party[:30], index=False)

print("top_words_by_party.xlsx generated")

wc_dir = OUT_DIR / "wordclouds"
wc_dir.mkdir(exist_ok=True)

if "media" in df_train.columns:
    for media_type, subset in df_train.groupby("media"):
        freqs = Counter(tokenize(" ".join(subset["texte"]))).most_common(300)
        wc = WordCloud(width=1000, height=500, background_color="white").generate_from_frequencies(dict(freqs))
        plt.figure(figsize=(12, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(wc_dir / f"wordcloud_{media_type}.png", dpi=300)
        plt.close()
        print(f"WordCloud {media_type} saved to {wc_dir / f'wordcloud_{media_type}.png'}")

for party, pairs in freq_by_party.items():
    wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(dict(pairs))
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(wc_dir / f"{party}.png", dpi=300)
    plt.close()
    print(f"WordCloud for {party} saved to {wc_dir / f'{party}.png'}")

print(f"All WordClouds generated in {wc_dir}")

miscls = np.where(y_pred != y_true)[0]
sample_idx = miscls[:20] if len(miscls) >= 20 else miscls

explainer = LimeTextExplainer(class_names=list(le.classes_))

def predict_proba(texts):
    model.eval()
    all_probs = []
    batch_size = 8
    for start in range(0, len(texts), batch_size):
        sub = texts[start:start + batch_size]
        enc = tokenizer(sub, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        del enc, logits
        torch.cuda.empty_cache()
    return np.vstack(all_probs)

for i, idx in enumerate(sample_idx):
    exp = explainer.explain_instance(df_test.iloc[idx]["texte"], predict_proba, num_features=10, num_samples=1000)
    fig = exp.as_pyplot_figure()
    out_path = OUT_DIR / f"lime_example_{i}_idx{idx}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"LIME example {i} (idx={idx}) saved as {out_path.name}")

example_txt = df_test.iloc[sample_idx[0]]["texte"]
bg_texts = df_train["texte"].sample(100, random_state=SEED).tolist()

def predict_proba_shap(texts):
    if isinstance(texts, np.ndarray):
        texts = texts.astype(str).tolist()
    elif isinstance(texts, str):
        texts = [texts]
    enc = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    return torch.softmax(logits, dim=1).cpu().numpy()

masker = shap.maskers.Text(tokenizer)
explainer_shap = shap.Explainer(predict_proba_shap, masker, output_names=list(le.classes_))
shap_values = explainer_shap([example_txt])
shap.plots.text(shap_values[0], display=False)
plt.savefig(OUT_DIR / "shap_example.png", dpi=300, bbox_inches="tight")
plt.close()
print("shap_example.png saved")

model.config.id2label = {i: lbl for i, lbl in enumerate(le.classes_)}
model.config.label2id = {lbl: i for i, lbl in model.config.id2label.items()}
model.save_pretrained(OUT_DIR / "model")
tokenizer.save_pretrained(OUT_DIR / "model")
print(f"Model and tokenizer saved in {OUT_DIR / 'model'}")