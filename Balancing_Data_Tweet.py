from pathlib import Path
import pandas as pd

input_path = Path("/mnt/c/Users/Lilian/Desktop/tweets_clean_choixCamemBERT-5mts.csv")
output_path = Path("/mnt/c/Users/Lilian/Desktop/tweets_clean_1392perParty.csv")

df = pd.read_csv(input_path, sep="|", dtype=str, encoding="utf-8")

df = df.rename(columns={"parti": "party", "texte": "text"})

required_columns = {"party", "date", "text"}
if not required_columns.issubset(df.columns):
    missing = required_columns - set(df.columns)
    raise ValueError(f"Missing columns: {missing}")

df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
df = df.dropna(subset=["date"])

df["mention_count"] = df["text"].str.count("__mention__")

df_sorted = df.sort_values(
    ["party", "mention_count", "date"],
    ascending=[True, True, False]
)

df_trimmed = (
    df_sorted
    .groupby("party", group_keys=False)
    .head(1392)
    .reset_index(drop=True)
)

df_trimmed = df_trimmed.drop(columns="mention_count")

print(df_trimmed["party"].value_counts())

df_trimmed.to_csv(output_path, sep="|", index=False, encoding="utf-8")

print(f"Export completed → {output_path} ({len(df_trimmed)} tweets)")
