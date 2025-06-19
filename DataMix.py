from pathlib import Path
import pandas as pd

BASE_DIR = Path("/mnt/c/Users/Lilian/Desktop")
INPUT_FILE = BASE_DIR / "DataGPSentimentIronie.csv"
OUTPUT_FILE = BASE_DIR / "DataCamemBERT2.csv"
SEPARATOR = "|"
ENCODING = "utf-8"


def filter_and_save():
    df = pd.read_csv(INPUT_FILE, sep=SEPARATOR, encoding=ENCODING)
    required_columns = ["texte", "parti", "media", "ironie", "sentiment"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {INPUT_FILE}: {missing}")

    filtered_df = df[required_columns].copy()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(OUTPUT_FILE, sep=SEPARATOR, encoding=ENCODING, index=False)
    print(f"Filtered columns ({', '.join(required_columns)}) and saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    filter_and_save()