import os
import json
import time
from pathlib import Path
from typing import List

import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from tenacity import retry, wait_fixed, stop_after_attempt
from tqdm import tqdm

BASE_DIR = Path("/mnt/c/Users/Lilian/Desktop")
INPUT_FILE = BASE_DIR / "ConcaSWVFinalUtf8.csv"
OUTPUT_FILE = BASE_DIR / "ConcaSWVFinal_cleanV2.csv"
DEBUG_DIR = BASE_DIR / "debug"

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()
API_KEY = os.environ["GEMINI_API_KEY"]

genai.configure(api_key=API_KEY)
MODEL_ID = "gemini-2.0-flash"
model = genai.GenerativeModel(MODEL_ID)

TARGET_COLUMN = "texte"

SYSTEM_PROMPT = (
    "You are a text-cleaning assistant.\n"
    "Rules:\n"
    "1) Remove newlines (replace with space).\n"
    "2) Remove hashtags, emojis, and @mentions.\n"
    "3) Remove photo credits, author names, links, and dates.\n"
    "4) Preserve overall meaning without stylistic rewriting.\n"
    "5) Do not censor content.\n"
    "6) If text contains only lists or metadata, reply: 'Text not relevant for analysis.'\n"
    "7) Remove political names and roles lists at start/end.\n"
    "8) Remove long repetitive signatures.\n"
    "9) Remove generic promotional phrases.\n"
    "Respond only with cleaned text, no comments or intros."
)

def save_json(obj: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

@retry(wait=wait_fixed(2), stop=stop_after_attempt(5))
def clean_line(text: str) -> str:
    prompt = f"{SYSTEM_PROMPT}\n\nTEXT TO CLEAN:\n{text}"
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.2},
    )
    return response.text.strip()


def main() -> None:
    print(f"Reading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, sep="|")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Column '{TARGET_COLUMN}' missing in {INPUT_FILE.name}")

    texts: List[str] = df[TARGET_COLUMN].fillna("").astype(str).tolist()
    cleaned_all: List[str] = []

    print(f"Cleaning {len(texts)} lines with Gemini...")
    for idx, line in enumerate(tqdm(texts, ncols=80), start=1):
        try:
            cleaned = clean_line(line)
        except Exception as e:
            print(f"[ERROR] Line {idx} -> {e}")
            cleaned = "<ERROR>"
        cleaned_all.append(cleaned)

        if idx == 1 or idx % 100 == 0:
            save_json({"input": line, "output": cleaned}, DEBUG_DIR / f"line_{idx:06d}.json")
        if idx % 9 == 0:
            time.sleep(20)

    df[f"{TARGET_COLUMN}_clean"] = cleaned_all
    df.to_csv(OUTPUT_FILE, sep='|', index=False)
    print(f"CSV written: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
