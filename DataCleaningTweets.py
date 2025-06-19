import re
import emoji
import pandas as pd
from pathlib import Path
from langdetect import detect, LangDetectException

# -----------------------------
# I/O settings
# -----------------------------
INPUT_FILE  = Path("/mnt/c/Users/Lilian/Desktop/all_parties_tweets_Sep.csv")
OUTPUT_FILE = Path("/mnt/c/Users/Lilian/Desktop/tweets_clean_camembert2.csv")

# -----------------------------
# Pre-compiled regular expressions
# -----------------------------
URL_RE     = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w{1,15}")        # 1–15 alphanum / underscore
HASHTAG_RE = re.compile(r"#(\w+)")           # keep the word, strip the '#'
QUOTE_RE   = re.compile(r"[\"«»“”']")

# -----------------------------
# Helper functions
# -----------------------------
def is_french(text: str) -> bool:
    """Return True if the detected language is French."""
    try:
        return detect(text) == "fr"
    except LangDetectException:
        return False


def common_cleaning(text: str) -> str:
    """Shared cleaning steps: emojis, URLs, hashtags, quotes, whitespace, lowercase."""
    if not isinstance(text, str):
        return ""
    text = emoji.replace_emoji(text, "")
    text = URL_RE.sub(" ", text)
    text = HASHTAG_RE.sub(r"\1", text)      # remove '#' but keep the hashtag word
    text = QUOTE_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def clean_with_placeholder(text: str) -> str:
    """Full cleaning and replace @mentions with '__MENTION__'."""
    text = MENTION_RE.sub("__MENTION__", text)
    return common_cleaning(text)


def clean_keep_mentions(text: str) -> str:
    """Clean text while preserving original @handles."""
    return common_cleaning(text)


# -----------------------------
# Main pipeline
# -----------------------------
if __name__ == "__main__":
    print("Reading CSV…")
    df = pd.read_csv(INPUT_FILE, sep="|", dtype=str, encoding="utf-8")

    # 1. Drop retweets
    df = df[~df["texte"].str.startswith("rt @", na=False)]

    # 2. Keep only French tweets
    df = df[df["texte"].apply(is_french)]

    # 3. Generate cleaned columns
    df["clean_text"]          = df["texte"].apply(clean_with_placeholder)
    df["clean_text_mentions"] = df["texte"].apply(clean_keep_mentions)

    # 4. Filter out very short tweets (<3 words)
    df = df[df["clean_text"].str.split().apply(len) >= 3]

    # 5. Remove duplicates
    df = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)

    # 6. Reorder columns
    remaining_cols = [c for c in df.columns if c not in ("texte", "clean_text", "clean_text_mentions")]
    df = df[["texte", "clean_text", "clean_text_mentions", *remaining_cols]]

    # 7. Export
    df.to_csv(OUTPUT_FILE, sep="|", index=False, encoding="utf-8")
    print(f"✅ Export complete → {OUTPUT_FILE}  ({len(df)} tweets)")