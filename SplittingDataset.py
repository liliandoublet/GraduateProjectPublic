from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# Paths & settings
BASE_DIR = Path("/mnt/c/Users/Lilian/Desktop")
INPUT_FILE = BASE_DIR / "DataCamemBERT.csv"  # Input CSV (delimiter = '|')
OUTPUT_DIR = BASE_DIR / "Data"               # Directory to write the splits

TRAIN_RATIO = 0.8    # Proportion for training set
VALID_RATIO = 0.1    # Proportion for validation set (the rest is test)
RANDOM_STATE = 42    # Seed for reproducibility

def main():
    # Load the source CSV
    df = pd.read_csv(INPUT_FILE, sep="|")

    # Ensure required columns are present
    if not {"parti", "media"}.issubset(df.columns):
        raise ValueError("CSV must contain 'parti' and 'media' columns.")

    # Create combined key for joint stratification
    df["parti_media"] = df["parti"] + "_" + df["media"]

    # --- Split 1: Train vs (Validation + Test) ---
    splitter1 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=1 - TRAIN_RATIO,
        random_state=RANDOM_STATE
    )
    train_idx, temp_idx = next(splitter1.split(df, df["parti_media"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df  = df.iloc[temp_idx].reset_index(drop=True)

    # --- Split 2: Validation vs Test ---
    val_fraction = VALID_RATIO / (1 - TRAIN_RATIO)
    splitter2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=1 - val_fraction,
        random_state=RANDOM_STATE
    )
    val_idx, test_idx = next(splitter2.split(temp_df, temp_df["parti_media"]))
    val_df  = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    # Save the splits
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(OUTPUT_DIR / "train.csv", sep="|", index=False)
    val_df.to_csv(  OUTPUT_DIR / "val.csv",   sep="|", index=False)
    test_df.to_csv( OUTPUT_DIR / "test.csv",  sep="|", index=False)

    print(
        f"✓ Splits completed: {len(train_df)} train / "
        f"{len(val_df)} val / {len(test_df)} test"
    )

if __name__ == "__main__":
    main()
