import os
import re
import json
import pandas as pd

INPUT_CSV = "data/raw/aspect_semantic_dataset_canonicalized.csv"
OUTPUT_DIR = "data/processed"


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def to_bool(x):
    return str(x).strip().lower() == "true"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required_cols = [
        "sentence",
        "raw_aspect",
        "sentiment",
        "has_feature"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Clean
    df["sentence"] = df["sentence"].apply(clean_text)
    df["raw_aspect"] = df["raw_aspect"].apply(clean_text)
    df["sentiment"] = df["sentiment"].apply(clean_text)
    df["has_feature"] = df["has_feature"].apply(to_bool)

    # Keep valid rows
    df = df[
        (df["has_feature"] == True) &
        (df["sentence"] != "") &
        (df["raw_aspect"] != "") &
        (df["sentiment"] != "")
    ].copy()

    if len(df) == 0:
        raise ValueError("No valid rows left after filtering.")

    # Only allow standard sentiment labels
    allowed = {"positive", "neutral", "negative"}
    df = df[df["sentiment"].isin(allowed)].copy()

    # Remove duplicates
    df = df.drop_duplicates(
        subset=["sentence", "raw_aspect", "sentiment"]
    ).reset_index(drop=True)

    # Fixed sentiment encoding
    label2id = {
        "negative": 0,
        "neutral": 1,
        "positive": 2
    }
    id2label = {v: k for k, v in label2id.items()}

    df["label"] = df["sentiment"]
    df["label_id"] = df["label"].map(label2id)

    final_df = df[["raw_aspect", "sentence", "label", "label_id"]].copy()

    # Save outputs
    out_csv = os.path.join(OUTPUT_DIR, "sentiment_train_full.csv")
    out_json = os.path.join(OUTPUT_DIR, "sentiment_label_map.json")
    out_dist = os.path.join(OUTPUT_DIR, "sentiment_label_distribution.csv")

    final_df.to_csv(out_csv, index=False)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "label2id": label2id,
                "id2label": {str(k): v for k, v in id2label.items()}
            },
            f,
            indent=2,
            ensure_ascii=False
        )

    label_dist = final_df["label"].value_counts().reset_index()
    label_dist.columns = ["label", "count"]
    label_dist.to_csv(out_dist, index=False)

    print("Sentiment dataset prepared successfully.")
    print(f"Total usable rows: {len(final_df)}")
    print(f"Total sentiment labels: {len(label2id)}")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_json}")
    print(f"Saved: {out_dist}")
    print("\nSentiment label distribution:")
    print(final_df["label"].value_counts())


if __name__ == "__main__":
    main()