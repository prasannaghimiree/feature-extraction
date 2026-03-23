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

    required_cols = ["sentence", "raw_aspect", "final_canonical_feature", "has_feature"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Clean
    df["sentence"] = df["sentence"].apply(clean_text)
    df["raw_aspect"] = df["raw_aspect"].apply(clean_text)
    df["final_canonical_feature"] = df["final_canonical_feature"].apply(clean_text)
    df["has_feature"] = df["has_feature"].apply(to_bool)

    # Keeping valid rows only
    df = df[
        (df["has_feature"] == True) &
        (df["sentence"] != "") &
        (df["raw_aspect"] != "") &
        (df["final_canonical_feature"] != "")
    ].copy()

    if len(df) == 0:
        raise ValueError("No valid rows left after filtering.")

    # Removing duplicates
    df = df.drop_duplicates(
        subset=["sentence", "raw_aspect", "final_canonical_feature"]
    ).reset_index(drop=True)

    # Creating labels
    labels = sorted(df["final_canonical_feature"].unique())
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    df["label"] = df["final_canonical_feature"]
    df["label_id"] = df["label"].map(label2id)

    final_df = df[["raw_aspect", "sentence", "label", "label_id"]].copy()

    # Savinng outputs
    out_csv = os.path.join(OUTPUT_DIR, "feature_train_full.csv")
    out_json = os.path.join(OUTPUT_DIR, "feature_label_map.json")
    out_dist = os.path.join(OUTPUT_DIR, "feature_label_distribution.csv")

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

    print("Feature dataset prepared successfully.")
    print(f"Total usable rows: {len(final_df)}")
    print(f"Total feature labels: {len(labels)}")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_json}")
    print(f"Saved: {out_dist}")
    print("\nFeature label distribution:")
    print(final_df["label"].value_counts())


if __name__ == "__main__":
    main()