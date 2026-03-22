import os
import re
import pandas as pd

INPUT_CSV = "data/sentence_reviews.csv"
OUTPUT_CSV = "outputs/clean_sentences.csv"


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    os.makedirs("outputs", exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    if "sentence" not in df.columns:
        raise ValueError("Input CSV must contain a 'sentence' column.")

    df["sentence"] = df["sentence"].apply(clean_text)
    df = df[df["sentence"] != ""].copy()

    out_df = df[["sentence"]].drop_duplicates().reset_index(drop=True)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print("Prepared sentences successfully.")
    print("Total sentences:", len(out_df))
    print("Saved to:", OUTPUT_CSV)


if __name__ == "__main__":
    main()