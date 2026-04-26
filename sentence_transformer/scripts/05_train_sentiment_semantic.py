# 05_train_sentiment_contrastive.py
import os
import re
import json
import random
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sentence_transformers import SentenceTransformer, InputExample, losses, util

DATA_PATH = "data/processed/sentiment_train_full.csv"
OUTPUT_DIR = "models/span_contrastive_sentiment_mapper"

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 42
VAL_SIZE = 0.15
BATCH_SIZE = 16
EPOCHS = 4
LR = 2e-5

SENTIMENT_DESCRIPTIONS = {
    "positive": "positive sentiment: praise, satisfaction, good experience, useful feature",
    "neutral": "neutral sentiment: factual, mixed, unclear, average, no strong opinion",
    "negative": "negative sentiment: complaint, dissatisfaction, poor quality, bad experience"
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_text(x):
    if pd.isna(x):
        return ""
    x = str(x)
    x = x.replace("\u2019", "'").replace("\u2018", "'")
    x = x.replace("\u201c", '"').replace("\u201d", '"')
    x = re.sub(r"\s+", " ", x).strip().lower()
    return x


def build_query(raw_aspect, sentence):
    return f"aspect span: {clean_text(raw_aspect)} [SEP] review context: {clean_text(sentence)}"


def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    for col in ["raw_aspect", "sentence", "label"]:
        df[col] = df[col].apply(clean_text)

    df = df[df["label"].isin(["positive", "neutral", "negative"])].copy()
    df = df[
        (df["raw_aspect"] != "") &
        (df["sentence"] != "") &
        (df["label"] != "")
    ].drop_duplicates().reset_index(drop=True)

    train_df, val_df = train_test_split(
        df,
        test_size=VAL_SIZE,
        random_state=SEED,
        stratify=df["label"]
    )

    train_examples = []
    for _, row in train_df.iterrows():
        query = build_query(row["raw_aspect"], row["sentence"])
        positive = SENTIMENT_DESCRIPTIONS[row["label"]]
        train_examples.append(InputExample(texts=[query, positive]))

    model = SentenceTransformer(BASE_MODEL)

    loader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    loss = losses.MultipleNegativesRankingLoss(model)

    labels = ["negative", "neutral", "positive"]
    prototypes = [SENTIMENT_DESCRIPTIONS[label] for label in labels]

    for epoch in range(1, EPOCHS + 1):
        model.fit(
            train_objectives=[(loader, loss)],
            epochs=1,
            warmup_steps=int(len(loader) * 0.1),
            optimizer_params={"lr": LR},
            output_path=OUTPUT_DIR,
            show_progress_bar=True
        )

        queries = [build_query(r["raw_aspect"], r["sentence"]) for _, r in val_df.iterrows()]

        q_emb = model.encode(queries, convert_to_tensor=True, normalize_embeddings=True)
        p_emb = model.encode(prototypes, convert_to_tensor=True, normalize_embeddings=True)

        scores = util.cos_sim(q_emb, p_emb).cpu().numpy()
        pred_ids = scores.argmax(axis=1)
        y_pred = [labels[i] for i in pred_ids]
        y_true = val_df["label"].tolist()

        acc = accuracy_score(y_true, y_pred)
        macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        print(f"\nEpoch {epoch}")
        print(f"Validation Accuracy: {acc:.4f}")
        print(f"Validation Macro F1: {macro:.4f}")
        print(f"Validation Weighted F1: {weighted:.4f}")

    model.save(OUTPUT_DIR)

    with open(os.path.join(OUTPUT_DIR, "sentiment_prototypes.json"), "w", encoding="utf-8") as f:
        json.dump(SENTIMENT_DESCRIPTIONS, f, indent=2)

    print("\nSaved sentiment contrastive model:", OUTPUT_DIR)


if __name__ == "__main__":
    main()