import os
import re
import json
import random
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sentence_transformers import SentenceTransformer, InputExample, losses, util

DATA_PATH = "data/processed/feature_train_full.csv"
OUTPUT_DIR = "models/span_contrastive_feature_mapper"

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 42
VAL_SIZE = 0.15
BATCH_SIZE = 16
EPOCHS = 4
LR = 2e-5
MIN_CLASS_COUNT = 3


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


def build_span_query(raw_aspect, sentence):
    raw_aspect = clean_text(raw_aspect)
    sentence = clean_text(sentence)
    return f"aspect span: {raw_aspect} [SEP] review context: {sentence}"


def build_feature_text(label):
    label = clean_text(label).replace("_", " ")
    return f"product feature: {label}"


def topk_accuracy(y_true, score_matrix, labels, k=3):
    label_to_idx = {l: i for i, l in enumerate(labels)}
    correct = 0

    for true_label, scores in zip(y_true, score_matrix):
        top_ids = np.argsort(scores)[::-1][:k]
        top_labels = [labels[i] for i in top_ids]
        if true_label in top_labels:
            correct += 1

    return correct / len(y_true)


def predict_by_retrieval(model, df, labels):
    queries = [
        build_span_query(row["raw_aspect"], row["sentence"])
        for _, row in df.iterrows()
    ]

    label_texts = [build_feature_text(label) for label in labels]

    query_emb = model.encode(
        queries,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    label_emb = model.encode(
        label_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    scores = util.cos_sim(query_emb, label_emb).cpu().numpy()
    pred_ids = scores.argmax(axis=1)
    preds = [labels[i] for i in pred_ids]

    return preds, scores


def evaluate(model, df, labels):
    y_true = df["label"].tolist()
    y_pred, scores = predict_by_retrieval(model, df, labels)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "top3_accuracy": topk_accuracy(y_true, scores, labels, k=3),
        "top5_accuracy": topk_accuracy(y_true, scores, labels, k=5),
    }


def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    required = ["raw_aspect", "sentence", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for col in required:
        df[col] = df[col].apply(clean_text)

    df = df[
        (df["raw_aspect"] != "") &
        (df["sentence"] != "") &
        (df["label"] != "")
    ].drop_duplicates().reset_index(drop=True)

    counts = df["label"].value_counts()
    valid_labels = counts[counts >= MIN_CLASS_COUNT].index.tolist()
    df = df[df["label"].isin(valid_labels)].reset_index(drop=True)

    labels = sorted(df["label"].unique().tolist())

    train_df, val_df = train_test_split(
        df,
        test_size=VAL_SIZE,
        random_state=SEED,
        stratify=df["label"]
    )

    train_examples = []
    for _, row in train_df.iterrows():
        query = build_span_query(row["raw_aspect"], row["sentence"])
        positive = build_feature_text(row["label"])
        train_examples.append(InputExample(texts=[query, positive]))

    model = SentenceTransformer(BASE_MODEL)

    train_loader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=BATCH_SIZE
    )

    loss = losses.MultipleNegativesRankingLoss(model)

    history = []

    for epoch in range(1, EPOCHS + 1):
        model.fit(
            train_objectives=[(train_loader, loss)],
            epochs=1,
            warmup_steps=int(len(train_loader) * 0.1),
            optimizer_params={"lr": LR},
            show_progress_bar=True,
            output_path=OUTPUT_DIR
        )

        train_metrics = evaluate(model, train_df, labels)
        val_metrics = evaluate(model, val_df, labels)

        row = {
            "epoch": epoch,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_weighted_f1": train_metrics["weighted_f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "val_top3_accuracy": val_metrics["top3_accuracy"],
            "val_top5_accuracy": val_metrics["top5_accuracy"],
        }

        history.append(row)

        print(f"\nEpoch {epoch}")
        print(f"Train Accuracy      : {row['train_accuracy']:.4f}")
        print(f"Train Macro F1      : {row['train_macro_f1']:.4f}")
        print(f"Validation Accuracy : {row['val_accuracy']:.4f}")
        print(f"Validation Macro F1 : {row['val_macro_f1']:.4f}")
        print(f"Validation Top-3 Acc: {row['val_top3_accuracy']:.4f}")

    pd.DataFrame(history).to_csv(
        os.path.join(OUTPUT_DIR, "training_history.csv"),
        index=False
    )

    with open(os.path.join(OUTPUT_DIR, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

    model.save(OUTPUT_DIR)

    print("\nSaved model to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()