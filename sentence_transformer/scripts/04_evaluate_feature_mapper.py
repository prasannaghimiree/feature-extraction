
import os
import re
import json
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer, util

MODEL_DIR = "models/span_contrastive_feature_mapper"
DATA_PATH = "data/processed/feature_train_full.csv"
OUTPUT_DIR = "outputs/feature_mapper_evaluation"

TEST_SIZE = 0.15
SEED = 42


def clean_text(x):
    if pd.isna(x):
        return ""
    x = str(x)
    x = x.replace("\u2019", "'").replace("\u2018", "'")
    x = x.replace("\u201c", '"').replace("\u201d", '"')
    x = re.sub(r"\s+", " ", x).strip().lower()
    return x


def build_span_query(raw_aspect, sentence):
    return f"aspect span: {clean_text(raw_aspect)} [SEP] review context: {clean_text(sentence)}"


def build_feature_text(label):
    return f"product feature: {clean_text(label).replace('_', ' ')}"


def topk_accuracy(y_true, scores, labels, k):
    correct = 0
    for true_label, score_row in zip(y_true, scores):
        top_ids = np.argsort(score_row)[::-1][:k]
        top_labels = [labels[i] for i in top_ids]
        if true_label in top_labels:
            correct += 1
    return correct / len(y_true)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    for col in ["raw_aspect", "sentence", "label"]:
        df[col] = df[col].apply(clean_text)

    df = df[
        (df["raw_aspect"] != "") &
        (df["sentence"] != "") &
        (df["label"] != "")
    ].drop_duplicates().reset_index(drop=True)

    with open(os.path.join(MODEL_DIR, "labels.json"), "r", encoding="utf-8") as f:
        labels = json.load(f)

    df = df[df["label"].isin(labels)].reset_index(drop=True)

    # fixed random test split for quick professor update
    test_df = df.sample(frac=TEST_SIZE, random_state=SEED).reset_index(drop=True)

    model = SentenceTransformer(MODEL_DIR)

    queries = [
        build_span_query(row["raw_aspect"], row["sentence"])
        for _, row in test_df.iterrows()
    ]

    label_texts = [build_feature_text(label) for label in labels]

    query_emb = model.encode(
        queries,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    label_emb = model.encode(
        label_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    scores = util.cos_sim(query_emb, label_emb).cpu().numpy()
    pred_ids = scores.argmax(axis=1)
    y_pred = [labels[i] for i in pred_ids]
    y_true = test_df["label"].tolist()

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "top3_accuracy": topk_accuracy(y_true, scores, labels, 3),
        "top5_accuracy": topk_accuracy(y_true, scores, labels, 5),
        "num_test_samples": len(test_df),
        "num_labels": len(labels)
    }

    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0
    )

    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        os.path.join(OUTPUT_DIR, "confusion_matrix.csv")
    )

    test_df["prediction"] = y_pred
    test_df["correct"] = test_df["label"] == test_df["prediction"]
    test_df["confidence"] = scores.max(axis=1)

    test_df.to_csv(
        os.path.join(OUTPUT_DIR, "test_predictions.csv"),
        index=False
    )

    print("\n===== FEATURE MAPPER EVALUATION =====")
    for k, v in results.items():
        print(f"{k}: {v}")

    print("\nSaved outputs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()