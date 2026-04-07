import os
import re
import json
import random
import shutil
import itertools
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader


# PATHS
DATA_PATH = "data/processed/feature_train_full.csv"
LABEL_MAP_PATH = "data/processed/feature_label_map.json"
OUTPUT_DIR = "models/feature_mapper_semantic_tuned"
BEST_MODEL_DIR = os.path.join(OUTPUT_DIR, "best_model")
BEST_TRIAL_DIR = os.path.join(OUTPUT_DIR, "best_trial_model")

# FIXED SETTINGS
SEED = 42
VAL_SIZE = 0.1
MIN_SAMPLES_PER_CLASS = 4
EARLY_STOPPING_PATIENCE = 3


# HYPERPARAMETER SEARCH SPACE
MODEL_CANDIDATES = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
]

BATCH_SIZE_CANDIDATES = [8, 16, 32]
LR_CANDIDATES = [2e-5, 3e-5, 5e-5]
WEIGHT_DECAY_CANDIDATES = [0.0, 0.01, 0.05]
EPOCH_CANDIDATES = [10, 15]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def build_input_text(raw_aspect, sentence):
    raw_aspect = clean_text(raw_aspect)
    sentence = clean_text(sentence)
    return f"aspect: {raw_aspect} [SEP] sentence: {sentence}"


def load_label_map(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Label map not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    label2id = data["label2id"]
    id2label = {int(k): v for k, v in data["id2label"].items()}
    return label2id, id2label


def build_examples(dataframe):
    examples = []
    for _, row in dataframe.iterrows():
        input_text = build_input_text(row["raw_aspect"], row["sentence"])
        target_text = clean_text(row["label"])
        examples.append(InputExample(texts=[input_text, target_text]))
    return examples


def compute_predictions(model, df, labels):
    if len(df) == 0:
        return [], []

    texts = [
        build_input_text(row["raw_aspect"], row["sentence"])
        for _, row in df.iterrows()
    ]
    true_labels = df["label"].tolist()

    text_embeddings = model.encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    label_embeddings = model.encode(
        labels,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    similarities = util.cos_sim(text_embeddings, label_embeddings)
    pred_ids = torch.argmax(similarities, dim=1).cpu().numpy()
    pred_labels = [labels[i] for i in pred_ids]

    return true_labels, pred_labels


def evaluate_metrics(model, df, labels):
    if len(df) == 0:
        return 0.0, 0.0, 0.0

    true_labels, pred_labels = compute_predictions(model, df, labels)

    accuracy = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro")
    weighted_f1 = f1_score(true_labels, pred_labels, average="weighted")

    return float(accuracy), float(macro_f1), float(weighted_f1)


def save_model_dir(src_dir, dst_dir):
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)


def plot_training_history(history_df, output_dir, prefix=""):
    if history_df.empty:
        return

    acc_path = os.path.join(output_dir, f"{prefix}train_vs_val_accuracy.png")
    f1_path = os.path.join(output_dir, f"{prefix}val_f1_curve.png")

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_accuracy"], marker="o", label="Train Accuracy")
    plt.plot(history_df["epoch"], history_df["val_accuracy"], marker="o", label="Validation Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(acc_path, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["val_macro_f1"], marker="o", label="Macro F1")
    plt.plot(history_df["epoch"], history_df["val_weighted_f1"], marker="o", label="Weighted F1")
    plt.title("Validation F1 Scores")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f1_path, dpi=200)
    plt.close()


def load_and_prepare_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Feature dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    _, _ = load_label_map(LABEL_MAP_PATH)

    required_cols = ["raw_aspect", "sentence", "label", "label_id"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    df["raw_aspect"] = df["raw_aspect"].apply(clean_text)
    df["sentence"] = df["sentence"].apply(clean_text)
    df["label"] = df["label"].apply(clean_text)

    df = df[
        (df["raw_aspect"] != "") &
        (df["sentence"] != "") &
        (df["label"] != "")
    ].copy()

    if len(df) == 0:
        raise ValueError("No valid rows found in dataset.")

    print("Original dataset size:", len(df))
    print("Original label distribution:")
    print(df["label"].value_counts())

    label_counts = df["label_id"].value_counts()

    removed_counts = label_counts[label_counts < MIN_SAMPLES_PER_CLASS]
    if len(removed_counts) > 0:
        print(f"\nRemoving labels with fewer than {MIN_SAMPLES_PER_CLASS} samples:")
        print(removed_counts)

    valid_label_ids = label_counts[label_counts >= MIN_SAMPLES_PER_CLASS].index
    df = df[df["label_id"].isin(valid_label_ids)].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No data left after removing rare labels.")

    print("\nFiltered dataset size:", len(df))
    print("Filtered label distribution:")
    print(df["label"].value_counts())

    if df["label_id"].nunique() < 2:
        raise ValueError("Need at least 2 classes after filtering rare labels.")

    train_df, val_df = train_test_split(
        df,
        test_size=VAL_SIZE,
        stratify=df["label_id"],
        random_state=SEED
    )

    print("\nTrain size:", len(train_df))
    print("Validation size:", len(val_df))

    labels = sorted(df["label"].unique().tolist())

    return train_df, val_df, labels


def run_single_trial(trial_id, model_name, batch_size, lr, weight_decay, epochs, train_df, val_df, labels):
    trial_dir = os.path.join(OUTPUT_DIR, f"trial_{trial_id}")
    os.makedirs(trial_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"Trial {trial_id}")
    print(f"Model        : {model_name}")
    print(f"Batch size   : {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Weight decay : {weight_decay}")
    print(f"Epochs       : {epochs}")
    print("=" * 80)

    train_examples = build_examples(train_df)
    if len(train_examples) == 0:
        raise ValueError("No training examples created.")

    model = SentenceTransformer(model_name)

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size
    )

    train_loss_obj = losses.MultipleNegativesRankingLoss(model)
    warmup_steps = int(len(train_dataloader) * epochs * 0.1)

    history = []
    best_val_weighted_f1 = -1.0
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.fit(
            train_objectives=[(train_dataloader, train_loss_obj)],
            epochs=1,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": lr, "weight_decay": weight_decay},
            output_path=trial_dir,
            show_progress_bar=True
        )

        train_accuracy, train_macro_f1, train_weighted_f1 = evaluate_metrics(model, train_df, labels)
        val_accuracy, val_macro_f1, val_weighted_f1 = evaluate_metrics(model, val_df, labels)

        history.append({
            "epoch": epoch,
            "train_accuracy": round(train_accuracy, 6),
            "train_macro_f1": round(train_macro_f1, 6),
            "train_weighted_f1": round(train_weighted_f1, 6),
            "val_accuracy": round(val_accuracy, 6),
            "val_macro_f1": round(val_macro_f1, 6),
            "val_weighted_f1": round(val_weighted_f1, 6)
        })

        print(f"Epoch {epoch}")
        print(f"Train Accuracy        : {train_accuracy:.4f}")
        print(f"Train Macro F1        : {train_macro_f1:.4f}")
        print(f"Train Weighted F1     : {train_weighted_f1:.4f}")
        print(f"Validation Accuracy   : {val_accuracy:.4f}")
        print(f"Validation Macro F1   : {val_macro_f1:.4f}")
        print(f"Validation Weighted F1: {val_weighted_f1:.4f}")
        print("-" * 60)

        if val_weighted_f1 > best_val_weighted_f1:
            best_val_weighted_f1 = val_weighted_f1
            best_epoch = epoch
            patience_counter = 0
            save_model_dir(trial_dir, os.path.join(trial_dir, "best_epoch_model"))
            print(f"Best epoch updated at epoch {epoch}")
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    history_df = pd.DataFrame(history)
    history_csv_path = os.path.join(trial_dir, "training_history.csv")
    history_df.to_csv(history_csv_path, index=False)

    plot_training_history(history_df, trial_dir)

    best_epoch_row = history_df.loc[history_df["val_weighted_f1"].idxmax()]

    result = {
        "trial_id": trial_id,
        "model_name": model_name,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "epochs_requested": epochs,
        "epochs_completed": int(history_df["epoch"].max()),
        "best_epoch": int(best_epoch_row["epoch"]),
        "best_train_accuracy": float(best_epoch_row["train_accuracy"]),
        "best_train_macro_f1": float(best_epoch_row["train_macro_f1"]),
        "best_train_weighted_f1": float(best_epoch_row["train_weighted_f1"]),
        "best_val_accuracy": float(best_epoch_row["val_accuracy"]),
        "best_val_macro_f1": float(best_epoch_row["val_macro_f1"]),
        "best_val_weighted_f1": float(best_epoch_row["val_weighted_f1"]),
        "trial_dir": trial_dir
    }

    result_path = os.path.join(trial_dir, "trial_summary.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df, val_df, labels = load_and_prepare_data()

    search_space = list(itertools.product(
        MODEL_CANDIDATES,
        BATCH_SIZE_CANDIDATES,
        LR_CANDIDATES,
        WEIGHT_DECAY_CANDIDATES,
        EPOCH_CANDIDATES
    ))

    print(f"\nTotal trials to run: {len(search_space)}")

    all_results = []
    global_best_score = -1.0
    global_best_result = None

    for idx, (model_name, batch_size, lr, weight_decay, epochs) in enumerate(search_space, start=1):
        try:
            result = run_single_trial(
                trial_id=idx,
                model_name=model_name,
                batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
                epochs=epochs,
                train_df=train_df,
                val_df=val_df,
                labels=labels
            )

            all_results.append(result)

            if result["best_val_weighted_f1"] > global_best_score:
                global_best_score = result["best_val_weighted_f1"]
                global_best_result = result
                save_model_dir(result["trial_dir"], BEST_TRIAL_DIR)
                print(f"\nGlobal best trial updated: Trial {idx}")

        except RuntimeError as e:
            print(f"\nTrial {idx} failed with RuntimeError: {e}")
            print("Skipping this trial...")
        except Exception as e:
            print(f"\nTrial {idx} failed with Exception: {e}")
            print("Skipping this trial...")

    if len(all_results) == 0:
        raise ValueError("All tuning trials failed. No result to save.")

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(
        by=["best_val_weighted_f1", "best_val_accuracy"],
        ascending=False
    ).reset_index(drop=True)

    tuning_results_csv = os.path.join(OUTPUT_DIR, "all_tuning_results.csv")
    results_df.to_csv(tuning_results_csv, index=False)

    best_config = results_df.iloc[0].to_dict()

    best_config_path = os.path.join(OUTPUT_DIR, "best_hyperparameters.json")
    with open(best_config_path, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)

    summary = {
        "total_trials_run": len(all_results),
        "selection_metric": "best_val_weighted_f1",
        "best_trial_id": int(best_config["trial_id"]),
        "best_model_name": best_config["model_name"],
        "best_batch_size": int(best_config["batch_size"]),
        "best_learning_rate": float(best_config["learning_rate"]),
        "best_weight_decay": float(best_config["weight_decay"]),
        "best_epochs_requested": int(best_config["epochs_requested"]),
        "best_epochs_completed": int(best_config["epochs_completed"]),
        "best_epoch": int(best_config["best_epoch"]),
        "best_val_accuracy": float(best_config["best_val_accuracy"]),
        "best_val_macro_f1": float(best_config["best_val_macro_f1"]),
        "best_val_weighted_f1": float(best_config["best_val_weighted_f1"]),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "num_labels_used": len(labels),
        "seed": SEED,
        "val_split": VAL_SIZE,
        "min_samples_per_class": MIN_SAMPLES_PER_CLASS
    }

    summary_path = os.path.join(OUTPUT_DIR, "tuning_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if os.path.exists(BEST_MODEL_DIR):
        shutil.rmtree(BEST_MODEL_DIR)
    shutil.copytree(BEST_TRIAL_DIR, BEST_MODEL_DIR)

    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING COMPLETED")
    print("=" * 80)
    print(f"Best Trial ID          : {summary['best_trial_id']}")
    print(f"Best Model             : {summary['best_model_name']}")
    print(f"Best Batch Size        : {summary['best_batch_size']}")
    print(f"Best Learning Rate     : {summary['best_learning_rate']}")
    print(f"Best Weight Decay      : {summary['best_weight_decay']}")
    print(f"Best Epoch             : {summary['best_epoch']}")
    print(f"Best Validation Acc    : {summary['best_val_accuracy']:.4f}")
    print(f"Best Validation MacroF1: {summary['best_val_macro_f1']:.4f}")
    print(f"Best Validation W-F1   : {summary['best_val_weighted_f1']:.4f}")
    print(f"All tuning results     : {tuning_results_csv}")
    print(f"Best hyperparameters   : {best_config_path}")
    print(f"Tuning summary         : {summary_path}")
    print(f"Best final model       : {BEST_MODEL_DIR}")


if __name__ == "__main__":
    main()