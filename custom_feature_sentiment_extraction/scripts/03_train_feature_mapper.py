import os
import re
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

from transformers import (
    AutoTokenizer,
    BertModel,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed,
)

MODEL_NAME = "bert-base-uncased"
DATA_PATH = "data/processed/feature_train_full.csv"
LABEL_MAP_PATH = "data/processed/feature_label_map.json"
OUTPUT_DIR = "models/feature_mapper"

MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 100
LR = 2e-5
SEED = 42
VAL_SIZE = 0.1


def set_all_seeds(seed):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clean_text(x):
    #checking if value is missing
    if pd.isna(x):
        return ""
    #removing  extra spaces and making lowercase
    x = str(x).lower().strip()
    # replacing one or more whitespace into a single space
    x = re.sub(r"\s+", " ", x)
    return x


#reads label mapping json file
def load_label_map(path):
    with open(path, "r") as f:
        data = json.load(f)
    label2id = data["label2id"]
    id2label = {int(k): v for k, v in data["id2label"].items()}
    return label2id, id2label


#custom dataset class
class FeatureDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        aspect = clean_text(row["raw_aspect"])
        sentence = clean_text(row["sentence"])
        label = int(row["label_id"])

        #since we are passing two texts, tokenizer treats thus as paired input
        # [CLS] aspect [SEP] sentence [SEP]
        encoding = self.tokenizer(
            aspect,
            sentence,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        # outputs : input_ids, attention_mask, label

        item = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": label,
        }

        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"]

        return item


class BertAttentionClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()

        #loading BERT architecture
        self.bert = BertModel.from_pretrained(
            model_name,
            output_attentions=True,
            return_dict=True
        )

        hidden = self.bert.config.hidden_size

        #custom token scoring layer
        self.attention = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        x = outputs.last_hidden_state 

        scores = self.attention(x).squeeze(-1) 

        scores = scores.masked_fill(attention_mask == 0, -1e4)

        weights = torch.softmax(scores, dim=1)

        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }

def main():
    set_all_seeds(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    print("Total samples:", len(df))
    print(df["label"].value_counts())

    label2id, id2label = load_label_map(LABEL_MAP_PATH)
    num_labels = len(label2id)

    # split
    try:
        train_df, val_df = train_test_split(
            df,
            test_size=VAL_SIZE,
            stratify=df["label_id"],
            random_state=SEED
        )
    except:
        train_df, val_df = train_test_split(
            df,
            test_size=VAL_SIZE,
            random_state=SEED
        )

    # loading model bert base uncased
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = FeatureDataset(train_df, tokenizer)
    val_dataset = FeatureDataset(val_df, tokenizer)

    # class weights
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_df["label_id"]),
        y=train_df["label_id"]
    )

    full_weights = np.ones(num_labels)
    for i, w in zip(np.unique(train_df["label_id"]), class_weights):
        full_weights[i] = w

    model = BertAttentionClassifier(MODEL_NAME, num_labels)
    model.class_weights = torch.tensor(full_weights, dtype=torch.float)

    data_collator = DataCollatorWithPadding(tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_steps=20,
        load_best_model_at_end=True,
        report_to="none",
        fp16=False, 
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )

    trainer.train()

    print("\nEvaluation:")
    print(trainer.evaluate())

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\nModel saved at: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()