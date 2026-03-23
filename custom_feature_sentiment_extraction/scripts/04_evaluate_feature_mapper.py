import os
import re
import json
import numpy as np
import torch
import torch.nn as nn

from transformers import AutoTokenizer, BertModel

#configs
MODEL_NAME = "bert-base-uncased"
MODEL_DIR = "models/feature_mapper"
LABEL_MAP_PATH = "data/processed/feature_label_map.json"
MAX_LENGTH = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clean_text(x):
    if x is None:
        return ""
    x = str(x).lower().strip()
    x = re.sub(r"\s+", " ", x)
    return x



def load_label_map(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    label2id = data["label2id"]
    id2label = {int(k): v for k, v in data["id2label"].items()}
    return label2id, id2label


#model same as training
class BertAttentionClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()

        self.bert = BertModel.from_pretrained(
            model_name,
            output_attentions=True,
            return_dict=True
        )

        hidden = self.bert.config.hidden_size

        self.attention = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden, num_labels)

        # needed only for compatibility
        self.class_weights = torch.ones(num_labels, dtype=torch.float)

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

        x = outputs.last_hidden_state  # [B, T, H]
        scores = self.attention(x).squeeze(-1)  # [B, T]
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



#loading trained model
def load_trained_model(model_dir, label_map_path):
    label2id, id2label = load_label_map(label_map_path)
    num_labels = len(label2id)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = BertAttentionClassifier(MODEL_NAME, num_labels)

    safetensor_path = os.path.join(model_dir, "model.safetensors")
    bin_path = os.path.join(model_dir, "pytorch_model.bin")

    if os.path.exists(safetensor_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensor_path)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded model from: {safetensor_path}")

    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded model from: {bin_path}")

    else:
        raise FileNotFoundError(
            f"No trained model weights found in {model_dir}. "
            f"Expected model.safetensors or pytorch_model.bin"
        )

    model.to(DEVICE)
    model.eval()

    return model, tokenizer, label2id, id2label


#predict
def predict_feature(model, tokenizer, id2label, aspect, sentence):
    aspect = clean_text(aspect)
    sentence = clean_text(sentence)

    encoding = tokenizer(
        aspect,
        sentence,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    encoding = {k: v.to(DEVICE) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    pred_id = int(np.argmax(probs))
    pred_label = id2label[pred_id]
    confidence = float(probs[pred_id])

    return {
        "aspect": aspect,
        "sentence": sentence,
        "pred_id": pred_id,
        "pred_label": pred_label,
        "confidence": confidence,
        "all_probs": probs
    }


def main():
    model, tokenizer, label2id, id2label = load_trained_model(MODEL_DIR, LABEL_MAP_PATH)

    test_cases = [
        {"aspect": "camera quality", "sentence": "The camera quality is amazing and photos are sharp."},
        {"aspect": "battery drains", "sentence": "Battery drains too quickly even with light usage."},
        {"aspect": "open fast", "sentence": "Apps open fast and multitasking is smooth."},
        {"aspect": "display", "sentence": "The display is bright and colors are very vivid."},
        {"aspect": "looks", "sentence": "The phone looks premium and feels sleek in hand."},
        {"aspect": "Night photography", "sentence": "Night photography is poor and images are noisy."},
        {"aspect": "battery", "sentence": "The battery easily lasts a full day without charging."},
        {"aspect": "Gaming performance", "sentence": "Gaming performance is laggy and slow sometimes."},
        {"aspect": "screen visibility", "sentence": "Screen visibility is bad under sunlight."},
        {"aspect": "phone design", "sentence": "The phone design is outdated and bulky."},
        {"aspect":"touch", "sentence":"touch is good"}
    ]

    print("\n===== PREDICTIONS =====\n")

    for i, case in enumerate(test_cases, 1):
        result = predict_feature(
            model=model,
            tokenizer=tokenizer,
            id2label=id2label,
            aspect=case["aspect"],
            sentence=case["sentence"]
        )

        print(f"Test Case {i}")
        print("Aspect     :", case["aspect"])
        print("Sentence   :", case["sentence"])
        print("Pred ID    :", result["pred_id"])
        print("Prediction :", result["pred_label"])
        print("Confidence :", round(result["confidence"], 4))
        print("-" * 60)


if __name__ == "__main__":
    main()