import os
import re
import json
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, BertModel
from pyabsa import AspectTermExtraction as ATEPC

# Configurations
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_MODEL_DIR = "models/feature_mapper"
SENTIMENT_MODEL_DIR = "models/sentiment_model"
FEATURE_LABEL_MAP = "data/processed/feature_label_map.json"
SENTIMENT_LABEL_MAP = "data/processed/sentiment_label_map.json"
MAX_LENGTH = 128

app = Flask(__name__)

# Model Architecture (Universal for both tasks in this project)
class BertAttentionClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name, output_attentions=True, return_dict=True)
        hidden = self.bert.config.hidden_size
        self.attention = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden, num_labels)
        self.class_weights = torch.ones(num_labels, dtype=torch.float)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        x = outputs.last_hidden_state
        scores = self.attention(x).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, -1e4)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return {"logits": logits}

# Helper functions
def load_label_map(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): v for k, v in data["id2label"].items()}

def load_model(model_dir, num_labels, model_name="bert-base-uncased"):
    model = BertAttentionClassifier(model_name, num_labels)
    safetensor_path = os.path.join(model_dir, "model.safetensors")
    bin_path = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.exists(safetensor_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensor_path)
        model.load_state_dict(state_dict, strict=False)
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer

def clean_text(x):
    if x is None: return ""
    x = str(x).lower().strip()
    x = re.sub(r"\s+", " ", x)
    return x

def predict(model, tokenizer, id2label, aspect, sentence):
    aspect = clean_text(aspect)
    sentence = clean_text(sentence)
    encoding = tokenizer(aspect, sentence, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    encoding = {k: v.to(DEVICE) for k, v in encoding.items()}
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    pred_id = int(np.argmax(probs))
    return id2label[pred_id], float(probs[pred_id])

# Initialize on startup
print("Initializing ABSA models...")
feature_id2label = load_label_map(FEATURE_LABEL_MAP)
sentiment_id2label = load_label_map(SENTIMENT_LABEL_MAP)
feature_model, feature_tokenizer = load_model(FEATURE_MODEL_DIR, len(feature_id2label))
sentiment_model, sentiment_tokenizer = load_model(SENTIMENT_MODEL_DIR, len(sentiment_id2label))

print("Loading PyABSA aspect extractor...")
# Using 'cpu' for PyABSA to avoid CUDA memory/conflict issues with the main BERT models
aspect_extractor = ATEPC.AspectExtractor('multilingual', device='cpu')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    sentence = data.get("sentence", "")
    if not sentence:
        return jsonify({"error": "No sentence provided"}), 400

    try:
        # PyABSA returns a dict for a single sentence
        atepc_result = aspect_extractor.predict(sentence, save_result=False, print_result=False)
        raw_aspects = atepc_result.get('aspect', []) if atepc_result else []
    except Exception as e:
        print(f"Extraction Error: {e}")
        raw_aspects = []

    results = []
    for aspect in raw_aspects:
        feat_label, feat_conf = predict(feature_model, feature_tokenizer, feature_id2label, aspect, sentence)
        sent_label, sent_conf = predict(sentiment_model, sentiment_tokenizer, sentiment_id2label, aspect, sentence)
        results.append({
            "aspect": aspect,
            "feature": feat_label,
            "feature_confidence": feat_conf,
            "sentiment": sent_label,
            "sentiment_confidence": sent_conf
        })

    return jsonify({"sentence": sentence, "results": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
