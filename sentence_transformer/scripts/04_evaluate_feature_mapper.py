import os
import re
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util


MODEL_DIR = "models/feature_mapper_semantic_tuned/trial_71"
LABEL_MAP_PATH = "data/processed/feature_label_map.json"
OUTPUT_CSV = os.path.join(MODEL_DIR, "demo_predictions.csv")


def clean_text(text):
    if text is None:
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
        raise FileNotFoundError(f"Label map file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    label2id = data["label2id"]
    id2label = {int(k): v for k, v in data["id2label"].items()}
    return label2id, id2label


def predict_feature(model, labels, raw_aspect, sentence):
    query = build_input_text(raw_aspect, sentence)

    query_embedding = model.encode(
        query,
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

    similarities = util.cos_sim(query_embedding, label_embeddings)[0].cpu().numpy()

    pred_idx = int(np.argmax(similarities))
    pred_label = labels[pred_idx]
    confidence = float(similarities[pred_idx])

    score_dict = {
        labels[i]: float(similarities[i])
        for i in range(len(labels))
    }

    return {
        "raw_aspect": raw_aspect,
        "sentence": sentence,
        "pred_label": pred_label,
        "confidence": confidence,
        "scores": score_dict
    }


def main():
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

    model = SentenceTransformer(MODEL_DIR)

    _, id2label = load_label_map(LABEL_MAP_PATH)
    labels = [id2label[i] for i in range(len(id2label))]

    test_cases = [
    {
        "raw_aspect": "frontcamara",
        "sentence": "The frontcamara captures clean selfies even in indoor lighting."
    },
    {
        "raw_aspect": "power",
        "sentence": "This phone has enough power even after using it all day."
    },
    {
        "raw_aspect": "power backup",
        "sentence": "Power backup is decent and it survives a full working day."
    },
    {
        "raw_aspect": "night shots",
        "sentence": "Night shots come out grainy and full of noise."
    },
    {
        "raw_aspect": "sunlight use",
        "sentence": "Using it outside is difficult because the screen becomes hard to read."
    },
    {
        "raw_aspect": "lag",
        "sentence": "It stutters when switching between apps and feels slow sometimes."
    },
    {
        "raw_aspect": "heating",
        "sentence": "The device gets warm quickly while gaming for just a few minutes."
    },
    {
        "raw_aspect": "looks premium",
        "sentence": "It feels elegant in hand and gives off a premium vibe."
    },
    {
        "raw_aspect": "in hand feel",
        "sentence": "The handset feels bulky and uncomfortable after long use."
    },
    {
        "raw_aspect": "unlock",
        "sentence": "Unlocking with the finger scanner is quick and reliable."
    },
    {
        "raw_aspect": "charging",
        "sentence": "It goes from almost dead to usable in a very short time."
    },
    {
        "raw_aspect": "speaker",
        "sentence": "Audio is loud enough, but it gets distorted at high volume."
    },
    {
        "raw_aspect": "selfie cam",
        "sentence": "Selfie cam smooths faces too much and loses detail."
    },
    {
        "raw_aspect": "main shooter",
        "sentence": "The main shooter produces sharp photos with natural colors."
    },
    {
        "raw_aspect": "touch response",
        "sentence": "The screen reacts instantly when typing or scrolling."
    },
    {
        "raw_aspect": "software feel",
        "sentence": "Animations are fluid and moving through menus feels smooth."
    },
    {
        "raw_aspect": "storage",
        "sentence": "I ran out of space quickly after installing a few apps and videos."
    },
    {
        "raw_aspect": "memory",
        "sentence": "Keeping many apps open makes the phone reload them less often."
    },
    {
        "raw_aspect": "sim tray",
        "sentence": "The sim tray is too flimsy and difficult to remove."
    },
    {
        "raw_aspect": "build",
        "sentence": "The back panel feels cheap although the frame looks solid."
    },
    {
        "raw_aspect": "body",
        "sentence": "Its slim body makes it comfortable to hold with one hand."
    },
    {
        "raw_aspect": "value",
        "sentence": "For this price, the overall package feels worth it."
    },
    {
        "raw_aspect": "brand trust",
        "sentence": "I bought it mainly because this brand is usually dependable."
    },
    {
        "raw_aspect": "model choice",
        "sentence": "Compared to the previous model, this one feels much faster."
    },
    {
        "raw_aspect": "connectivity",
        "sentence": "WiFi drops randomly and mobile data switching is slow."
    },
    {
        "raw_aspect": "signal",
        "sentence": "Network reception is weak inside buildings."
    },
    {
        "raw_aspect": "video",
        "sentence": "Video recording is stable but details look soft."
    },
    {
        "raw_aspect": "4k",
        "sentence": "The phone shoots 4k nicely, but low-light clips are poor."
    },
    {
        "raw_aspect": "notification",
        "sentence": "I keep missing alerts because the indicator is too subtle."
    },
    {
        "raw_aspect": "ui",
        "sentence": "The interface feels cluttered with too many unnecessary apps."
    },
    {
        "raw_aspect": "daily use",
        "sentence": "For calls, browsing, and messaging, it performs just fine."
    },
    {
        "raw_aspect": "gaming",
        "sentence": "Heavy games run, but frame drops appear after a while."
    },
    {
        "raw_aspect": "camera bump",
        "sentence": "It wobbles on the table because of the large camera bump."
    },
    {
        "raw_aspect": "portrait shots",
        "sentence": "Portrait shots separate the subject well, but edge detection fails sometimes."
    },
    {
        "raw_aspect": "brightness",
        "sentence": "Brightness is not enough outdoors under strong sun."
    },
    {
        "raw_aspect": "color tone",
        "sentence": "The display looks vibrant, though colors feel slightly oversaturated."
    },
    {
        "raw_aspect": "speed",
        "sentence": "Everything opens instantly and the phone feels snappy."
    },
    {
        "raw_aspect": "camera in dark",
        "sentence": "Photos taken in dark rooms lose detail and look muddy."
    },
    {
        "raw_aspect": "battery in standby",
        "sentence": "Even on standby, the battery drops more than expected overnight."
    },
    {
        "raw_aspect": "phone weight",
        "sentence": "It is lighter than expected and easy to carry in hand."
    },
    {
        "raw_aspect": "temperature",
        "sentence": "I like overall features but it gets high temperature very soon"
    },
    {
        "raw_aspect": "default application",
        "sentence": "I like overall features but there are too many default downloaded application which is useless"
    }
    ]
    all_results = []

    print("\n===== FEATURE MAPPER PREDICTIONS =====\n")

    for i, case in enumerate(test_cases, 1):
        result = predict_feature(
            model=model,
            labels=labels,
            raw_aspect=case["raw_aspect"],
            sentence=case["sentence"]
        )

        print(f"Test Case {i}")
        print("Aspect     :", case["raw_aspect"])
        print("Sentence   :", case["sentence"])
        print("Prediction :", result["pred_label"])
        print("Confidence :", round(result["confidence"], 4))
        print("-" * 60)

        all_results.append({
            "test_case": i,
            "raw_aspect": case["raw_aspect"],
            "sentence": case["sentence"],
            "prediction": result["pred_label"],
            "confidence": round(result["confidence"], 4)
        })

    result_df = pd.DataFrame(all_results)
    result_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved demo predictions to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()