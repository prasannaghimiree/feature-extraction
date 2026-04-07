import os
import json
import time
import re
from typing import List, Dict, Any

import pandas as pd
from tqdm import tqdm
from google import genai



GEMINI_API_KEY = "AIzaSyDQGlDxjQJreihw5z0y5mZrH2_U09qnA7U"

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Set it as an environment variable.")


MODEL_NAME = "gemini-2.5-flash-lite"

INPUT_CSV = "sentence_reviews.csv"
OUTPUT_CSV = "aspect_semantic_dataset.csv"

# Start small for testing
MAX_ROWS = 2000

SLEEP_SECONDS = 1.0


client = genai.Client(api_key=GEMINI_API_KEY)



def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Tries to extract JSON from a model response.
    """
    text = text.strip()

    # Remove markdown fences if present
    text = text.replace("```json", "").replace("```", "").strip()

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {"items": []}

    json_text = match.group(0)

    try:
        data = json.loads(json_text)
        if not isinstance(data, dict):
            return {"items": []}
        if "items" not in data or not isinstance(data["items"], list):
            return {"items": []}
        return data
    except Exception:
        return {"items": []}


def build_prompt(sentence: str, category: str, product: str, existing_features: list[str]) -> str:
    existing_str = ", ".join(existing_features[:100]) if existing_features else "None yet"

    return f"""
        You are building a high-quality aspect-based sentiment dataset for product reviews.

        Task:
        For the given review sentence, extract ALL EXPLICIT product-related features/aspects.
        For each extracted aspect:
        1. return the raw feature phrase as it appears in the sentence
        2. assign a canonical feature label
        3. assign sentiment: positive, negative, or neutral

        IMPORTANT SENTIMENT POLICY:
        - positive = clear strong praise or favorable evaluation
        - negative = clear complaint, defect, weakness, or unfavorable evaluation
        - neutral = mixed, moderate, comparative, weak, qualified, unclear, or descriptive sentiment

        Use neutral when:
        - the sentence contains both positive and negative clues
        - the statement is weak or qualified, such as "okay", "moderate", "better in this range", "good for the price"
        - the sentence is descriptive but not clearly polarized

        Rules:
        - Extract only explicit product features/aspects.
        - Do NOT infer hidden aspects.
        - Ignore seller, delivery, packaging, shipping, platform, and purchase timing.
        - Canonical feature should be short, clean, and reusable.
        - Merge paraphrases when they mean the same feature.
        - Keep subfeatures separate only when meaning is clearly different.
        - If there is no valid product feature, return an empty list.
        - Return JSON only.

        Category: {category}
        Product: {product}

        Existing canonical features already used:
        {existing_str}

        Examples:

        Sentence: Camera quality is awesome.
        Output:
        {{
        "items": [
            {{
            "raw_aspect": "Camera quality",
            "canonical_feature": "camera",
            "sentiment": "positive"
            }}
        ]
        }}

        Sentence: Touch is not much good but better in this range.
        Output:
        {{
        "items": [
            {{
            "raw_aspect": "Touch",
            "canonical_feature": "touchscreen",
            "sentiment": "neutral"
            }}
        ]
        }}

        Sentence: Display is good according to its price range.
        Output:
        {{
        "items": [
            {{
            "raw_aspect": "Display",
            "canonical_feature": "display",
            "sentiment": "neutral"
            }}
        ]
        }}

        Sentence: Battery drains quickly.
        Output:
        {{
        "items": [
            {{
            "raw_aspect": "Battery",
            "canonical_feature": "battery",
            "sentiment": "negative"
            }}
        ]
        }}

        Sentence: Fast delivery and nice packaging.
        Output:
        {{
        "items": []
        }}

        Now process this sentence:
        Sentence: "{sentence}"
        """.strip()


def normalize_items(parsed: Dict[str, Any]) -> List[Dict[str, str]]:
    valid_sentiments = {"positive", "negative", "neutral"}
    cleaned = []

    for item in parsed.get("items", []):
        if not isinstance(item, dict):
            continue

        raw_aspect = clean_text(item.get("raw_aspect", ""))
        canonical_feature = clean_text(item.get("canonical_feature", "")).lower()
        sentiment = clean_text(item.get("sentiment", "")).lower()

        if not raw_aspect:
            continue
        if not canonical_feature:
            continue
        if sentiment not in valid_sentiments:
            continue

        cleaned.append({
            "raw_aspect": raw_aspect,
            "canonical_feature": canonical_feature,
            "sentiment": sentiment
        })

    return cleaned




df = pd.read_csv(INPUT_CSV)

required_cols = {"product", "category", "rating", "review_title", "sentence"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.head(MAX_ROWS).copy()
df["sentence"] = df["sentence"].astype(str).apply(clean_text)

print(f"Loaded {len(df)} rows from {INPUT_CSV}")



results = []
canonical_feature_memory = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Annotating"):
    sentence = row["sentence"]
    product = clean_text(row["product"])
    category = clean_text(row["category"])
    rating = row["rating"]
    review_title = clean_text(row["review_title"])

    prompt = build_prompt(
        sentence=sentence,
        category=category,
        product=product,
        existing_features=canonical_feature_memory
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        raw_text = response.text if response.text else ""
        parsed = safe_json_loads(raw_text)
        items = normalize_items(parsed)

    except Exception as e:
        print(f"\nError at row {idx}: {e}")
        items = []

    # Save row-level expansion
    if not items:
        results.append({
            "product": product,
            "category": category,
            "rating": rating,
            "review_title": review_title,
            "sentence": sentence,
            "raw_aspect": "",
            "canonical_feature": "",
            "sentiment": "",
            "has_feature": False
        })
    else:
        for item in items:
            results.append({
                "product": product,
                "category": category,
                "rating": rating,
                "review_title": review_title,
                "sentence": sentence,
                "raw_aspect": item["raw_aspect"],
                "canonical_feature": item["canonical_feature"],
                "sentiment": item["sentiment"],
                "has_feature": True
            })

            # update memory only if new
            if item["canonical_feature"] not in canonical_feature_memory:
                canonical_feature_memory.append(item["canonical_feature"])

    time.sleep(SLEEP_SECONDS)




out_df = pd.DataFrame(results)
out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

print(f"\nSaved output to {OUTPUT_CSV}")
print("\nSample rows:")
print(out_df.head(20))

print("\nUnique canonical features found so far:")
print(sorted([x for x in out_df["canonical_feature"].dropna().unique() if x]))