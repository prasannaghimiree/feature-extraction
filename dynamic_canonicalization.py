import os
import json
import time
import re
from typing import List, Dict

import pandas as pd
from tqdm import tqdm
from google import genai


# =========================================================
# 1. CONFIG
# =========================================================

GEMINI_API_KEY = "AIzaSyDQGlDxjQJreihw5z0y5mZrH2_U09qnA7U"
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Set it as an environment variable.")

MODEL_NAME = "gemini-2.5-flash-lite"

INPUT_CSV = "aspect_semantic_dataset.csv"
OUTPUT_CSV = "aspect_semantic_dataset_canonicalized.csv"

MAX_ROWS = 2000
SLEEP_SECONDS = 1.0

client = genai.Client(api_key=GEMINI_API_KEY)


# =========================================================
# 2. HELPERS
# =========================================================

def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def safe_json_loads(text: str) -> Dict:
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}

    try:
        return json.loads(match.group(0))
    except Exception:
        return {}

def normalize_feature_string(text: str) -> str:
    text = clean_text(text).lower()
    text = text.replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]+", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


# =========================================================
# 3. PROMPT
# =========================================================

def build_prompt(sentence: str, raw_aspect: str, category: str, product: str, existing_features: List[str]) -> str:
    existing_str = ", ".join(existing_features[:150]) if existing_features else "None yet"

    return f"""
You are normalizing product review feature labels for an aspect-based sentiment dataset.

Task:
Given:
- review sentence
- raw extracted aspect
- existing canonical feature inventory

Decide whether the raw aspect is semantically equivalent to one of the existing canonical features.
If yes, return that exact existing canonical feature.
If no, create a new short canonical feature label.

Rules:
1. Reuse an existing canonical feature whenever meaning is the same or very close.
2. Only create a new canonical feature if none of the existing ones fit.
3. Canonical feature must be:
   - short
   - reusable
   - lowercase
   - underscore-separated if multiple words
4. Keep important subfeatures separate only when meaning is clearly different.
5. Use sentence context to decide meaning.
6. Return JSON only.

Category: {category}
Product: {product}
Sentence: "{sentence}"
Raw aspect: "{raw_aspect}"

Existing canonical features:
{existing_str}

Examples:

Sentence: "i like its camera result the most"
Raw aspect: "camera result"
Existing canonical features: camera, battery, display
Output:
{{
  "match_type": "existing",
  "canonical_feature": "camera"
}}

Sentence: "Finger sensor very fast"
Raw aspect: "Finger sensor"
Existing canonical features: camera, battery, display
Output:
{{
  "match_type": "new",
  "canonical_feature": "fingerprint_sensor"
}}

Sentence: "it generates good colors"
Raw aspect: "colors"
Existing canonical features: camera, battery, display
Output:
{{
  "match_type": "existing",
  "canonical_feature": "display"
}}

Now process the given input.
""".strip()


# =========================================================
# 4. LOAD DATA
# =========================================================

df = pd.read_csv(INPUT_CSV)

# Keep only rows with extracted features
df = df[df["has_feature"] == True].copy()

# Remove empty raw aspects
df["raw_aspect"] = df["raw_aspect"].astype(str).apply(clean_text)
df["sentence"] = df["sentence"].astype(str).apply(clean_text)
df["category"] = df["category"].astype(str).apply(clean_text)
df["product"] = df["product"].astype(str).apply(clean_text)

df = df[df["raw_aspect"] != ""].copy()

df = df.head(MAX_ROWS).copy()

print(f"Loaded {len(df)} rows for canonicalization.")


# =========================================================
# 5. BUILD INITIAL FEATURE MEMORY
# =========================================================
# Start from already proposed canonical features, but let Gemini refine them.
feature_memory = []

for feat in df["canonical_feature"].dropna().astype(str).tolist():
    feat = normalize_feature_string(feat)
    if feat and feat not in feature_memory:
        feature_memory.append(feat)

print("\nInitial feature memory:")
print(feature_memory)


# =========================================================
# 6. MAIN LOOP
# =========================================================

final_rows = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Canonicalizing"):
    sentence = row["sentence"]
    raw_aspect = row["raw_aspect"]
    category = row["category"]
    product = row["product"]

    prompt = build_prompt(
        sentence=sentence,
        raw_aspect=raw_aspect,
        category=category,
        product=product,
        existing_features=feature_memory
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        raw_text = response.text if response.text else ""
        parsed = safe_json_loads(raw_text)

        final_feature = normalize_feature_string(parsed.get("canonical_feature", ""))
        match_type = clean_text(parsed.get("match_type", "")).lower()

        if not final_feature:
            # fallback to original canonical feature if Gemini fails
            final_feature = normalize_feature_string(row.get("canonical_feature", ""))
            match_type = "fallback"

    except Exception as e:
        print(f"\nError at row {idx}: {e}")
        final_feature = normalize_feature_string(row.get("canonical_feature", ""))
        match_type = "fallback"

    if final_feature and final_feature not in feature_memory:
        feature_memory.append(final_feature)

    final_rows.append({
        "product": row["product"],
        "category": row["category"],
        "rating": row["rating"],
        "review_title": row["review_title"],
        "sentence": row["sentence"],
        "raw_aspect": row["raw_aspect"],
        "initial_canonical_feature": row["canonical_feature"],
        "final_canonical_feature": final_feature,
        "sentiment": row["sentiment"],
        "match_type": match_type,
        "has_feature": row["has_feature"]
    })

    time.sleep(SLEEP_SECONDS)


# =========================================================
# 7. SAVE
# =========================================================

out_df = pd.DataFrame(final_rows)
out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

print(f"\nSaved canonicalized dataset to {OUTPUT_CSV}")

print("\nSample rows:")
print(out_df.head(20))

print("\nFinal canonical feature inventory:")
print(sorted(out_df["final_canonical_feature"].dropna().unique().tolist()))