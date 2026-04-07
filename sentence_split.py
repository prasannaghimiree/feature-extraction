import pandas as pd
import re

# Load cleaned dataset
df = pd.read_csv("clean_reviews.csv")

def clean_text(text):
    text = str(text)
    text = re.sub(r'(\d+)\)', '. ', text)
    text = re.sub(r'\.{2,}', '.', text)

    text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text

def split_into_sentences(text):
    text = clean_text(text)

    parts = re.split(r'[.!?]+', text)

    sentences = []
    for part in parts:
        part = part.strip(" ,:-")
        part = re.sub(r'\s+', ' ', part).strip()

        if len(part.split()) >= 3:
            sentences.append(part)

    return sentences

rows = []

for _, row in df.iterrows():
    product = row["product"]
    category = row["category"]
    rating = row["rating"]
    review_title = row["review_title"]
    description = row["description"]

    sentences = split_into_sentences(description)

    for sent in sentences:
        rows.append({
            "product": product,
            "category": category,
            "rating": rating,
            "review_title": review_title,
            "sentence": sent
        })

df_sent = pd.DataFrame(rows)

print("Sentence-level dataset shape:", df_sent.shape)

print("\nSample sentence-level rows:")
print(df_sent.head(20))

df_sent.to_csv("sentence_reviews.csv", index=False, encoding="utf-8")
print("\nSaved sentence-level dataset to sentence_reviews.csv")