import json
import pandas as pd

# Path to your JSON file
json_path = "electronics_reviews_uniq.json"

# Load JSON
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

print("Shape of dataset:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 3 rows:")
print(df.head(3))

print("\nNull values per column:")
print(df.isnull().sum())

# Keep only useful columns
useful_cols = ["product", "category", "rating", "review_title", "description"]
df_clean = df[useful_cols].copy()

# Remove rows with empty review text
df_clean["description"] = df_clean["description"].astype(str).str.strip()
df_clean = df_clean[df_clean["description"] != ""]

print("\nCleaned dataset shape:", df_clean.shape)

print("\nSample cleaned reviews:")
for i in range(min(5, len(df_clean))):
    print(f"\nReview {i+1}:")
    print("Product:", df_clean.iloc[i]["product"])
    print("Rating:", df_clean.iloc[i]["rating"])
    print("Text:", df_clean.iloc[i]["description"])

# Save cleaned file
df_clean.to_csv("clean_reviews.csv", index=False, encoding="utf-8")
print("\nSaved cleaned dataset to clean_reviews.csv")