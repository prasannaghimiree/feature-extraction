import os
import json
import pandas as pd

INPUT_JSON = "outputs/pyabsa_raw_results.json"
OUTPUT_CSV = "outputs/pyabsa_formatted_results.csv"


def safe_get(d, keys, default=None):
    for key in keys:
        if isinstance(d, dict) and key in d:
            return d[key]
    return default


def main():
    if not os.path.exists(INPUT_JSON):
        raise FileNotFoundError(f"Input not found: {INPUT_JSON}")

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for item in data:
        sentence = safe_get(item, ["text", "sentence", "source"], "")

        aspects = safe_get(item, ["aspect", "aspects", "aspect_terms"], [])
        sentiments = safe_get(item, ["sentiment", "sentiments", "aspect_sentiments"], [])

        # Case 1: list of aspects + list of sentiments
        if isinstance(aspects, list) and len(aspects) > 0:
            if isinstance(sentiments, list) and len(sentiments) == len(aspects):
                for asp, sent in zip(aspects, sentiments):
                    rows.append({
                        "sentence": sentence,
                        "raw_aspect": asp,
                        "sentiment": sent
                    })
            else:
                for asp in aspects:
                    rows.append({
                        "sentence": sentence,
                        "raw_aspect": asp,
                        "sentiment": ""
                    })
        else:
            rows.append({
                "sentence": sentence,
                "raw_aspect": "",
                "sentiment": ""
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print("Formatted results saved to:", OUTPUT_CSV)
    print("Total output rows:", len(out_df))


if __name__ == "__main__":
    main()