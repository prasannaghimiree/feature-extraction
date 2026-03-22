import os
import json
import pandas as pd

from pyabsa import AspectTermExtraction as ATEPC

INPUT_CSV = "outputs/clean_sentences.csv"
OUTPUT_JSON = "outputs/pyabsa_raw_results.json"


def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    sentences = df["sentence"].dropna().astype(str).tolist()

    if len(sentences) == 0:
        raise ValueError("No sentences found for inference.")

    aspect_extractor = ATEPC.AspectExtractor(
        "multilingual",
        auto_device=True,
        cal_perplexity=False
    )

    results = aspect_extractor.predict(
        sentences,
        save_result=False,
        print_result=False,
        pred_sentiment=True,
        ignore_error=True
    )

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Inference completed.")
    print("Saved raw results to:", OUTPUT_JSON)


if __name__ == "__main__":
    main()