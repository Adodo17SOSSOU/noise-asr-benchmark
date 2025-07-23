import os
import json
import pandas as pd

INPUT_DIR = "models/outputs"
OUTPUT_CSV = "results/leaderboard.csv"

def extract_info(filename):
    parts = filename.replace("metrics_", "").replace(".json", "").split("_")
    model = parts[0]
    split = parts[1]
    return model, split

def main():
    rows = []
    os.makedirs("results", exist_ok=True)

    for fname in os.listdir(INPUT_DIR):
        if fname.startswith("metrics_") and fname.endswith(".json"):
            path = os.path.join(INPUT_DIR, fname)
            with open(path, "r") as f:
                metrics = json.load(f)

            model, split = extract_info(fname)
            rows.append({
                "Model": model,
                "Data": split,
                "WER": round(metrics["wer"], 3),
                "CER": round(metrics["cer"], 3)
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["Model", "Data"])
    df.to_csv(OUTPUT_CSV, index=False)
    print(" Leaderboard saved to", OUTPUT_CSV)
    print(df)

if __name__ == "__main__":
    main()

