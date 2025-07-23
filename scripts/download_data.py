# scripts/download_data.py

import os
import csv
import json
import shutil
from tqdm import tqdm

CORPUS_DIR = "/tsi/medical/SSL-Brain/Projects/accented-asr-benchmark/data/common_voice/raw/common_voice_16_en/en"
CLIPS_DIR = os.path.join(CORPUS_DIR, "clips")
VALIDATION_FILE = os.path.join(CORPUS_DIR, "validated.tsv")

OUTPUT_DIR = "data/clean"
METADATA_PATH = os.path.join(OUTPUT_DIR, "metadata.json")
MAX_SAMPLES = 50

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metadata = []

    print("Reading TSV and copying audio files...")
    with open(VALIDATION_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        count = 0

        for row in tqdm(reader):
            filename = row.get("path")
            sentence = row.get("sentence")

            if not filename or not sentence:
                continue

            source_path = os.path.join(CLIPS_DIR, filename)
            target_path = os.path.join(OUTPUT_DIR, filename)

            if not os.path.exists(source_path):
                continue

            shutil.copy2(source_path, target_path)

            metadata.append({
                "path": target_path,
                "text": sentence
            })

            count += 1
            if count >= MAX_SAMPLES:
                break

    print(f"Saving metadata to {METADATA_PATH}")
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f" Done. {len(metadata)} audio files copied to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
