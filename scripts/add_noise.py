import os
import json
import torchaudio
import torch
from tqdm import tqdm

INPUT_METADATA = "data/clean/metadata.json"
OUTPUT_DIR = "data/noisy"
NOISY_CLIPS_DIR = os.path.join(OUTPUT_DIR, "clips")
OUTPUT_METADATA = os.path.join(OUTPUT_DIR, "metadata.json")

def add_gaussian_noise(waveform, noise_level=0.02):
    noise = torch.randn_like(waveform) * noise_level
    return waveform + noise

def main():
    os.makedirs(NOISY_CLIPS_DIR, exist_ok=True)

    with open(INPUT_METADATA, "r") as f:
        metadata = json.load(f)

    noisy_metadata = []

    for sample in tqdm(metadata, desc="Adding noise"):
        try:
            waveform, sample_rate = torchaudio.load(sample["path"])
            noisy_waveform = add_gaussian_noise(waveform)

            # Save noisy file
            filename = os.path.basename(sample["path"])
            noisy_path = os.path.join(NOISY_CLIPS_DIR, filename)
            torchaudio.save(noisy_path, noisy_waveform, sample_rate)

            noisy_metadata.append({
                "path": noisy_path,
                "text": sample["text"]
            })
        except Exception as e:
            print(f"Failed on {sample['path']}: {e}")

    with open(OUTPUT_METADATA, "w") as f:
        json.dump(noisy_metadata, f, indent=2)

    print(f"Saved {len(noisy_metadata)} noisy samples to {OUTPUT_METADATA}")

if __name__ == "__main__":
    main()

