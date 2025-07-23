import os
import json
import whisper
import argparse
from tqdm import tqdm
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/clean/metadata.json', help='Path to metadata JSON')
    parser.add_argument('--output', type=str, default='models/outputs/whisper_predictions.json', help='Path to save predictions')
    parser.add_argument('--device', type=str, default='cpu', help='Device: "cpu" or "cuda"')
    parser.add_argument('--model_size', type=str, default='small', help='Whisper model size (tiny, base, small, medium, large)')
    args = parser.parse_args()

    print("Loading metadata...")
    with open(args.input, "r") as f:
        metadata = json.load(f)

    print(f"Loaded {len(metadata)} samples.")
    print(f"Loading Whisper model ({args.model_size}) on {args.device}...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(args.model_size, device=device)


    predictions = []

    for sample in tqdm(metadata, desc="Transcribing"):
        audio_path = sample["path"]
        try:
            result = model.transcribe(audio_path, language="en")
            predictions.append({
                "path": audio_path,
                "text": sample["text"],
                "prediction": result["text"]
            })
        except Exception as e:
            print(f"Failed to process {audio_path}: {e}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"Saved predictions to {args.output}")

if __name__ == "__main__":
    main()
