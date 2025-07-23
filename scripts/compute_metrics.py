import json
import argparse
import jiwer
import os

def compute_wer_cer(samples):
    references = [s["text"].lower() for s in samples]
    predictions = [s["prediction"].lower() for s in samples]

    wer = jiwer.wer(references, predictions)
    cer = jiwer.cer(references, predictions)
    return {"wer": wer, "cer": cer}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to prediction JSON")
    parser.add_argument("--output", type=str, required=True, help="Where to save metrics JSON")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        predictions = json.load(f)

    print(f"Loaded {len(predictions)} predictions from {args.input}")
    metrics = compute_wer_cer(predictions)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved WER: {metrics['wer']:.3f}, CER: {metrics['cer']:.3f} to {args.output}")

if __name__ == "__main__":
    main()
