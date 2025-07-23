import os
import json
import argparse
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import torchaudio

def load_audio(path):
    waveform, sample_rate = torchaudio.load(path)
    return waveform.squeeze(), sample_rate

def transcribe(sample, processor, model, device):
    audio, sr = load_audio(sample["path"])
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio = resampler(audio)

    input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values.to(device)
    with torch.no_grad():
        logits = model(input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(pred_ids[0])
    return transcription

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, choices=["clean", "noisy"], default="clean")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    input_path = f"data/{args.split}/metadata.json"
    output_path = f"models/outputs/wav2vec_predictions_{args.split}.json"

    print(f"Loading metadata from {input_path}...")
    with open(input_path, "r") as f:
        metadata = json.load(f)

    print(f"Loaded {len(metadata)} samples.")
    print(f"Loading Wav2Vec2 model on {args.device}...")

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").to(args.device)
    model.eval()

    predictions = []
    for sample in tqdm(metadata, desc="Transcribing"):
        try:
            pred = transcribe(sample, processor, model, args.device)
            predictions.append({
                "path": sample["path"],
                "text": sample["text"],
                "prediction": pred
            })
        except Exception as e:
            print(f"Failed to transcribe {sample['path']}: {e}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"Saved predictions to {output_path}")

if __name__ == "__main__":
    main()

