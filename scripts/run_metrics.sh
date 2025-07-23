#!/bin/bash
#SBATCH --job-name=metrics
#SBATCH --output=logs/metrics.out
#SBATCH --error=logs/metrics.err
#SBATCH --partition=L40S
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

echo "Starting metrics at $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate asr_env

# Whisper
python scripts/compute_metrics.py \
  --input models/outputs/whisper_clean_predictions.json \
  --output models/outputs/metrics_whisper_clean.json

python scripts/compute_metrics.py \
  --input models/outputs/whisper_noisy_predictions.json \
  --output models/outputs/metrics_whisper_noisy.json

# Wav2Vec2
python scripts/compute_metrics.py \
  --input models/outputs/wav2vec_predictions_clean.json \
  --output models/outputs/metrics_wav2vec_clean.json

python scripts/compute_metrics.py \
  --input models/outputs/wav2vec_predictions_noisy.json \
  --output models/outputs/metrics_wav2vec_noisy.json

echo "Done computing metrics at $(date)"

