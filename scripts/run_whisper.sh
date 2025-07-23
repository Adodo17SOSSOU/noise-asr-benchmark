#!/bin/bash
#SBATCH --job-name=run_whisper
#SBATCH --output=logs/whisper.out
#SBATCH --error=logs/whisper.err
#SBATCH --partition=A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

echo "Starting at $(date) on $(hostname)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate asr_env
export HF_HUB_READ_TIMEOUT=60

echo "Running Whisper on CLEAN data..."
python scripts/whisper_infer.py \
  --input data/clean/metadata.json \
  --output models/outputs/whisper_clean_predictions.json \
  --device cuda \
  --model_size small

echo "Running Whisper on NOISY data..."
python scripts/whisper_infer.py \
  --input data/noisy/metadata.json \
  --output models/outputs/whisper_noisy_predictions.json \
  --device cuda \
  --model_size small

echo "Finished at $(date)"

