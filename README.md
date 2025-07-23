# 🔪 Noise-Robust ASR Benchmark

This project benchmarks the performance of two Automatic Speech Recognition (ASR) models — **Whisper** and **Wav2Vec2** — under clean and noisy audio conditions using a subset of the [Common Voice](https://commonvoice.mozilla.org/) dataset.

---

## 🔀 Project Pipeline

1. **Data Preparation**

   * Extract and clean Common Voice samples
   * Add Gaussian noise to generate a noisy subset

2. **Inference**

   * Run Whisper and Wav2Vec2 on both clean and noisy samples

3. **Evaluation**

   * Compute Word Error Rate (WER) and Character Error Rate (CER)

4. **Reporting**

   * Save results to CSV and generate bar chart comparisons

---

## 📁 Project Structure

```
noise_asr_benchmark/
├── data/                  # Contains clean and noisy datasets
├── logs/                 # Slurm log outputs
├── models/               # Can be used for saving model checkpoints
├── results/              # WER/CER metrics, plots and leaderboard
│   ├── leaderboard.csv
│   ├── leaderboard_cer.png
│   └── leaderboard_wer.png
├── scripts/              # All processing and evaluation scripts
│   ├── add_noise.py
│   ├── compute_metrics.py
│   ├── download_data.py
│   ├── generate_leaderboard.py
│   ├── visualize_results.py
│   ├── whisper_infer.py
│   ├── wav2vec_infer.py
│   ├── run_whisper.sh
│   ├── run_wav2vec.sh
│   └── run_metrics.sh
├── utils/                # Helper functions (e.g., metrics)
├── requirements.txt      # Python dependencies
└── README.md             # You're reading it!
```

---

## 📊 Leaderboard

| Model    | Data  | WER   | CER   |
| -------- | ----- | ----- | ----- |
| Whisper  | clean | 0.121 | 0.051 |
| Whisper  | noisy | 0.253 | 0.143 |
| Wav2Vec2 | clean | 0.259 | 0.079 |
| Wav2Vec2 | noisy | 0.414 | 0.198 |

Plots:

* `results/leaderboard_wer.png`
* `results/leaderboard_cer.png`

---

## 🚀 How to Run

```bash
# Activate your environment
conda activate asr_env

# 1. Prepare data
python scripts/download_data.py
python scripts/add_noise.py

# 2. Inference
bash scripts/run_whisper.sh
bash scripts/run_wav2vec.sh

# 3. Evaluate
bash scripts/run_metrics.sh

# 4. Visualize
python scripts/generate_leaderboard.py
python scripts/visualize_results.py
```

---

## 🧐 Key Insights

* Whisper is significantly more robust to noise than Wav2Vec2.
* Wav2Vec2’s WER increases drastically under noisy conditions.
* Clean modular design allows easy extension (e.g., new ASR models, noise types).

---

## 🛠 Dependencies

Install via:

```bash
pip install -r requirements.txt
```

---

## 🪪 License

MIT License © 2025 A. Sossou

