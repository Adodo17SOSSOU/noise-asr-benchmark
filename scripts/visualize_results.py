import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    input_csv = "results/leaderboard.csv"
    output_path = "results/leaderboard.png"

    df = pd.read_csv(input_csv)
    df = df.sort_values(by=["Model", "Data"])

    fig, ax = plt.subplots(figsize=(8, 5))

    for metric in ["WER", "CER"]:
        ax.clear()
        df_plot = df.pivot(index="Model", columns="Data", values=metric)
        df_plot.plot(kind="bar", ax=ax)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} Comparison on Clean vs Noisy Data")
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.xticks(rotation=0)

        # Save one file per metric
        out_file = output_path.replace(".png", f"_{metric.lower()}.png")
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_file)
        print(f" Saved: {out_file}")

if __name__ == "__main__":
    main()
