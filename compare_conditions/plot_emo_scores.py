#!/usr/bin/env python3
"""
Stage‑2 script:
  • Reads data/emo_zscores.csv
  • Produces split_{patient,therapist}.png in ./plots
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

PALETTE = {"depression": "#1f77b4",  # blue
           "anxiety":    "#ff7f0e"}  # orange

ORDERED_EMOTIONS = [
    "anger", "anticipation", "disgust", "fear",
    "joy", "sadness", "surprise", "trust"
]

def main():
    df = pd.read_csv("data/emo_zscores.csv")

    os.makedirs("plots", exist_ok=True)
    for role in df["role"].unique():
        sub = df[df["role"] == role]

        fig, axes = plt.subplots(
            nrows=1, ncols=2, figsize=(14, 5),
            sharey=True, constrained_layout=True
        )

        for ax, model in zip(axes, ["gemma3", "llama3.3"]):
            pivot = (
                sub[sub["model"] == model]
                .pivot(index="emotion", columns="condition", values="zscore")
                .reindex(ORDERED_EMOTIONS)
            )
            ordered_cols = ["depression", "anxiety"]
            pivot = pivot[ordered_cols]

            pivot.plot(kind="bar",
                       ax=ax,
                       width=0.8,
                       color=[PALETTE[c] for c in ordered_cols])
            ax.set_title(f"{model.capitalize()} – {role.capitalize()}")
            ax.set_xlabel("")
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
            ax.legend(title="Prompt")

        fig.supylabel("Average Z‑Score")
        plt.savefig(f"plots/split_{role}.png")
        plt.close()

    print("✅  Plots saved in ./plots")

if __name__ == "__main__":
    main()