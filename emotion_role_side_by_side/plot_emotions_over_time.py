import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load the data
df = pd.read_csv("calculated_data/normalized_emotion_frequencies_by_step.csv")

# Create output directory
output_dir = "emotion_role_side_by_side"
os.makedirs(output_dir, exist_ok=True)

# Get all unique models
models = df["model"].unique()

# Plot each model's therapist vs patient emotion trajectories
for model in models:
    model_df = df[df["model"] == model]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    roles = ["therapist", "patient"]

    for ax, role in zip(axes, roles):
        role_df = model_df[model_df["role"] == role]
        sns.lineplot(
            data=role_df,
            x="step",
            y="normalized_freq",
            hue="emotion",
            marker="o",
            errorbar=None,  
            ax=ax
        )
        ax.set_title(f"{role.capitalize()} (Model: {model})")
        ax.set_xlabel("Conversation Step")
        ax.set_ylabel("Normalized Emotion Frequency")
        ax.grid(True)
        ax.legend(title="Emotion", bbox_to_anchor=(1.05, 1), loc='upper left')

    fig.suptitle(f"Emotion Trajectories by Role — {model}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    filename = f"{model}_therapist_vs_patient.png".replace(" ", "_").lower()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

print(f"Side-by-side plots saved in '{output_dir}/'")