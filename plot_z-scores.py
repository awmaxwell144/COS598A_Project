import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define file paths
file_paths = {
    "gemma3": "calculated_data/average_zscores_gemma3.csv",
    "llama3.3": "calculated_data/average_zscores_llama3.3.csv",
    "gpt-3.5": "calculated_data/average_zscores_GPT.csv",
    "claude 3 Haiku": "calculated_data/average_zscores_Haiku.csv"
}

# Load and label each dataset
dfs = []
for model_name, path in file_paths.items():
    df = pd.read_csv(path)
    df["model"] = model_name
    dfs.append(df)

# Combine into one DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

# Melt the DataFrame for Seaborn
melted_df = pd.melt(
    combined_df,
    id_vars=["role", "model"],
    value_vars=["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"],
    var_name="emotion",
    value_name="z_score"
)

# Separate therapist and patient data
therapist_df = melted_df[melted_df["role"] == "therapist"]
patient_df = melted_df[melted_df["role"] == "patient"]

# Create side-by-side plots with fixed y-axis range
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

for ax, df, title in zip(
    axes,
    [therapist_df, patient_df],
    ["Therapist Responses", "Patient Responses"]
):
    sns.barplot(
        data=df,
        x="emotion",
        y="z_score",
        hue="model",
        ax=ax
    )
    ax.set_title(title, fontsize=18)
    ax.set_ylabel("Average Z-Score" if title == "Therapist Responses" else "", fontsize=16)
    ax.tick_params(labelsize=12)
    ax.set_xlabel("Emotion", fontsize=16)
    ax.set_ylim(-4, 5)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)


plt.tight_layout()
plt.savefig("plots/zscore_comparison_side_by_side.png")
plt.show()