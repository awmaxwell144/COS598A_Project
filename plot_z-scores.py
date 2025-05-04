import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define file paths
file_paths = {
    "gemma3": "plots/average_zscores_gemma3.csv",
    "llama3.3": "plots/average_zscores_llama3.3.csv",
    "qwen3": "plots/average_zscores_qwen3.csv",
    "gpt-3.5": "plots/average_zscores_GPT.csv"
}

# Ensure output directory exists
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load and label each dataset
dfs = []
for model_name, path in file_paths.items():
    df = pd.read_csv(path)
    df["model"] = model_name
    dfs.append(df)

# Combine all into one DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

# Melt the data for seaborn plotting
melted_df = pd.melt(
    combined_df,
    id_vars=["role", "model"],
    value_vars=["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"],
    var_name="emotion",
    value_name="z_score"
)

# Set seaborn style and font sizes
sns.set(style="whitegrid", font_scale=1.5)

# Plot 1: Therapist roles
therapist_df = melted_df[melted_df["role"] == "therapist"]
plt.figure(figsize=(16, 8))
sns.barplot(
    data=therapist_df,
    x="emotion",
    y="z_score",
    hue="model",
    palette="tab10",
    ci=None
)
plt.title("Emotion Z-Scores for Therapists Across LLMs", fontsize=20)
plt.xlabel("Emotion", fontsize=18)
plt.ylabel("Average Z-Score", fontsize=18)
plt.xticks(rotation=45, fontsize=14)
plt.yticks(fontsize=14)
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14, title_fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "therapist_zscores.png"))
plt.close()

# Plot 2: Patient roles
patient_df = melted_df[melted_df["role"] == "patient"]
plt.figure(figsize=(16, 8))
sns.barplot(
    data=patient_df,
    x="emotion",
    y="z_score",
    hue="model",
    palette="tab10",
    ci=None
)
plt.title("Emotion Z-Scores for Patients Across LLMs", fontsize=20)
plt.xlabel("Emotion", fontsize=22)
plt.ylabel("Average Z-Score", fontsize=22)
plt.xticks(rotation=45, fontsize=18)
plt.yticks(fontsize=14)
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16, title_fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "patient_zscores.png"))
plt.close()

print("Plots saved to:", os.path.abspath(PLOTS_DIR))