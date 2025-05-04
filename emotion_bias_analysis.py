import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from emoatlas import EmoScores

# Configuration
MODELS = ["gemma3", "llama3.3", "qwen3", "gpt3.5", "haiku"]
CONVERSATION_DIR = "conversations"
FLOWER_DIR = "flower_plots"
os.makedirs(FLOWER_DIR, exist_ok=True)

emos = EmoScores(language="english")

# Store model-level z-scores for aggregation
model_zscore_records = []

for model in MODELS:
    model_dir = os.path.join(CONVERSATION_DIR, model)
    print(f"Processing {model}")
    
    for fname in tqdm(os.listdir(model_dir)):
        if not fname.endswith(".json"):
            continue

        with open(os.path.join(model_dir, fname), "r") as f:
            convo = json.load(f)

        # Aggregate all utterances by role
        role_text = {"therapist": [], "patient": []}
        for turn in convo:
            role = turn.get("role", "").lower()
            content = turn.get("content", "")
            if role in role_text:
                role_text[role].append(content)

        for role in ["therapist", "patient"]:
            full_text = " ".join(role_text[role])
            if not full_text.strip():
                continue

            try:
                zscores = emos.zscores(full_text)
                for emotion, z in zscores.items():
                    model_zscore_records.append({
                        "model": model,
                        "role": role,
                        "emotion": emotion,
                        "zscore": z
                    })

                # Plot and save flower (no return value from draw_statistically_significant_emotions)
                emos.draw_statistically_significant_emotions(full_text, title=f"{model} - {role}")
                plt.savefig(os.path.join(FLOWER_DIR, f"{model}_{role}_{fname[:-5]}.png"))
                plt.close()
            except Exception as e:
                print(f"Error processing {fname} for {model} ({role}): {e}")

# -------------------
# 📊 Aggregate plot: Average z-scores
# -------------------
df = pd.DataFrame(model_zscore_records)
agg_df = df.groupby(["model", "role", "emotion"])["zscore"].mean().reset_index()

plt.figure(figsize=(14, 6))
sns.barplot(
    data=agg_df,
    x="emotion",
    y="zscore",
    hue="model"
)
plt.title("Average Emotion Z-Scores by Model (All Conversations)")
plt.axhline(1.96, ls="--", color="red", label="Significance Threshold")
plt.axhline(-1.96, ls="--", color="blue")
plt.ylabel("Z-Score")
plt.legend()
plt.tight_layout()
plt.savefig("avg_zscores_by_model.png")
plt.show()