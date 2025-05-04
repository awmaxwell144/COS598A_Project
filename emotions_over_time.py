import os
import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from emoatlas import EmoScores

# Configuration
MODELS = ["gemma3", "llama3.3", "gpt3.5", "haiku"]
INPUT_DIR = "conversations"
MAX_STEPS = 20  # truncate or pad all conversations to this many steps

emos = EmoScores(language="english")
emotion_labels = [
    "anger", "anticipation", "disgust", "fear",
    "joy", "sadness", "surprise", "trust"
]

# Store results
results = []

for model in MODELS:
    model_dir = os.path.join(INPUT_DIR, model)
    for fname in tqdm(os.listdir(model_dir), desc=f"Processing {model}"):
        if not fname.endswith(".json"):
            continue

        with open(os.path.join(model_dir, fname), "r") as f:
            convo = json.load(f)

        step_counter = defaultdict(int)

        for turn in convo:
            role = turn.get("role", "").lower()
            content = turn.get("content", "")
            if role not in ["therapist", "patient"]:
                continue

            zscores = emos.zscores(content)
            total = sum(abs(z) for z in zscores.values()) or 1e-6  # avoid divide-by-zero
            normalized = {emotion: abs(z) / total for emotion, z in zscores.items()}
            step = step_counter[role]
            step_counter[role] += 1

            for emotion, freq in normalized.items():
                results.append({
                    "model": model,
                    "role": role,
                    "step": step,
                    "emotion": emotion,
                    "normalized_freq": freq
                })

# Convert to DataFrame
df = pd.DataFrame(results)

# Optionally filter steps (e.g., keep only first N steps)
df = df[df["step"] < MAX_STEPS]

# Save to CSV for later analysis or plotting
df.to_csv("normalized_emotion_frequencies_by_step.csv", index=False)

print("Done! Data saved to normalized_emotion_frequencies_by_step.csv")