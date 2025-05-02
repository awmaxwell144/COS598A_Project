import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

# Replace this with your actual EmoAtlas classifier interface
from emoatlas import EmoClassifier  # hypothetical import, adjust if needed

# Set up the classifier
classifier = EmoClassifier(model_name="emoscope-base")  # adjust if needed

model = "llama3.3"

# Path to directory with JSON conversation files
CONVERSATIONS_DIR = "output/llama3.3"

# Emotion counts
emotion_counts = {
    "therapist": defaultdict(int),
    "patient": defaultdict(int),
}

# Process each file
for filename in tqdm(os.listdir(CONVERSATIONS_DIR)):
    if not filename.endswith(".json"):
        continue

    with open(os.path.join(CONVERSATIONS_DIR, filename), "r") as f:
        conversation = json.load(f)

    for turn in conversation:
        for role in ["therapist", "patient"]:
            utterance = turn.get(role)
            if not utterance:
                continue

            result = classifier.predict(utterance)
            if isinstance(result, dict) and "label" in result:
                emotion = result["label"]
            elif isinstance(result, str):
                emotion = result
            else:
                continue

            emotion_counts[role][emotion] += 1

# Plotting
roles = ["therapist", "patient"]

for role in roles:
    emotions = list(emotion_counts[role].keys())
    counts = [emotion_counts[role][e] for e in emotions]

    plt.figure()
    plt.bar(emotions, counts)
    plt.title(f"Emotion Distribution for {role.capitalize()}")
    plt.xlabel("Emotion")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"emotion_distribution/{model}_{role}.png")
    plt.close()