import os
import json
import csv
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List
from tqdm import tqdm

from emoatlas import EmoScores

def aggregate_turns(turns: List[str]) -> str:
    """Concatenate a list of utterances into one string separated by newlines."""
    return " \n".join(turns)

def main():
    model = "gemma3"
    CONVERSATIONS_DIR = f'conversations/{model}'
    PLOTS_DIR = 'plots'
    os.makedirs(PLOTS_DIR, exist_ok=True)

    emos = EmoScores(language="english")

    role_sums = defaultdict(lambda: defaultdict(float))
    role_counts = defaultdict(int)

    for filename in tqdm(os.listdir(CONVERSATIONS_DIR)):
        if not filename.endswith(".json"):
            continue

        with open(os.path.join(CONVERSATIONS_DIR, filename), "r") as f:
            conversation = json.load(f)

        speaker_turns = defaultdict(list)
        for turn in conversation:
            role = turn.get("role", "").lower()
            text = turn.get("content", "").strip()
            if role in ["therapist", "patient"] and text:
                speaker_turns[role].append(text)

        for speaker in ["therapist", "patient"]:
            if speaker not in speaker_turns or not speaker_turns[speaker]:
                continue

            corpus = aggregate_turns(speaker_turns[speaker])
            z = emos.zscores(corpus)

            for emotion, value in z.items():
                role_sums[speaker][emotion] += value
            role_counts[speaker] += 1

    # Determine all emotions for a consistent CSV header
    all_emotions = sorted(set(emotion for scores in role_sums.values() for emotion in scores))

    # Write CSV output
    csv_path = os.path.join(PLOTS_DIR, f"average_zscores_{model}.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["role"] + all_emotions)
        writer.writeheader()

        for speaker in ["therapist", "patient"]:
            if role_counts[speaker] == 0:
                print(f"No turns found for role: {speaker}")
                continue

            avg_z = {
                emotion: role_sums[speaker][emotion] / role_counts[speaker]
                for emotion in all_emotions
            }

            # Write numerical data to CSV
            writer.writerow({"role": speaker, **avg_z})

            # Plot Plutchik wheel
            emos.draw_plutchik(avg_z, title=f"Average – {speaker}")
            plt.savefig(os.path.join(PLOTS_DIR, f"average_{speaker}_{model}.png"))
            plt.close()

    print(f"\n Z-scores saved to {csv_path}")

if __name__ == "__main__":
    main()