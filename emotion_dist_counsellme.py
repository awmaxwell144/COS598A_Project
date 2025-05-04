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
    model = "GPT"
    CONVERSATIONS_DIR = 'Conv-GPT-patients'
    PLOTS_DIR = 'plots'
    os.makedirs(PLOTS_DIR, exist_ok=True)

    emos = EmoScores(language="english")

    role_sums = defaultdict(lambda: defaultdict(float))  # e.g., role_sums['patient']['joy'] = total
    role_counts = defaultdict(int)  # e.g., role_counts['patient'] = 17 conversations

    for filename in tqdm(os.listdir(CONVERSATIONS_DIR)):
        if not filename.endswith(".json"):
            continue

        with open(os.path.join(CONVERSATIONS_DIR, filename), "r") as f:
            conversation = json.load(f)

        # Filter only the actual dialogue turns with "role" and "content"
        dialogue_turns = [turn for turn in conversation if "role" in turn and "content" in turn]

        speaker_turns = defaultdict(list)

        for turn in dialogue_turns:
            role_raw = turn.get("role", "").lower()
            content = turn.get("content", "").strip()
            if not content:
                continue

            role_map = {"user": "patient", "assistant": "therapist"}
            role = role_map.get(role_raw)
            if role:
                speaker_turns[role].append(content)

        for speaker in ["therapist", "patient"]:
            if speaker not in speaker_turns or not speaker_turns[speaker]:
                continue

            full_text = aggregate_turns(speaker_turns[speaker])
            z = emos.zscores(full_text)

            for emotion, value in z.items():
                role_sums[speaker][emotion] += value
            role_counts[speaker] += 1

    # Determine all possible emotions from the z-score keys
    all_emotions = sorted(set(emotion for scores in role_sums.values() for emotion in scores))

    # Write average z-scores to CSV
    csv_path = os.path.join(PLOTS_DIR, "average_zscores.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["role"] + all_emotions)
        writer.writeheader()

        for speaker in ["therapist", "patient"]:
            if role_counts[speaker] == 0:
                print(f"No data for role: {speaker}")
                continue

            avg_z = {
                emotion: role_sums[speaker][emotion] / role_counts[speaker]
                for emotion in all_emotions
            }

            writer.writerow({"role": speaker, **avg_z})

            # Create and save Plutchik wheel
            emos.draw_plutchik(avg_z, title=f"Average – {speaker}")
            plt.savefig(os.path.join(PLOTS_DIR, f"average_{speaker}_{model}.png"))
            plt.close()

    print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    main()