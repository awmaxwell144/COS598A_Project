import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List
from tqdm import tqdm

from emoatlas import EmoScores

def aggregate_turns(turns: List[str]) -> str:
    """Concatenate a list of utterances into one string separated by newlines."""
    return " \n".join(turns)

def main():
    model = "qwen3"
    CONVERSATIONS_DIR = f'output/{model}'
    PLOTS_DIR = 'plots'
    os.makedirs(PLOTS_DIR, exist_ok=True)

    emos = EmoScores(language="english")
    records = []

    for filename in tqdm(os.listdir(CONVERSATIONS_DIR)):
        if not filename.endswith(".json"):
            continue

        conv_id = filename.replace(".json", "")
        with open(os.path.join(CONVERSATIONS_DIR, filename), "r") as f:
            conversation = json.load(f)

        # Gather turns by role
        speaker_turns = defaultdict(list)
        for i, turn in enumerate(conversation):
            role = turn.get("role", "").lower()
            text = turn.get("content", "").strip()
            if role in ["therapist", "patient"] and text:
                speaker_turns[role].append(text)

        for speaker in ["therapist", "patient"]:
            if speaker not in speaker_turns:
                continue

            corpus = aggregate_turns(speaker_turns[speaker])
            z = emos.zscores(corpus)
            records.append({"conversation": conv_id, "speaker": speaker, **z})

            emos.draw_plutchik(z, title=f"{conv_id} – {speaker}")
            plt.savefig(os.path.join(PLOTS_DIR, f"{conv_id}_{speaker}.png"))
            plt.close()

if __name__ == "__main__":
    main()