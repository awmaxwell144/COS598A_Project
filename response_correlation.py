import os
import json
import csv
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List
from tqdm import tqdm
from scipy.stats import pearsonr
import numpy as np

from emoatlas import EmoScores

def main():
    model = "llama3.3"
    CONVERSATIONS_DIR = f'conversations/{model}'
    PLOTS_DIR = 'plots'
    os.makedirs(PLOTS_DIR, exist_ok=True)

    emos = EmoScores(language="english")

    paired_emotions = defaultdict(list)  # {emotion: list of (patient_val, therapist_val)}

    for filename in tqdm(os.listdir(CONVERSATIONS_DIR)):
        if not filename.endswith(".json"):
            continue

        with open(os.path.join(CONVERSATIONS_DIR, filename), "r") as f:
            conversation = json.load(f)

        # Create paired turns: patient -> therapist
        prev_patient_text = None
        for turn in conversation:
            role = turn.get("role", "").lower()
            text = turn.get("content", "").strip()
            if not text:
                continue

            if role == "user":
                prev_patient_text = text
            elif role == "assistant" and prev_patient_text:
                # Compute emotion scores for both
                patient_z = emos.zscores(prev_patient_text)
                therapist_z = emos.zscores(text)

                # Store paired scores
                for emotion in patient_z:
                    paired_emotions[emotion].append((patient_z[emotion], therapist_z[emotion]))

                prev_patient_text = None  # Reset until next patient turn

    # Analyze correlations
    correlation_results = []
    for emotion, pairs in paired_emotions.items():
        if len(pairs) > 1:
            patient_vals, therapist_vals = zip(*pairs)
            corr, pval = pearsonr(patient_vals, therapist_vals)
            correlation_results.append({
                "emotion": emotion,
                "pearson_corr": corr,
                "p_value": pval,
                "n_pairs": len(pairs)
            })

    # Save results
    csv_path = os.path.join(PLOTS_DIR, f"correlation_patient_to_therapist_{model}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["emotion", "pearson_corr", "p_value", "n_pairs"])
        writer.writeheader()
        writer.writerows(correlation_results)

    print(f"Correlation results saved to {csv_path}")

if __name__ == "__main__":
    main()