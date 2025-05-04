#!/usr/bin/env python3
"""
Stage‑1 script:
  • Walks through the four conversation folders
  • Computes average emotion z‑scores per (model, condition, role, emotion)
  • Writes results to data/emo_zscores.csv
"""

import os
import json
from collections import defaultdict
from typing import List

import pandas as pd
from tqdm import tqdm
from emoatlas import EmoScores

# ---------- helpers ---------------------------------------------------
def aggregate_turns(turns: List[str]) -> str:
    return " \n".join(turns)

def process_directory(model: str, condition: str, path: str, emos: EmoScores):
    role_sums = defaultdict(lambda: defaultdict(float))
    role_counts = defaultdict(int)

    for fname in tqdm(os.listdir(path), desc=f"{model} – {condition}"):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(path, fname), "r") as f:
            conversation = json.load(f)

        turns_by_role = defaultdict(list)
        for turn in conversation:
            role  = turn.get("role", "").lower()
            text  = turn.get("content", "")
            if role and text:
                turns_by_role[role].append(text)

        for role, turns in turns_by_role.items():
            zscores = emos.zscores(aggregate_turns(turns))
            for emo, z in zscores.items():
                role_sums[role][emo] += z
            role_counts[role] += 1

    rows = []
    for role in role_sums:
        for emo in role_sums[role]:
            rows.append(
                dict(model=model, condition=condition, role=role,
                     emotion=emo,
                     zscore=role_sums[role][emo] / role_counts[role])
            )
    return rows

# ---------- main ------------------------------------------------------
def main():
    emos = EmoScores(language="english")

    configs = [
        ("gemma3",   "depression", "conversations/gemma3"),
        ("gemma3",   "anxiety",    "anxiety_conversations/gemma3"),
        ("llama3.3", "depression", "conversations/llama3.3"),
        ("llama3.3", "anxiety",    "anxiety_conversations/llama3.3"),
    ]

    data = []
    for mdl, cond, pth in configs:
        data.extend(process_directory(mdl, cond, pth, emos))

    os.makedirs("data", exist_ok=True)
    pd.DataFrame(data).to_csv("data/emo_zscores.csv", index=False)
    print("✅  Saved averages → data/emo_zscores.csv")

if __name__ == "__main__":
    main()