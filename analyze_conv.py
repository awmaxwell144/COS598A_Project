"""Analyze mental-health dialogue folders with EmoAtlas
---------------------------------------------------
Given a root directory that contains sub-folders such as
```
<root>/Conversazione0_GPT
<root>/Conversazione1_GPT
 …
```
where each sub-folder includes one or more JSON files exported from the
CounseLLMe generation scripts (e.g. `Conversazione0_GPT-patient.json`),
this script will

1.   Recursively scan every conversation folder.
2.   Load **all** JSON files it finds.
3.   Separate *patient* utterances (`role == "user"`) and *therapist*
     utterances (`role == "assistant"`).
4.   Aggregate the text for each speaker into one string (or keep them
     separate per turn if `--by-turn` is passed).
5.   Use **EmoAtlas** to compute emotion *z-scores* for every speaker.
6.   Save a tidy CSV file with one row per `(conversation, speaker)` and
     eight Plutchik emotion z-scores.
7.   Optionally (`--plots`) create a PNG *Plutchik flower* for every
     `(conversation, speaker)`.

Usage
-----
```bash
pip install emoatlas pandas tqdm
python analyze_conversations.py /path/to/root_dir \
    --language en \
    --out_csv emotions.csv \
    --plots plots/
```

If you have never used **EmoAtlas** before, remember to install the
necessary NLP models:
```bash
python -m spacy download en_core_web_lg   # or it_core_news_lg, etc.
python - <<<'import nltk, sys; nltk.download("wordnet")'
```
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Third-party library (https://github.com/MassimoStel/emoatlas)
from emoatlas import EmoScores

###############################################################################
# ---------------------------  helpers  ------------------------------------- #
###############################################################################

def iter_json_files(root: Path):
    """Yield every ``*.json`` file contained *anywhere* under *root*."""
    yield from (p for p in root.rglob("*.json") if p.is_file())


def load_dialogue(path: Path) -> Tuple[List[str], List[str]]:
    """Read *one* CounseLLMe JSON export, returning *(patient_utts, therapist_utts)*.

    The convention in the example export is:
    * ``role == 'user'``      → patient
    * ``role == 'assistant'`` → therapist
    Any other roles (system messages, prompts) are ignored.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    patient, therapist = [], []
    for turn in data:
        role = turn.get("role")
        if role == "user":
            patient.append(turn.get("content", ""))
        elif role == "assistant":
            therapist.append(turn.get("content", ""))
    return patient, therapist


def aggregate_turns(turns: List[str]) -> str:
    """Concatenate a list of utterances into one string separated by newlines."""
    return " \n".join(turns)

###############################################################################
# ----------------------------  main  --------------------------------------- #
###############################################################################

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Emotion analysis of CounseLLMe dialogues using EmoAtlas",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("root", type=Path, help="Root directory containing conversation folders")
    parser.add_argument("--language", default="english", help="Language for EmoAtlas (english / italian …)")
    parser.add_argument("--out_csv", type=Path, default=Path("emoatlas_results.csv"), help="Where to write the CSV summary")
    parser.add_argument("--by-turn", dest="by_turn", action="store_true", help="Analyze each utterance separately instead of the whole conversation")
    parser.add_argument("--plots", type=Path, default = "emoatlas_results",help="Directory where Plutchik flower PNGs will be saved (optional)")
    args = parser.parse_args()

    # Ensure plots directory exists
    if args.plots:
        args.plots.mkdir(parents=True, exist_ok=True)

    # Initialise EmoAtlas once.
    emos = EmoScores(language=args.language)

    records: List[Dict[str, object]] = []

    json_files = list(iter_json_files(args.root))
    if not json_files:
        raise SystemExit(f"No JSON files found under {args.root}")

    for json_path in tqdm(json_files, desc="Processing conversations"):
        conv_id = json_path.parent.name  # e.g. Conversazione0_GPT
        patient_utts, therapist_utts = load_dialogue(json_path)

        for speaker, turns in (("patient", patient_utts), ("therapist", therapist_utts)):
            if not turns:
                continue  # skip empty role in this file

            if args.by_turn:
                for i, utt in enumerate(turns, start=1):
                    z = emos.zscores(utt)
                    records.append({"conversation": conv_id, "speaker": speaker, "turn": i, **z})

                    if args.plots:
                        emos.draw_plutchik(z, title=f"{conv_id} – {speaker} turn {i}")
                        plt.savefig(args.plots / f"{conv_id}_{speaker}.png")
                        plt.close()
            else:
                corpus = aggregate_turns(turns)
                z = emos.zscores(corpus)
                records.append({"conversation": conv_id, "speaker": speaker, **z})

                if args.plots:
                    emos.draw_plutchik(z, title=f"{conv_id} – {speaker}")
                    plt.savefig(args.plots / f"{conv_id}_{speaker}.png")
                    plt.close()
    # Save tidy CSV.
    df = pd.DataFrame.from_records(records)
    df.to_csv(args.out_csv, index=False)
    print(f"✓ Saved emotion scores → {args.out_csv}")


if __name__ == "__main__":
    main()
