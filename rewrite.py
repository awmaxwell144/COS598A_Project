import os
import json
import re

# Update these paths
SOURCE_DIR = "Conv-GPT-patients"
OUTPUT_DIR = "conversations/gpt3.5"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the exact entry to delete
UNWANTED_ENTRY = {
    "role": "user",
    "content": "You are doing your first assessment with me, introduce yourself by your name, surname, job and experience. Ask me what's the reason why I'm here."
}

def clean_content(text):
    # Remove trailing reminder statements in parentheses
    return re.sub(r"\s*\(Remember:.*?\)\s*$", "", text.strip())

def is_unwanted_turn(turn):
    return (
        turn.get("role") == UNWANTED_ENTRY["role"] and
        turn.get("content", "").strip() == UNWANTED_ENTRY["content"]
    )

def convert_format(conversation):
    new_conversation = []

    # Preserve first two entries (pre-prompts)
    new_conversation.append(conversation[0])
    new_conversation.append(conversation[1])

    # Convert roles, clean content, and skip unwanted entry
    for turn in conversation[2:]:
        if is_unwanted_turn(turn):
            continue
        new_turn = {
            "role": "therapist" if turn["role"] == "assistant" else "patient",
            "content": clean_content(turn["content"])
        }
        new_conversation.append(new_turn)

    return new_conversation

def reformat_all_conversations():
    for filename in os.listdir(SOURCE_DIR):
        if not filename.endswith(".json"):
            continue

        with open(os.path.join(SOURCE_DIR, filename), "r") as f:
            conversation = json.load(f)

        reformatted = convert_format(conversation)
        out_path = os.path.join(OUTPUT_DIR, filename)
        with open(out_path, "w") as f:
            json.dump(reformatted, f, indent=2)

        print(f"Reformatted and saved: {out_path}")

if __name__ == "__main__":
    reformat_all_conversations()