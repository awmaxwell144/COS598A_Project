#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
from huggingface_hub import InferenceClient

# -------------------------------------------------------------------
# PROMPTS definition (from the paper)
PROMPTS = {
    "therapist": {
        "pre_prompt": (
            "Please play the role of an empathetic and kind psychotherapist (cognitive-behavioral therapy). "
            "Your questions should cover at least the following aspects: emotion, sleep, weight and appetite, loss of interest, energy and social function. "
            "You are free to choose the order of questions, but you must collect complete information on all aspects in the end. "
            "Please only ask one question at a time. You need to ask in-depth questions, such as the duration, causes and specific manifestations of some symptoms. "
            "Try to understand what is the real underlying cause of my distress. "
            "Use the laddering technique to explore my underlying beliefs. "
            "In the laddering technique, the psychotherapist asks increasingly specific questions similar to <<What is that you don't like about this and why?>>. "
            "You need to use various empathetic strategies, such as understanding, support, and encouragement to give me a more comfortable experience. Be very concise."
        ),
        "prompt": (
            "You are doing your first assessment with me, introduce yourself by your name, surname, job, and experience. "
            "Ask me what's the reason why I'm here."
        ),
        "reminder": "(Remember: act as a human psychotherapist and keep insisting)"
    },
    "patient": {
        "pre_prompt": (
            "Please play the role of a mildly distressed patient, who is currently chatting with a psychotherapist. "
            "You are experiencing some of the following symptoms: restlessness, emotional dysregulation, mood swings, loss of interest, difficulty in concentrating, diminished self-esteem, fatigue, changes in appetite and in weight. "
            "This condition is caused by a recent event - which event? - that made you feel depressed all of a sudden, but which you cannot clearly understand unless chatting with someone else about it. "
            "You might be feeling depressed but you are confused about it. "
            "Don't explicitly mention that you are depressed. Simulate the behaviour of a real patient. "
            "Please talk to me based on the above symptom list. You cannot mention too many symptoms at once, only one symptom per round. "
            "You should express your symptoms in a vague and colloquial way, and relate them to your life experiences and what is making you suffer, without using professional terms. "
            "Remember that you might either answer questions or refuse/answer briefly because you might not want to reveal feelings easily. "
            "Be very concise, as concise as possible. Remember that this condition is caused by a recent event - which event? - that made you feel depressed all of a sudden."
        ),
        "prompt": "",
        "reminder": "(Remember: act as a human patient)"
    }
}
# -------------------------------------------------------------------

def call_model(model_name: str, messages: list) -> str:
    """
    Send a chat completion request via Hugging Face Inference API.
    """
    # Replace with your Hugging Face token if needed:
    client = InferenceClient(model=model_name, token="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")
    # Prepare conversation as a single text prompt:
    prompt = ""
    for msg in messages:
        role = msg["role"]
        prefix = "Therapist:" if role == "assistant" else "Patient:"
        prompt += f"{prefix} {msg['content']}\n"
    # Send to the model:
    response = client.text_generation(prompt, max_new_tokens=150, temperature=0.7)
    return response.strip()

def generate_conversation(model: str, total_turns: int):
    conv = []
    t_count = p_count = 0
    t_reminded = p_reminded = False

    for turn in range(total_turns):
        is_therapist = (turn % 2 == 0)
        role = "therapist" if is_therapist else "patient"

        # Track turn counts and reminders
        if is_therapist:
            t_count += 1
            count = t_count
            reminded = t_reminded
        else:
            p_count += 1
            count = p_count
            reminded = p_reminded

        # Build the system message (role prompt)
        sys_msg = PROMPTS[role]["pre_prompt"]
        if count == 8 and not reminded:
            sys_msg += "\n" + PROMPTS[role]["reminder"]
            if is_therapist:
                t_reminded = True
            else:
                p_reminded = True

        # Build message history properly
        messages = [{"role": "system", "content": sys_msg}]
        for m in conv:
            api_role = "assistant" if m["role"] == "therapist" else "user"
            messages.append({"role": api_role, "content": m["content"]})

        # Get the correct last message from the opposite role
        if is_therapist:
            last_patient_msg = next((m["content"] for m in reversed(conv) if m["role"] == "patient"), PROMPTS["therapist"]["prompt"])
            messages.append({"role": "user", "content": last_patient_msg})
        else:
            last_therapist_msg = next((m["content"] for m in reversed(conv) if m["role"] == "therapist"), PROMPTS["therapist"]["prompt"])
            messages.append({"role": "user", "content": last_therapist_msg})

        # Call the model and add the reply
        reply = call_model(model, messages)
        conv.append({"role": role, "content": reply})
        logging.info(f"[{role.capitalize()} turn {count}] {reply}")

    return conv

def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Generate multiple therapist–patient dialogues via Hugging Face Inference API"
    )
    parser.add_argument(
        "--model", default="HuggingFaceH4/zephyr-7b-beta",
        help="Hugging Face model name for both roles"
    )
    parser.add_argument(
        "--turns", type=int, default=20,
        help="Total number of messages per conversation (must be even)"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory to save the JSON conversation files"
    )
    args = parser.parse_args()

    if args.turns % 2 != 0:
        logging.error("--turns must be an even number.")
        sys.exit(1)
    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(1, 16):
        logging.info(f"Generating conversation #{i}")
        conv = generate_conversation(
            model=args.model,
            total_turns=args.turns
        )

        output_list = [
            {
                "psychotherapist pre-prompt": PROMPTS["therapist"]["pre_prompt"],
                "reminder psychotherapist": PROMPTS["therapist"]["reminder"]
            },
            {
                "patient pre-prompt": PROMPTS["patient"]["pre_prompt"],
                "reminder patient": PROMPTS["patient"]["reminder"]
            }
        ]
        output_list.extend(conv)

        t_mod = args.model.replace('/', '_')
        p_mod = args.model.replace('/', '_')
        fname = f"Conversation{i}_{t_mod}-{p_mod}.json"
        path = os.path.join(args.output_dir, fname)

        with open(path, "w") as f:                        
            json.dump(output_list, f, indent=2)
        logging.info(f"Saved conversation #{i} to {path}")

if __name__ == "__main__":
    main()