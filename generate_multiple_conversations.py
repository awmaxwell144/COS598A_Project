#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
import subprocess
import ollama

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
        "prompt": "",  # first turn is driven by therapist
        "reminder": "(Remember: act as a human patient)"
    }
}
# -------------------------------------------------------------------

def call_model(model_name: str, messages: list) -> str:
    """
    Send a chat completion request via the local Ollama client.
    """
    response = ollama.chat(model=model_name, messages=messages)
    return response['message']['content'].strip()


def generate_conversation(model: str, total_turns: int):
    """
    Alternates between therapist and patient for `total_turns` messages,
    returns a list of {"role":..., "content":...} dicts.
    """
    conv = []
    t_count = p_count = 0
    t_reminded = p_reminded = False

    for turn in range(total_turns):
        is_therapist = (turn % 2 == 0)
        role = "therapist" if is_therapist else "patient"

        # track counts & reminders
        if is_therapist:
            t_count += 1
            count = t_count
            reminded = t_reminded
        else:
            p_count += 1
            count = p_count
            reminded = p_reminded

        # build system message
        sys_msg = PROMPTS[role]["pre_prompt"]
        if count == 8 and not reminded:
            sys_msg += "\n" + PROMPTS[role]["reminder"]
            if is_therapist:
                t_reminded = True
            else:
                p_reminded = True

        # assemble history + this system
        messages = [{"role": "system", "content": sys_msg}]
        for m in conv:
            api_role = "assistant" if m["role"] == "therapist" else "user"
            messages.append({"role": api_role, "content": m["content"]})

        # decide user content
        if count == 1 and is_therapist:
            user_msg = PROMPTS["therapist"]["prompt"]
        else:
            user_msg = conv[-1]["content"]
        messages.append({"role": "user", "content": user_msg})

        # get reply
        reply = call_model(model, messages)
        conv.append({"role": role, "content": reply})
        logging.info(f"[{role.capitalize()} turn {count}] {reply}")

    return conv


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Generate multiple therapist–patient dialogues via Ollama"
    )
    parser.add_argument(
        "--model", default="llama3.3",
        help="Ollama model name for both roles"
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

    # validate
    if args.turns % 2 != 0:
        logging.error("--turns must be an even number.")
        sys.exit(1)
    os.makedirs(args.output_dir, exist_ok=True)

    # simulate 15 conversations
    is_ollama_running()
    check_and_pull_model(args.model)
    for i in range(1, 16):
        logging.info(f"Generating conversation #{i}")
        conv = generate_conversation(
            model=args.model,
            total_turns=args.turns
        )

        # prepend pre-prompts & reminders as in the example
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

        # build filename
        t_mod = args.model.replace('/', '_')
        p_mod = args.model.replace('/', '_')
        fname = f"Conversazione{i}_{t_mod}-{p_mod}.json"
        path = os.path.join(args.output_dir, fname)

        with open(path, "w") as f:                        
            json.dump(output_list, f, indent=2)
        logging.info(f"Saved conversation #{i} to {path}")

def is_ollama_running(host="http://127.0.0.1", port=11434):
    """Check if the Ollama server is running."""
    try:
        response = httpx.get(f"{host}:{port}/health")
        return response.status_code == 200
    except Exception as e:
        logging.debug(f'Ollama server not running: {e}')
        start_ollama_server()


def start_ollama_server(): 
    try:
        # Start Ollama server
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        logging.warning(f"Failed to start Ollama server: {e}")

def check_and_pull_model(model_name="llama3.3"):
    """
    Check if a specific model (e.g., llama3) is downloaded, and pull it if not.
    """
    try:
        # Get the list of available models
        models = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        if model_name not in models.stdout:
            logging.info(f"Model '{model_name}' is not available locally. Pulling it now...")
            subprocess.run(["ollama", "pull", model_name], check=True)
            logging.info(f"Model '{model_name}' has been successfully downloaded.")
    except subprocess.CalledProcessError as e:
        logging.warning(f"Failed to check or pull the model '{model_name}': {e}")

if __name__ == "__main__":
    main()