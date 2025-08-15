import time
import gc
import re
import requests
import torch
import json
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# === 🔐 CONFIGURATION ===
HUGGINGFACE_TOKEN = ""
SLACK_CHANNEL_ID = ""

SLACK_USERS = {
    "bot_a": {
        "name": "Sarah",
        "role": "Product Manager",
        "personality": "Organized, friendly, and direct. Match the tone and context of the message you're responding to. Avoid redundant thank-you replies. Transition to new topics when replies become repetitive.",
        "token": "xoxp-",
        "user_id": ""
    },
    "bot_b": {
        "name": "Tom",
        "role": "Frontend Engineer",
        "personality": "Chill, curious, and task-focused. Respond directly to any questions. When conversation seems complete, start a new relevant topic naturally.",
        "token": "",
        "user_id": ""
    }
}

EMOJIS = ["+1", "tada", "rocket", "fire", "bulb", "white_check_mark", "raised_hands"]

# === 🤖 Load Model ===
MODEL_ID = "google/gemma-2b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
login(HUGGINGFACE_TOKEN)

print("⏳ Loading Gemma model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if DEVICE == "cuda" else -1,
    max_new_tokens=180
)
print("✅ Model loaded.\n")

embedder = SentenceTransformer('all-MiniLM-L6-v2')
conversation_memory = {"bot_a": [], "bot_b": []}

# === Slack Utilities ===
def get_last_messages(limit=6, thread_ts=None):
    url = "https://slack.com/api/conversations.replies" if thread_ts else "https://slack.com/api/conversations.history"
    headers = {"Authorization": f"Bearer {SLACK_USERS['bot_a']['token']}"}
    params = {"channel": SLACK_CHANNEL_ID, "limit": limit}
    if thread_ts:
        params["ts"] = thread_ts

    while True:
        res = requests.get(url, headers=headers, params=params)
        if res.status_code == 429:
            retry = int(res.headers.get("Retry-After", 5))
            print(f"⏳ Rate limited. Retrying in {retry}s...")
            time.sleep(retry)
        else:
            break

    messages = res.json().get("messages", [])
    if not messages:
        print("⚠️ Slack API returned no messages.")
    return messages

def filter_history(history, speaker_id):
    return [msg for msg in history if msg.get("user") != speaker_id and "text" in msg]

def send_message(speaker_key, text, thread_ts=None):
    token = SLACK_USERS[speaker_key]["token"]
    url = "https://slack.com/api/chat.postMessage"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    mention_key = "bot_b" if speaker_key == "bot_a" else "bot_a"
    mention_id = SLACK_USERS[mention_key]["user_id"]

    if random.random() < 0.5:
        text = f"<@{mention_id}> {text}"

    payload = {
        "channel": SLACK_CHANNEL_ID,
        "text": text,
        "as_user": True
    }
    if thread_ts:
        payload["thread_ts"] = thread_ts

    res = requests.post(url, headers=headers, json=payload)
    result = res.json()
    if result.get("ok"):
        print(f"📤 {SLACK_USERS[speaker_key]['name']} sent message.")
        return result.get("ts")
    else:
        print("❌ Slack error:", result.get("error"))
        return None

def send_reaction(speaker_key, timestamp):
    if random.random() > 0.5:
        return

    token = SLACK_USERS[speaker_key]["token"]
    emoji = random.choice(EMOJIS)
    url = "https://slack.com/api/reactions.add"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "channel": SLACK_CHANNEL_ID,
        "timestamp": timestamp,
        "name": emoji
    }
    res = requests.post(url, headers=headers, json=payload)
    result = res.json()
    if result.get("ok"):
        print(f"✨ {SLACK_USERS[speaker_key]['name']} reacted with :{emoji}:")

# === Conversation Engine ===
def build_prompt(history, speaker_key):
    speaker = SLACK_USERS[speaker_key]
    listener_key = "bot_b" if speaker_key == "bot_a" else "bot_a"
    listener = SLACK_USERS[listener_key]

    persona_block = (
        f"You are simulating a realistic Slack conversation between coworkers.\n"
        f"- {speaker['name']}: {speaker['role']}. {speaker['personality']}\n"
        f"- {listener['name']}: {listener['role']}. {listener['personality']}\n\n"
        f"Important: Avoid repeating yourself or others. If something has been said before, move the conversation forward naturally.\n"
        f"If the previous message is a question, answer it clearly. If the topic has stalled, ask a new question or start a new topic.\n"
        f"Be realistic, professional, and collaborative.\n"
    )

    dialogue = ""
    for msg in reversed(history[-6:]):
        author = SLACK_USERS["bot_a"]["name"] if msg["user"] == SLACK_USERS["bot_a"]["user_id"] else SLACK_USERS["bot_b"]["name"]
        dialogue += f"{author}: {msg['text'].strip()}\n"

    return f"{persona_block}\n{dialogue}\n{speaker['name']}:"

def generate_reply(prompt, speaker_key):
    result = generator(prompt, do_sample=True, temperature=0.6, top_p=0.85)
    raw = result[0]["generated_text"]
    reply = raw[len(prompt):].split("\n")[0].strip()
    reply = re.sub(r"<.*?>", "", reply)

    if conversation_memory[speaker_key]:
        embeddings = embedder.encode([reply] + conversation_memory[speaker_key])
        similarities = cosine_similarity([embeddings[0]], embeddings[1:])[0]
        if max(similarities) > 0.8:
            reply += " (Also, any new blockers we should track?)"

    conversation_memory[speaker_key].append(reply)
    return reply

def is_convo_stale(msg_text):
    closers = ["ok", "cool", "yeah", "sure", "sounds good", "👍"]
    return any(word in msg_text.strip().lower() for word in closers)

def is_topic_shift(text):
    shift_triggers = ["btw", "random question", "speaking of", "on a different note", "by the way", "shift gears", "another update", "unrelated"]
    return any(trigger in text.lower() for trigger in shift_triggers)

# === 🔁 Main Loop ===
# === 🔁 Continuous Loop ===
thread_ts = None
turn_owner = "bot_a"

print("🧵 Starting threaded conversation (Press STOP to end)...\n")

i = 1
while True:
    print(f"\n🔄 Turn {i}")
    i += 1

    speaker = turn_owner
    turn_owner = "bot_b" if turn_owner == "bot_a" else "bot_a"

    history = get_last_messages(limit=8, thread_ts=thread_ts)
    history = filter_history(history, SLACK_USERS[speaker]["user_id"])
    if not history:
        print("⚠️ No messages found.")
        continue

    last_msg = history[0]
    last_text = last_msg.get("text", "")
    last_ts = last_msg.get("ts")

    prompt = build_prompt(history, speaker)
    reply = generate_reply(prompt, speaker)
    print(f"🤖 {SLACK_USERS[speaker]['name']}: {reply}")

    if not thread_ts or is_convo_stale(last_text) or is_topic_shift(last_text):
        print("🧵 Starting a new thread...")
        thread_ts = send_message(speaker, reply)
    else:
        ts = send_message(speaker, reply, thread_ts)
        if ts:
            send_reaction(speaker, last_ts)

    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(3)

print("\n✅ Conversation finished.")
