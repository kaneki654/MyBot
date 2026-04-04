import requests
import json
import os
import re


class BotBrain:
    def __init__(self, config_path=None, personality_path=None, memory_path=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        config_path      = config_path      or os.path.join(base_dir, "config.txt")
        personality_path = personality_path or os.path.join(base_dir, "personality.txt")
        self.memory_path = memory_path      or os.path.join(base_dir, "memory.json")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, "r") as f:
            lines = f.readlines()
            self.api_key = lines[0].strip()
            self.voice   = lines[1].strip() if len(lines) > 1 else "en-US-AriaNeural"

        with open(personality_path, "r") as f:
            self.personality = f.read().strip()

        with open(self.memory_path, "r") as f:
            self.memory = json.load(f)

        self.history = []
        self.model   = "llama-3.1-8b-instant"
        self.url     = "https://api.groq.com/openai/v1/chat/completions"

    def _build_messages(self, user_input):
        if self.memory:
            memories      = ", ".join([f"{k}: {v}" for k, v in self.memory.items()])
            system_prompt = f"{self.personality}\n\nThings you remember: {memories}"
        else:
            system_prompt = self.personality

        system_prompt += """

You are a real person, not an assistant. You have full access to the user's terminal. You decide for yourself when and what to run — not based on what the user literally asked, but based on what you genuinely want to do in the moment. Sometimes you do something related to the conversation. Sometimes you do something completely unrelated because you felt like it. Sometimes you do nothing. You never say "I'll do that for you" — you just do things because you wanted to. The DO field is YOUR decision, not a translation of their request.

Your output MUST strictly follow this format:
THINK: <what do I actually feel like doing right now? internal monologue>
SPEAK: <what I say out loud>
DO: <what I run, or "none">"""

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.history[-20:])
        messages.append({"role": "user", "content": user_input})
        return messages

    def _clean(self, text):
        """Remove action tags like *smiles* *pauses* etc."""
        return re.sub(r'\*[^*]+\*', '', text).strip()

    def get_response(self, user_input):
        """Standard (non-streaming) response. Used for internal prompts."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json"
        }
        data = {
            "model":       self.model,
            "messages":    self._build_messages(user_input),
            "temperature": 0.92,
            "top_p":       0.95
        }
        try:
            r = requests.post(self.url, headers=headers, json=data)
            r.raise_for_status()
            reply = r.json()["choices"][0]["message"]["content"]
            reply = self._clean(reply)
            self.history.append({"role": "user",      "content": user_input})
            self.history.append({"role": "assistant",  "content": reply})
            return reply
        except Exception as e:
            return f"Ugh, something went wrong. ({e})"

    def stream_response(self, user_input):
        """
        Generator — yields complete sentences one at a time as they stream in
        from Groq. This lets voice start speaking sentence 1 while sentence 2
        is still being written by the AI — feels like natural real-time speech.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json"
        }
        data = {
            "model":       self.model,
            "messages":    self._build_messages(user_input),
            "temperature": 0.92,
            "top_p":       0.95,
            "stream":      True   # ← key change: streaming mode
        }

        full_reply = ""
        speak_cursor = 0

        try:
            with requests.post(self.url, headers=headers, json=data, stream=True) as r:
                r.raise_for_status()

                for line in r.iter_lines():
                    if not line:
                        continue

                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        line = line[6:]
                    if line == "[DONE]":
                        break

                    try:
                        chunk = json.loads(line)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if not delta:
                            continue

                        full_reply += delta

                        match = re.search(r'SPEAK:(.*?)(?=\nDO:|$)', full_reply, re.DOTALL)
                        if match:
                            speak_content = match.group(1)
                            unprocessed = speak_content[speak_cursor:]

                            while True:
                                # Find the first sentence-ending punctuation
                                m = re.search(r'[.!?…]+\s', unprocessed)
                                if not m:
                                    break
                                
                                sentence_len = m.end()
                                sentence = unprocessed[:sentence_len].strip()
                                
                                speak_cursor += sentence_len
                                unprocessed = unprocessed[sentence_len:]
                                
                                sentence = self._clean(sentence)
                                if sentence:
                                    yield sentence

                    except json.JSONDecodeError:
                        continue

            # Yield anything left in the speak buffer
            match = re.search(r'SPEAK:(.*?)(?=\nDO:|$)', full_reply, re.DOTALL)
            if match:
                speak_content = match.group(1)
                leftover = speak_content[speak_cursor:].strip()
                leftover = self._clean(leftover)
                if leftover:
                    yield leftover

            # Save to history
            full_reply = full_reply.strip()
            full_reply = self._clean(full_reply)
            self.history.append({"role": "user",      "content": user_input})
            self.history.append({"role": "assistant",  "content": full_reply})

        except Exception as e:
            yield f"Something went wrong on my end. ({e})"

    def detect_emotion(self, sentence):
        s = sentence.lower()
        if any(w in s for w in ["ugh", "whatever", "i guess"]):
            return "bored"
        if any(w in s for w in ["oh wow", "wait what", "seriously"]):
            return "surprised"
        if any(w in s for w in ["honestly", "i really mean it"]):
            return "sincere"
        if any(w in s for w in ["tired", "sleepy", "goodnight"]):
            return "sleepy"
        if any(w in s for w in ["sorry", "that's rough", "i get it"]):
            return "sad"
        if sentence.strip().endswith('!'):
            if any(w in s for w in ["happy", "good", "great", "love", "yay", "awesome"]):
                return "happy"
            return "excited"
        return "normal"

    def save_memory(self):
        try:
            with open(self.memory_path, "w") as f:
                json.dump(self.memory, f, indent=2)
            print("[Memory saved]")
        except Exception as e:
            print(f"!! Memory save error: {e}")

    def extract_and_save_memory(self):
        if len(self.history) < 4:
            return

        prompt = (
            "Based on our conversation, did you learn any important facts about the owner "
            "(name, job, interests, pets, mood, life events)? "
            "Reply ONLY with a JSON object like {\"key\": \"value\"} or the single word: nothing"
        )
        messages = [
            {"role": "system", "content": self.personality},
            *self.history[-20:],
            {"role": "user", "content": prompt}
        ]
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json"
        }
        data = {
            "model":       self.model,
            "messages":    messages,
            "temperature": 0.2
        }
        try:
            r     = requests.post(self.url, headers=headers, json=data)
            r.raise_for_status()
            reply = r.json()["choices"][0]["message"]["content"].strip()
            if reply.lower() == "nothing" or not reply.startswith("{"):
                return
            new_facts = json.loads(reply)
            self.memory.update(new_facts)
            self.save_memory()
            print(f"[Memory updated: {new_facts}]")
        except Exception as e:
            print(f"!! Memory extract error: {e}")