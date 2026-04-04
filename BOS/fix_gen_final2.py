import re
with open("train_ai.py", "r") as f:
    text = f.read()

# Fix spacing in prompt to ensure word level tokenization maps it
text = text.replace('prompt = f"User: {user_input}\\nAI: "', 'prompt = f"User: {user_input}\\nAI:"')

with open("train_ai.py", "w") as f:
    f.write(text)
