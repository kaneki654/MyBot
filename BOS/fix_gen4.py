with open("train_ai.py", "r") as f:
    text = f.read()

# Fix prompt spacing
text = text.replace(
    'prompt = f"User: {user_input}\\nAI:"',
    'prompt = f"User: {user_input}\\nAI: "'
)
# Make it stop when it hits a user line properly
text = text.replace(
    'if word == "\\n" or "User" in word or i >= length - 1:',
    'if "User" in word or "User:" in word or i >= length - 1:'
)
# Remove the newline stop so it can generate multiline code
text = text.replace('if word == "\\n" or "User"', 'if "User"')

with open("train_ai.py", "w") as f:
    f.write(text)
