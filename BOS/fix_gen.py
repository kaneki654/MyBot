import re
with open('train_ai.py', 'r') as f:
    text = f.read()

# Make it stop generating when it hits "\n" instead of just hoping for it
text = re.sub(
    r'if word == "\\n" or "User" in word:\n                break', 
    r'if word == "\\n" or "User" in word or i >= length - 1:\n                break', 
    text
)
# Train it for a reasonable amount of epochs
text = re.sub(r'epochs=15', 'epochs=40', text)
# Restore hidden size
text = re.sub(r'hidden_size=128', 'hidden_size=256', text)

with open('train_ai.py', 'w') as f:
    f.write(text)
