import re
with open("train_ai.py", "r") as f:
    text = f.read()
# Reduce epochs so it doesn't timeout the 2-minute bash window
text = re.sub(r'epochs = 150', 'epochs = 30', text)
with open("train_ai.py", "w") as f:
    f.write(text)
