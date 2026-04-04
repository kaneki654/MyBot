import re
with open('train_ai.py', 'r') as f:
    text = f.read()

# Reduce epochs to ensure it finishes within the timeout window
text = re.sub(r'epochs=80', 'epochs=40', text)

with open('train_ai.py', 'w') as f:
    f.write(text)
