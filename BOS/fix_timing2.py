import re
with open('train_ai.py', 'r') as f:
    text = f.read()

# Reduce epochs even more and lower hidden size so it trains fast
text = re.sub(r'epochs=40', 'epochs=15', text)
text = re.sub(r'hidden_size=512', 'hidden_size=128', text)

with open('train_ai.py', 'w') as f:
    f.write(text)
