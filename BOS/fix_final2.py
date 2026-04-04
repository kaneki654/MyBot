import re
with open("train_ai.py", "r") as f:
    text = f.read()

# Make it generate actual responses based on the prompt!
text = re.sub(
    r'def generate\(self, tokenizer, seed_text, length=30, temperature=0\.7\):',
    r'def generate(self, tokenizer, seed_text, length=40, temperature=0.01):',
    text
)
# Reset hidden size to max
text = re.sub(r'hidden_size=256', 'hidden_size=128', text)
# Increase epochs massively so it forces memorization
text = re.sub(r'epochs=60', 'epochs=120', text)

with open("train_ai.py", "w") as f:
    f.write(text)
