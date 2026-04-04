with open("train_ai.py", "r") as f:
    text = f.read()

text = text.replace(
    """def train_attention(model, encoded_data, epochs=80, seq_len=30, batch_size=4):""",
    """def train_attention(model, encoded_data, epochs=80, seq_len=18, batch_size=8):"""
)

with open("train_ai.py", "w") as f:
    f.write(text)
