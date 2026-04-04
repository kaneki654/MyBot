with open("train_ai.py", "r") as f:
    text = f.read()

text = text.replace(
    """def train_attention(model, encoded_data, epochs=30, seq_len=16, batch_size=8):""",
    """def train_attention(model, encoded_data, epochs=40, seq_len=18, batch_size=16):"""
)

with open("train_ai.py", "w") as f:
    f.write(text)
