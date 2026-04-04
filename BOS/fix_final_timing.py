with open("train_ai.py", "r") as f:
    text = f.read()

text = text.replace("epochs=250", "epochs=100")

with open("train_ai.py", "w") as f:
    f.write(text)
