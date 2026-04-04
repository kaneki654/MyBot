with open("train_ai.py", "r") as f:
    text = f.read()

# Fix the tokenizer string that still splits parenthesis sometimes
text = text.replace(
    r'return re.findall(r"[\w\']+|[.,!?;:\n\(\)\{\}\[\]=+\-*/<>\"\' ]", text)',
    r'return re.findall(r"[\w]+|[^\s\w]", text)'
)

# And boost hidden size and epoch just one more time.
text = text.replace("hidden_size=128", "hidden_size=256")
text = text.replace("epochs=120", "epochs=250")

with open("train_ai.py", "w") as f:
    f.write(text)
