with open("train_ai.py", "r") as f:
    text = f.read()

# Fix space issue and generation looping
text = text.replace(
    'return tokenizer.decode(result_indices)',
    'output = tokenizer.decode(result_indices)\n        # Clean up output to only show up to the next User prompt\n        if "User:" in output:\n            output = output.split("User:")[0]\n        return output.strip()'
)

with open("train_ai.py", "w") as f:
    f.write(text)
