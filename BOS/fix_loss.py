import re
with open("train_ai.py", "r") as f:
    text = f.read()
    
# We want to increase the epochs and hidden size so it perfectly memorizes the responses.
# I will use a high learning rate but lower the epochs so it trains quickly in the shell environment.
text = re.sub(r'epochs=40', 'epochs=60', text)

with open("train_ai.py", "w") as f:
    f.write(text)
