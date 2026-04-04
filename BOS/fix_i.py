import re
with open('train_ai.py', 'r') as f:
    text = f.read()

# Fix the variable name in the loop
text = re.sub(
    r'for _ in range\(length\):', 
    r'for i in range(length):', 
    text
)

with open('train_ai.py', 'w') as f:
    f.write(text)
