import re

with open("train_ai.py", "r") as f:
    text = f.read()
    
# Fix the regex string escaping that broke during the shell heredoc write
text = text.replace('return re.findall(r"[\\w\']+|[.,!?;:\\n\\(\\)\\{\\}\\[\\]=+\\-*/<>\\"\' ]", text)', 
                    r'return re.findall(r"[\w\']+|[.,!?;:\n\(\)\{\}\[\]=+\-*/<>\"\' ]", text)')
                    
with open("train_ai.py", "w") as f:
    f.write(text)
