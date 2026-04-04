from train_ai_step7 import *

# Let's fix the generation test to ensure we see the correct prediction!
tokenizer = WordTokenizer()
tokenizer.fit(qa_dataset)
encoded_data = np.array(tokenizer.encode(qa_dataset), dtype=np.int32)

vocab_size = tokenizer.vocab_size
d_model = 64
max_seq_len = 12
batch_size = 4
num_heads = 4
d_ff = 256
learning_rate = 0.005

token_emb = EmbeddingLayer(vocab_size, d_model)
pos_emb = PositionalEncoding(max_seq_len, d_model)
block = TransformerBlock(d_model, num_heads, d_ff)
lm_head = LMHead(d_model, vocab_size)

optimizer = AdamOptimizer([lm_head.W, lm_head.b], lr=learning_rate)
batches = get_batches(encoded_data, batch_size, max_seq_len)

for epoch in range(1, 30): # Train a bit longer just to be sure
    for X_batch, Y_batch in batches:
        x_emb = token_emb.forward(X_batch) + pos_emb.forward(max_seq_len)
        hidden_states = block.forward(x_emb)
        logits = lm_head.forward(hidden_states)
        
        loss, d_logits = cross_entropy_loss(logits, Y_batch)
        
        X_flat = lm_head.x.reshape(-1, d_model)
        d_logits_flat = d_logits.reshape(-1, vocab_size)
        dW = np.dot(X_flat.T, d_logits_flat)
        db = np.sum(d_logits_flat, axis=0)
        
        optimizer.step([dW, db])

print("\n--- Fixed Chat Interface Simulation ---")
prompt = "User: Write a function to add two numbers.\nAI:"
print(f"Prompt: {prompt}")

prompt_ids = tokenizer.encode(prompt)
# Ensure we map exactly correctly for the forward pass
input_ids = prompt_ids[-max_seq_len:]
padded_prompt = np.zeros((1, max_seq_len), dtype=np.int32)
padded_prompt[0, :len(input_ids)] = input_ids

x_emb = token_emb.forward(padded_prompt) + pos_emb.forward(max_seq_len)
hidden_states = block.forward(x_emb)
logits = lm_head.forward(hidden_states)

# Grab the logits of the last token
last_token_logits = logits[0, len(input_ids) - 1, :]
probs = softmax(np.expand_dims(last_token_logits, 0))[0]

# Get the top 3 predictions
top_3_indices = np.argsort(probs)[-3:][::-1]
for i, idx in enumerate(top_3_indices):
    word = tokenizer.decode([idx])
    prob = probs[idx] * 100
    if i == 0:
        print(f"-> {word} ({prob:.1f}%)")
    else:
        print(f"   Alternative {i+1}: {word} ({prob:.1f}%)")
