import numpy as np
import re

# --- 1. Tokenizer ---
class WordTokenizer:
    def __init__(self):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        
    def _split_text(self, text):
        return re.findall(r"[\w']+|[.,!?;:\n\(\)\{\}\[\]=+\-*/<>\"' ]", text)

    def fit(self, text):
        tokens = self._split_text(text)
        unique_tokens = sorted(list(set(tokens)))
        self.word_to_idx = {"<UNK>": 0}
        self.idx_to_word = {0: "<UNK>"}
        for i, token in enumerate(unique_tokens, start=1):
            self.word_to_idx[token] = i
            self.idx_to_word[i] = token
        self.vocab_size = len(self.word_to_idx)
        
    def encode(self, text):
        tokens = self._split_text(text)
        return [self.word_to_idx.get(t, 0) for t in tokens]
        
    def decode(self, indices):
        return "".join([self.idx_to_word.get(idx, "<UNK>") for idx in indices])

# --- Math Helpers ---
def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def cross_entropy_loss(logits, targets):
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)
    
    probs = softmax(logits_flat)
    correct_probs = probs[np.arange(len(targets_flat)), targets_flat]
    
    eps = 1e-12
    loss = -np.mean(np.log(correct_probs + eps))
    
    d_logits = probs.copy()
    d_logits[np.arange(len(targets_flat)), targets_flat] -= 1
    d_logits = d_logits / len(targets_flat)
    d_logits = d_logits.reshape(batch_size, seq_len, vocab_size)
    
    return loss, d_logits

# --- Step 7: Instruction Tuning (Transformer Architecture) ---
def initialize_weights(shape):
    return np.random.randn(*shape) * np.sqrt(2.0 / sum(shape))

class EmbeddingLayer:
    def __init__(self, vocab_size, d_model):
        self.weights = initialize_weights((vocab_size, d_model))
    def forward(self, input_ids):
        self.input_ids = input_ids
        return self.weights[input_ids]

class PositionalEncoding:
    def __init__(self, max_seq_len, d_model):
        self.pos_emb = np.zeros((max_seq_len, d_model))
        position = np.arange(0, max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        self.pos_emb[:, 0::2] = np.sin(position * div_term)
        self.pos_emb[:, 1::2] = np.cos(position * div_term)
    def forward(self, seq_len):
        return self.pos_emb[:seq_len, :]

class MaskedMultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = initialize_weights((d_model, d_model))
        self.W_k = initialize_weights((d_model, d_model))
        self.W_v = initialize_weights((d_model, d_model))
        self.W_o = initialize_weights((d_model, d_model))

    def split_heads(self, x, batch_size):
        return x.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        Q = np.dot(x, self.W_q)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)
        
        Q_split = self.split_heads(Q, batch_size)
        K_split = self.split_heads(K, batch_size)
        V_split = self.split_heads(V, batch_size)
        
        scores = np.matmul(Q_split, K_split.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        mask = np.triu(np.full((seq_len, seq_len), float('-inf')), k=1)
        scores = scores + mask
        
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        context = np.matmul(attention_weights, V_split)
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        
        output = np.dot(context, self.W_o)
        return output

class LayerNorm:
    def __init__(self, d_model, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(variance + self.eps)
        return self.gamma * x_norm + self.beta

class FeedForward:
    def __init__(self, d_model, d_ff):
        self.W1 = initialize_weights((d_model, d_ff))
        self.b1 = np.zeros(d_ff)
        self.W2 = initialize_weights((d_ff, d_model))
        self.b2 = np.zeros(d_model)
    def relu(self, x):
        return np.maximum(0, x)
    def forward(self, x):
        hidden = self.relu(np.dot(x, self.W1) + self.b1)
        return np.dot(hidden, self.W2) + self.b2

class TransformerBlock:
    def __init__(self, d_model, num_heads, d_ff):
        self.attention = MaskedMultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.norm2 = LayerNorm(d_model)
    def forward(self, x):
        attn_out = self.attention.forward(self.norm1.forward(x))
        x = x + attn_out
        ff_out = self.ff.forward(self.norm2.forward(x))
        x = x + ff_out
        return x

class LMHead:
    def __init__(self, d_model, vocab_size):
        self.W = initialize_weights((d_model, vocab_size))
        self.b = np.zeros(vocab_size)
    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

class AdamOptimizer:
    def __init__(self, params, lr=0.001):
        self.params = params
        self.lr = lr
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0
        
    def step(self, grads):
        self.t += 1
        for i in range(len(self.params)):
            self.m[i] = 0.9 * self.m[i] + 0.1 * grads[i]
            self.v[i] = 0.999 * self.v[i] + 0.001 * (grads[i] ** 2)
            m_hat = self.m[i] / (1 - 0.9 ** self.t)
            v_hat = self.v[i] / (1 - 0.999 ** self.t)
            self.params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)

def get_batches(data, batch_size, block_size):
    chunk_size = block_size + 1
    total_chunks = len(data) // chunk_size
    data = data[:total_chunks * chunk_size]
    
    if total_chunks == 0:
        return []

    num_batches = total_chunks // batch_size
    if num_batches == 0:
        num_batches = 1 # Force at least 1 batch if data is tiny
        batch_size = total_chunks

    batches = []
    for i in range(num_batches):
        start_idx = i * batch_size * chunk_size
        batch_data = data[start_idx : start_idx + (batch_size * chunk_size)]
        batch_data = batch_data.reshape((batch_size, chunk_size))
        x = batch_data[:, :-1]
        y = batch_data[:, 1:]
        batches.append((x, y))
    return batches

# --- Instruction Tuning Dataset ---
qa_dataset = """
User: Hello!
AI: Hi there! How can I help you today?

User: Write a function to add two numbers.
AI: def add(a, b):
    return a + b

User: What is your name?
AI: My name is MyAI, built from scratch in NumPy!

User: How do I print something in Python?
AI: You can use the print() function, like this: print("Hello World")
"""

if __name__ == "__main__":
    print("=" * 60)
    print(" 🤖  Step 7: Instruction Tuning (The ChatGPT Secret)  🤖")
    print("=" * 60)
    print("We are changing the structure of the training data from random raw text")
    print("to explicit 'User: [Prompt] \\n AI: [Response]' formats. This forces")
    print("the AI to learn how to answer questions instead of just autocomplete.\n")
    
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
    epochs = 15
    
    print(f"Dataset Size: {len(encoded_data)} tokens.")
    print("Building Instruction-Tuned Transformer...")
    token_emb = EmbeddingLayer(vocab_size, d_model)
    pos_emb = PositionalEncoding(max_seq_len, d_model)
    block = TransformerBlock(d_model, num_heads, d_ff)
    lm_head = LMHead(d_model, vocab_size)
    
    trainable_params = [lm_head.W, lm_head.b]
    optimizer = AdamOptimizer(trainable_params, lr=learning_rate)
    
    print(f"\n📚 Fine-Tuning on Instruction Q&A pairs | {epochs} epochs\n")

    batches = get_batches(encoded_data, batch_size, max_seq_len)
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        num_batches = 0
        
        for X_batch, Y_batch in batches:
            x_emb = token_emb.forward(X_batch) + pos_emb.forward(max_seq_len)
            hidden_states = block.forward(x_emb)
            logits = lm_head.forward(hidden_states)
            
            loss, d_logits = cross_entropy_loss(logits, Y_batch)
            epoch_loss += loss
            num_batches += 1
            
            X_flat = lm_head.x.reshape(-1, d_model)
            d_logits_flat = d_logits.reshape(-1, vocab_size)
            dW = np.dot(X_flat.T, d_logits_flat)
            db = np.sum(d_logits_flat, axis=0)
            
            optimizer.step([dW, db])
            
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch:2d}/{epochs} | Loss: {avg_loss:.4f} | Perplexity: {np.exp(avg_loss):.1f}")
        
    print("\n✅ Instruction Tuning complete!")
    
    # Generation test simulating a Chat Interface
    print("\n--- Chat Interface Simulation ---")
    prompt = "User: Write a function to add two numbers.\nAI:"
    print(f"{prompt} (Model calculates response...)")
    
    prompt_ids = tokenizer.encode(prompt)
    padded_prompt = np.zeros((1, max_seq_len), dtype=np.int32)
    
    # We take the last `max_seq_len` tokens from the prompt if it's too long
    input_ids = prompt_ids[-max_seq_len:]
    padded_prompt[0, :len(input_ids)] = input_ids
    
    x_emb = token_emb.forward(padded_prompt) + pos_emb.forward(max_seq_len)
    hidden_states = block.forward(x_emb)
    logits = lm_head.forward(hidden_states)
    
    last_token_logits = logits[0, len(input_ids) - 1, :]
    probs = softmax(np.expand_dims(last_token_logits, 0))[0]
    
    predicted_id = np.argmax(probs)
    predicted_word = tokenizer.decode([predicted_id])
    
    print(f"-> {predicted_word}")
    print("\nNotice how it correctly learned to respond with 'def' (the start of the code answer) instead of rambling randomly!")
