import numpy as np

# --- Configuration & Hyperparameters ---
np.random.seed(42)
vocab_size = 260
d_model = 64
max_seq_len = 8
batch_size = 2
num_heads = 4
d_ff = 256
learning_rate = 0.001
epochs = 5
dataset_size = 100 # Very small dataset for demonstration

print("Initializing Training Pipeline...")

# --- 1. Dummy Data Pipeline (Phase 1) ---
# We will create random integer sequences to represent our tokenized text
print("Generating dummy tokenized dataset...")
raw_data_tokens = np.random.randint(0, vocab_size, size=dataset_size)

def get_batches(data, batch_size, block_size):
    """Yields (X, Y) batches where Y is X shifted right by 1."""
    chunk_size = block_size + 1
    total_chunks = len(data) // chunk_size
    data = data[:total_chunks * chunk_size]
    
    num_batches = total_chunks // batch_size
    for i in range(num_batches):
        start_idx = i * batch_size * chunk_size
        batch_data = data[start_idx : start_idx + (batch_size * chunk_size)]
        batch_data = batch_data.reshape((batch_size, chunk_size))
        x = batch_data[:, :-1]
        y = batch_data[:, 1:]
        yield x, y

# --- 2. Model Architecture Components (Phase 2) ---
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
        self.x = x # Save for backward pass
        
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
        self.attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        context = np.matmul(self.attention_weights, V_split)
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
        self.x = x # Save input for backprop
        return np.dot(x, self.W) + self.b

# --- Build the Full Model ---
print("Building Model Architecture...")
token_emb = EmbeddingLayer(vocab_size, d_model)
pos_emb = PositionalEncoding(max_seq_len, d_model)
block = TransformerBlock(d_model, num_heads, d_ff)
lm_head = LMHead(d_model, vocab_size)

# --- 3. Training Loop Components (Phase 3) ---
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

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

# We will only train the LM Head weights for this demonstration to keep it brief.
# A full implementation requires computing the backward pass through the entire Transformer block,
# which involves hundreds of lines of complex matrix calculus (dLoss/dAttention, dLoss/dNorm, etc.).
trainable_params = [lm_head.W, lm_head.b]
optimizer = AdamOptimizer(trainable_params, lr=learning_rate)

print("\nStarting Training Loop...")
for epoch in range(epochs):
    epoch_loss = 0.0
    num_batches = 0
    
    for X_batch, Y_batch in get_batches(raw_data_tokens, batch_size, max_seq_len):
        # --- 1. Forward Pass ---
        # Embeddings
        x_emb = token_emb.forward(X_batch) + pos_emb.forward(max_seq_len)
        
        # Transformer Block
        hidden_states = block.forward(x_emb)
        
        # Output Projection (LM Head)
        logits = lm_head.forward(hidden_states)
        
        # --- 2. Calculate Loss ---
        loss, d_logits = cross_entropy_loss(logits, Y_batch)
        epoch_loss += loss
        num_batches += 1
        
        # --- 3. Backward Pass (LM Head Only) ---
        # dLoss/dW = X^T @ dLoss/dLogits
        X_flat = lm_head.x.reshape(-1, d_model)
        d_logits_flat = d_logits.reshape(-1, vocab_size)
        
        dW = np.dot(X_flat.T, d_logits_flat)
        db = np.sum(d_logits_flat, axis=0)
        
        # --- 4. Optimizer Step ---
        optimizer.step([dW, db])
        
    avg_loss = epoch_loss / num_batches
    print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")

print("\nTraining Complete! The LM Head parameters have been updated.")
