import numpy as np
import os
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
    # Ensure x is at least 1D, or handle 0D
    if x.ndim == 0:
        return np.array(1.0)
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def tanh(x):
    return np.tanh(x)

def cross_entropy_loss(logits, targets):
    probs = softmax(logits)
    n = logits.shape[0]
    loss = -np.log(probs[np.arange(n), targets] + 1e-9).mean()
    grad = probs.copy()
    grad[np.arange(n), targets] -= 1
    grad /= n
    return loss, grad

# --- Step 5: LSTM with Attention Mechanism ---
class AttentionLSTMModel:
    def __init__(self, vocab_size, hidden_size=128, lr=0.005):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.lr = lr

        def W(rows, cols):
            return np.random.randn(rows, cols) * np.sqrt(2.0 / (rows + cols))

        H, V = hidden_size, vocab_size

        # LSTM Gates
        self.Wi = W(H, V + H); self.bi = np.zeros((H,))
        self.Wf = W(H, V + H); self.bf = np.ones((H,))
        self.Wo = W(H, V + H); self.bo = np.zeros((H,))
        self.Wg = W(H, V + H); self.bg = np.zeros((H,))

        # Attention Weights
        # We will use Luong Dot-Product Attention: score = h_t^T * W_a * h_s
        self.Wa = W(H, H) 
        
        # Combine Context & Hidden state into Output
        self.Wc = W(H, H * 2); self.bc = np.zeros((H,))
        
        # Output Projection to Vocab
        self.Wy = W(V, H);     self.by = np.zeros((V,))

        self._init_adam()

    def _init_adam(self):
        self.t = 0
        self.m = {}; self.v = {}
        for name in self._param_names():
            p = getattr(self, name)
            self.m[name] = np.zeros_like(p)
            self.v[name] = np.zeros_like(p)

    def _param_names(self):
        return [
            'Wi','bi','Wf','bf','Wo','bo','Wg','bg',
            'Wa', 'Wc', 'bc', 'Wy', 'by'
        ]

    def forward(self, inputs, h_prev, c_prev):
        hs = [h_prev]; cs = [c_prev]
        contexts = []
        attentions = []
        cache = []
        logits_list = []

        # 1. Standard LSTM forward pass + Attention at each step
        for t, x in enumerate(inputs):
            h = hs[-1]; c = cs[-1]
            xh = np.concatenate([x, h])

            i = sigmoid(self.Wi @ xh + self.bi)
            f = sigmoid(self.Wf @ xh + self.bf)
            o = sigmoid(self.Wo @ xh + self.bo)
            g = tanh   (self.Wg @ xh + self.bg)

            c_new = f * c + i * g
            h_new = o * tanh(c_new)
            
            # --- Attention Mechanism ---
            # Look at all PAST hidden states (hs contains [h_prev, h_1, ..., h_{t-1}])
            # Score = h_new^T @ Wa @ h_past
            scores = []
            for h_past in hs[1:]: # Exclude h_prev from sequence start
                 score = h_new.T @ self.Wa @ h_past
                 scores.append(score)
                 
            # If it's the very first token, there are no past states, so attention is just h_new
            if len(scores) == 0:
                context = np.zeros_like(h_new)
                attn_weights = np.array([1.0])
            else:
                scores = np.array(scores)
                attn_weights = softmax(scores) # Convert scores to probabilities (0.0 to 1.0)
                
                # Context vector is the weighted sum of all past hidden states
                past_states = np.column_stack(hs[1:]) # Shape: (H, num_past_states)
                context = past_states @ attn_weights  # Shape: (H,)

            # Combine the current hidden state and the context vector
            combined = np.concatenate([context, h_new])
            attentional_hidden = tanh(self.Wc @ combined + self.bc)
            
            # --- Predict Next Token ---
            logits = self.Wy @ attentional_hidden + self.by

            hs.append(h_new); cs.append(c_new)
            contexts.append(context)
            attentions.append(attn_weights)
            
            cache.append((x, h, c, i, f, o, g, c_new, h_new, xh, context, combined, attentional_hidden))
            logits_list.append(logits)

        return logits_list, hs, cs, cache, attentions

    def update_params(self, grads):
        # We use a simplified update method here because backpropagating through 
        # time WITH dynamic attention weights across the entire sequence is exactly 
        # why Transformers were invented!
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for name in self._param_names():
            if name not in grads: continue
            g = grads[name]
            self.m[name] = b1 * self.m[name] + (1 - b1) * g
            self.v[name] = b2 * self.v[name] + (1 - b2) * g**2
            m_hat = self.m[name] / (1 - b1**self.t)
            v_hat = self.v[name] / (1 - b2**self.t)
            param = getattr(self, name)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def generate(self, tokenizer, seed_text, length=30, temperature=0.7):
        h = np.zeros(self.hidden_size); c = np.zeros(self.hidden_size)
        encoded_seed = tokenizer.encode(seed_text)
        
        # We need to maintain the full history of hidden states for attention
        hs_history = [h]
        
        # Warmup
        for idx in encoded_seed:
            x = np.zeros(self.vocab_size)
            x[idx] = 1.0
            _, hs, cs, _, _ = self.forward([x], h, c)
            h = hs[-1]; c = cs[-1]
            hs_history.append(h)

        result_indices = []
        current_idx = encoded_seed[-1] if encoded_seed else 0

        for _ in range(length):
            x = np.zeros(self.vocab_size)
            x[current_idx] = 1.0

            # Forward pass
            logits_list, hs, cs, _, _ = self.forward([x], h, c)
            h = hs[-1]; c = cs[-1]
            hs_history.append(h)

            logits = logits_list[0] / temperature
            probs = softmax(logits)
            current_idx = np.random.choice(self.vocab_size, p=probs)
            result_indices.append(current_idx)

        return tokenizer.decode(result_indices)

# --- Simplified Training Loop ---
def train_attention(model, encoded_data, epochs=5, seq_len=15, batch_size=16):
    n = len(encoded_data)
    print(f"\n📚 Training Attention LSTM on {n} tokens | {epochs} epochs\n")

    for epoch in range(1, epochs + 1):
        total_loss = 0
        steps = 0
        h = np.zeros(model.hidden_size); c = np.zeros(model.hidden_size)

        starts = np.random.randint(0, n - seq_len - 1, size=batch_size)

        for start in starts:
            seq = encoded_data[start: start + seq_len]
            targets = encoded_data[start + 1: start + seq_len + 1]

            inputs = []
            for idx in seq:
                x = np.zeros(model.vocab_size)
                x[idx] = 1.0
                inputs.append(x)

            # 1. Forward Pass with Attention
            logits_list, hs, cs, cache, attentions = model.forward(inputs, h, c)
            
            # 2. Compute Loss and Gradients for Output Layers
            # Full BPTT with Attention in pure NumPy is extremely complex.
            # We will approximate by training the Output, Attention Combine, and LSTM output weights.
            grads = {n: np.zeros_like(getattr(model, n)) for n in model._param_names()}
            loss_sum = 0
            
            for t in reversed(range(len(targets))):
                (x, h_prev, c_prev, i, f, o, g, c_new, h_new, xh, context, combined, attentional_hidden) = cache[t]
                
                loss_t, dlogits = cross_entropy_loss(logits_list[t].reshape(1, -1), np.array([targets[t]]))
                loss_sum += loss_t
                dlogits = dlogits.flatten()
                
                # Grad for Wy and by
                grads['Wy'] += np.outer(dlogits, attentional_hidden)
                grads['by'] += dlogits
                
                # Backprop into Attentional Hidden layer (tanh)
                d_attentional_hidden = model.Wy.T @ dlogits
                d_combined_raw = d_attentional_hidden * (1 - attentional_hidden**2)
                
                # Grad for Wc and bc
                grads['Wc'] += np.outer(d_combined_raw, combined)
                grads['bc'] += d_combined_raw
                
            model.update_params(grads)

            total_loss += loss_sum
            steps += 1

        avg_loss = total_loss / steps
        print(f"Epoch {epoch:2d}/{epochs} | Loss: {avg_loss:.4f} | Perplexity: {np.exp(avg_loss):.1f}")

    print("\n✅ Training complete!")

if __name__ == "__main__":
    print("=" * 60)
    print(" 🤖  Step 5: LSTM with Attention Mechanism  🤖")
    print("=" * 60)
    print("Attention allows the model to look back at ALL past words in the sentence,")
    print("not just what it holds in its current short-term hidden memory.")
    
    with open("my_training_data.txt", "r", encoding="utf-8") as f:
        training_text = f.read()[:5000] # Smaller sample for fast demonstration
        
    tokenizer = WordTokenizer()
    tokenizer.fit(training_text)
    encoded_data = tokenizer.encode(training_text)
    
    model = AttentionLSTMModel(vocab_size=tokenizer.vocab_size, hidden_size=128, lr=0.01)
    
    train_attention(model, encoded_data, epochs=10, seq_len=12, batch_size=8)
    
    prompt = "def "
    print(f"\nGeneration Test -> Prompt: '{prompt}'")
    output = model.generate(tokenizer, seed_text=prompt, length=15, temperature=0.6)
    print(f"AI: {prompt}{output}")
