import numpy as np
import os
import json
import re

# --- 1. Word-Level Tokenizer ---
class WordTokenizer:
    def __init__(self):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        
    def _split_text(self, text):
        # Improved regex to split by words, punctuation, and newlines
        # This keeps words intact but separates punctuation
        tokens = re.findall(r"[\w']+|[.,!?;:\n\(\)\{\}\[\]=+\-*/<>\"' ]", text)
        return tokens

    def fit(self, text):
        tokens = self._split_text(text)
        # Find unique tokens
        unique_tokens = sorted(list(set(tokens)))
        
        # Always add a special <UNK> token for unknown words
        self.word_to_idx = {"<UNK>": 0}
        self.idx_to_word = {0: "<UNK>"}
        
        for i, token in enumerate(unique_tokens, start=1):
            self.word_to_idx[token] = i
            self.idx_to_word[i] = token
            
        self.vocab_size = len(self.word_to_idx)
        print(f"Tokenizer trained! Found {self.vocab_size} unique words/symbols.")
        
    def encode(self, text):
        tokens = self._split_text(text)
        return [self.word_to_idx.get(t, 0) for t in tokens]
        
    def decode(self, indices):
        return "".join([self.idx_to_word.get(idx, "<UNK>") for idx in indices])


# --- Math Helpers ---
def softmax(x):
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

# --- The LSTM Model ---
class LSTMModel:
    def __init__(self, vocab_size, hidden_size=256, lr=0.002):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.lr = lr

        def W(rows, cols):
            return np.random.randn(rows, cols) * np.sqrt(2.0 / (rows + cols))

        H, V = hidden_size, vocab_size

        self.Wi = W(H, V + H); self.bi = np.zeros((H,))
        self.Wf = W(H, V + H); self.bf = np.ones((H,))
        self.Wo = W(H, V + H); self.bo = np.zeros((H,))
        self.Wg = W(H, V + H); self.bg = np.zeros((H,))
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
        return ['Wi','bi','Wf','bf','Wo','bo','Wg','bg','Wy','by']

    def forward(self, inputs, h_prev, c_prev):
        seq_len = len(inputs)
        H = self.hidden_size

        hs = [h_prev]
        cs = [c_prev]
        cache = []
        logits_list = []

        for x in inputs:
            h = hs[-1]; c = cs[-1]
            xh = np.concatenate([x, h])

            i  = sigmoid(self.Wi @ xh + self.bi)
            f  = sigmoid(self.Wf @ xh + self.bf)
            o  = sigmoid(self.Wo @ xh + self.bo)
            g  = tanh   (self.Wg @ xh + self.bg)

            c_new = f * c + i * g
            h_new = o * tanh(c_new)

            logits = self.Wy @ h_new + self.by

            hs.append(h_new)
            cs.append(c_new)
            cache.append((x, h, c, i, f, o, g, c_new, h_new, xh))
            logits_list.append(logits)

        return logits_list, hs, cs, cache

    def backward(self, logits_list, targets, cache, hs, cs):
        grads = {n: np.zeros_like(getattr(self, n)) for n in self._param_names()}

        dh_next = np.zeros(self.hidden_size)
        dc_next = np.zeros(self.hidden_size)
        total_loss = 0.0

        for t in reversed(range(len(targets))):
            x, h, c, i_g, f_g, o_g, g_g, c_new, h_new, xh = cache[t]

            logits = logits_list[t]
            loss_t, dlogits = cross_entropy_loss(
                logits.reshape(1, -1), np.array([targets[t]])
            )
            total_loss += loss_t
            dlogits = dlogits.flatten()

            grads['Wy'] += np.outer(dlogits, h_new)
            grads['by'] += dlogits
            dh = self.Wy.T @ dlogits + dh_next

            do = tanh(c_new) * dh
            do_raw = do * o_g * (1 - o_g)

            dc = o_g * (1 - tanh(c_new)**2) * dh + dc_next

            df = c * dc
            df_raw = df * f_g * (1 - f_g)

            di = g_g * dc
            di_raw = di * i_g * (1 - i_g)

            dg = i_g * dc
            dg_raw = dg * (1 - g_g**2)

            for raw, W_name, b_name in [
                (di_raw, 'Wi', 'bi'),
                (df_raw, 'Wf', 'bf'),
                (do_raw, 'Wo', 'bo'),
                (dg_raw, 'Wg', 'bg'),
            ]:
                grads[W_name] += np.outer(raw, xh)
                grads[b_name] += raw

            dxh = (self.Wi.T @ di_raw + self.Wf.T @ df_raw +
                   self.Wo.T @ do_raw + self.Wg.T @ dg_raw)

            dh_next = dxh[self.vocab_size:]
            dc_next = f_g * dc

        for k in grads:
            np.clip(grads[k], -5, 5, out=grads[k])

        return grads, total_loss

    def update(self, grads, lr=None):
        lr = lr or self.lr
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for name in self._param_names():
            g = grads[name]
            self.m[name] = b1 * self.m[name] + (1 - b1) * g
            self.v[name] = b2 * self.v[name] + (1 - b2) * g**2
            m_hat = self.m[name] / (1 - b1**self.t)
            v_hat = self.v[name] / (1 - b2**self.t)
            param = getattr(self, name)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def generate(self, tokenizer, seed_text, length=50, temperature=0.7):
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)

        # Encode the seed text
        encoded_seed = tokenizer.encode(seed_text)
        
        # Warm up on seed
        for idx in encoded_seed:
            x = np.zeros(self.vocab_size)
            x[idx] = 1.0
            logits_list, hs, cs, _ = self.forward([x], h, c)
            h = hs[-1]; c = cs[-1]

        result_indices = []
        current_idx = encoded_seed[-1] if encoded_seed else 0

        for _ in range(length):
            x = np.zeros(self.vocab_size)
            x[current_idx] = 1.0

            logits_list, hs, cs, _ = self.forward([x], h, c)
            h = hs[-1]; c = cs[-1]

            # Temperature sampling
            logits = logits_list[0] / temperature
            probs = softmax(logits)
            
            # Sample the next token index
            current_idx = np.random.choice(self.vocab_size, p=probs)
            result_indices.append(current_idx)

        # Decode the generated indices back into a string
        return tokenizer.decode(result_indices)

# --- Trainer ---
def train(model, encoded_data, epochs=5, seq_len=20, batch_size=32):
    n = len(encoded_data)
    print(f"\n📚 Training on {n} tokens | {epochs} epochs | seq_len={seq_len}\n")

    for epoch in range(1, epochs + 1):
        total_loss = 0
        steps = 0
        h = np.zeros(model.hidden_size)
        c = np.zeros(model.hidden_size)

        starts = np.random.randint(0, n - seq_len - 1, size=batch_size)

        for start in starts:
            seq = encoded_data[start: start + seq_len]
            targets = encoded_data[start + 1: start + seq_len + 1]

            inputs = []
            for idx in seq:
                x = np.zeros(model.vocab_size)
                x[idx] = 1.0
                inputs.append(x)

            logits_list, hs, cs, cache = model.forward(inputs, h, c)
            grads, loss = model.backward(logits_list, targets, cache, hs, cs)
            model.update(grads)

            total_loss += loss
            steps += 1

        avg_loss = total_loss / steps
        perplexity = np.exp(avg_loss)

        print(f"Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | Perplexity: {perplexity:.1f}")

    print("\n✅ Training complete!")

# --- Main Execution ---
if __name__ == "__main__":
    print("=" * 55)
    print("   🤖  Upgraded AI (Word-Level Tokens)  🤖")
    print("=" * 55)
    
    # 1. Load Data
    with open("my_training_data.txt", "r", encoding="utf-8") as f:
        # We'll just read the first 10,000 chars of the OS module so training doesn't take forever
        training_text = f.read()[:10000] 
        
    print(f"\nLoaded {len(training_text)} characters of training data.")
    
    # 2. Build and train Tokenizer
    tokenizer = WordTokenizer()
    tokenizer.fit(training_text)
    
    # 3. Encode Data
    encoded_data = tokenizer.encode(training_text)
    print(f"Data compressed to {len(encoded_data)} word-level tokens.\n")
    
    # 4. Initialize Model with larger hidden size
    # Increased hidden size from 128 to 256 for better memory
    model = LSTMModel(vocab_size=tokenizer.vocab_size, hidden_size=256, lr=0.005)
    
    # 5. Train
    # We use a shorter sequence length because words hold more meaning than chars
    train(model, encoded_data, epochs=5, seq_len=15, batch_size=16)
    
    # 6. Test Generation
    print("\n--- Generation Test ---")
    prompt = "def "
    print(f"Prompt: '{prompt}'")
    
    output = model.generate(tokenizer, seed_text=prompt, length=20, temperature=0.8)
    print(f"AI: {prompt}{output}")
