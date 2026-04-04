import numpy as np
import os
import re

# --- 1. Tokenizer (from Step 1) ---
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

# --- Step 3: Stacked (2-Layer) LSTM Model ---
class StackedLSTMModel:
    def __init__(self, vocab_size, hidden_size=256, max_lr=0.005):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.lr = max_lr # We will update this dynamically

        def W(rows, cols):
            return np.random.randn(rows, cols) * np.sqrt(2.0 / (rows + cols))

        H, V = hidden_size, vocab_size

        # Layer 1
        self.W1i = W(H, V + H); self.b1i = np.zeros((H,))
        self.W1f = W(H, V + H); self.b1f = np.ones((H,))
        self.W1o = W(H, V + H); self.b1o = np.zeros((H,))
        self.W1g = W(H, V + H); self.b1g = np.zeros((H,))

        # Layer 2 (Input is H from Layer 1, concatenated with H from Layer 2 previous)
        self.W2i = W(H, H + H); self.b2i = np.zeros((H,))
        self.W2f = W(H, H + H); self.b2f = np.ones((H,))
        self.W2o = W(H, H + H); self.b2o = np.zeros((H,))
        self.W2g = W(H, H + H); self.b2g = np.zeros((H,))

        # Output projection
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
            'W1i','b1i','W1f','b1f','W1o','b1o','W1g','b1g',
            'W2i','b2i','W2f','b2f','W2o','b2o','W2g','b2g',
            'Wy','by'
        ]

    def forward(self, inputs, h1_prev, c1_prev, h2_prev, c2_prev):
        hs1 = [h1_prev]; cs1 = [c1_prev]
        hs2 = [h2_prev]; cs2 = [c2_prev]
        cache = []
        logits_list = []

        for x in inputs:
            # --- Layer 1 ---
            h1 = hs1[-1]; c1 = cs1[-1]
            xh1 = np.concatenate([x, h1])
            i1 = sigmoid(self.W1i @ xh1 + self.b1i)
            f1 = sigmoid(self.W1f @ xh1 + self.b1f)
            o1 = sigmoid(self.W1o @ xh1 + self.b1o)
            g1 = tanh   (self.W1g @ xh1 + self.b1g)
            c1_new = f1 * c1 + i1 * g1
            h1_new = o1 * tanh(c1_new)

            # --- Layer 2 ---
            h2 = hs2[-1]; c2 = cs2[-1]
            xh2 = np.concatenate([h1_new, h2]) # Input from L1 + Prev state from L2
            i2 = sigmoid(self.W2i @ xh2 + self.b2i)
            f2 = sigmoid(self.W2f @ xh2 + self.b2f)
            o2 = sigmoid(self.W2o @ xh2 + self.b2o)
            g2 = tanh   (self.W2g @ xh2 + self.b2g)
            c2_new = f2 * c2 + i2 * g2
            h2_new = o2 * tanh(c2_new)

            # --- Output Layer ---
            logits = self.Wy @ h2_new + self.by

            hs1.append(h1_new); cs1.append(c1_new)
            hs2.append(h2_new); cs2.append(c2_new)
            
            cache.append((x, h1, c1, i1, f1, o1, g1, c1_new, h1_new, xh1,
                             h2, c2, i2, f2, o2, g2, c2_new, h2_new, xh2))
            logits_list.append(logits)

        return logits_list, hs1, cs1, hs2, cs2, cache

    def backward(self, logits_list, targets, cache):
        grads = {n: np.zeros_like(getattr(self, n)) for n in self._param_names()}

        dh1_next = np.zeros(self.hidden_size)
        dc1_next = np.zeros(self.hidden_size)
        dh2_next = np.zeros(self.hidden_size)
        dc2_next = np.zeros(self.hidden_size)
        total_loss = 0.0

        for t in reversed(range(len(targets))):
            (x, h1, c1, i1, f1, o1, g1, c1_new, h1_new, xh1,
                h2, c2, i2, f2, o2, g2, c2_new, h2_new, xh2) = cache[t]

            logits = logits_list[t]
            loss_t, dlogits = cross_entropy_loss(logits.reshape(1, -1), np.array([targets[t]]))
            total_loss += loss_t
            dlogits = dlogits.flatten()

            # Output gradients
            grads['Wy'] += np.outer(dlogits, h2_new)
            grads['by'] += dlogits
            dh2 = self.Wy.T @ dlogits + dh2_next

            # --- Layer 2 Backprop ---
            do2 = tanh(c2_new) * dh2
            do2_raw = do2 * o2 * (1 - o2)
            dc2 = o2 * (1 - tanh(c2_new)**2) * dh2 + dc2_next
            df2 = c2 * dc2
            df2_raw = df2 * f2 * (1 - f2)
            di2 = g2 * dc2
            di2_raw = di2 * i2 * (1 - i2)
            dg2 = i2 * dc2
            dg2_raw = dg2 * (1 - g2**2)

            for raw, W_name, b_name in [(di2_raw, 'W2i', 'b2i'), (df2_raw, 'W2f', 'b2f'),
                                        (do2_raw, 'W2o', 'b2o'), (dg2_raw, 'W2g', 'b2g')]:
                grads[W_name] += np.outer(raw, xh2)
                grads[b_name] += raw

            dxh2 = (self.W2i.T @ di2_raw + self.W2f.T @ df2_raw +
                    self.W2o.T @ do2_raw + self.W2g.T @ dg2_raw)

            dh1 = dxh2[:self.hidden_size] + dh1_next  # Gradient to pass down to Layer 1
            dh2_next = dxh2[self.hidden_size:]        # Gradient to pass back in time for Layer 2
            dc2_next = f2 * dc2

            # --- Layer 1 Backprop ---
            do1 = tanh(c1_new) * dh1
            do1_raw = do1 * o1 * (1 - o1)
            dc1 = o1 * (1 - tanh(c1_new)**2) * dh1 + dc1_next
            df1 = c1 * dc1
            df1_raw = df1 * f1 * (1 - f1)
            di1 = g1 * dc1
            di1_raw = di1 * i1 * (1 - i1)
            dg1 = i1 * dc1
            dg1_raw = dg1 * (1 - g1**2)

            for raw, W_name, b_name in [(di1_raw, 'W1i', 'b1i'), (df1_raw, 'W1f', 'b1f'),
                                        (do1_raw, 'W1o', 'b1o'), (dg1_raw, 'W1g', 'b1g')]:
                grads[W_name] += np.outer(raw, xh1)
                grads[b_name] += raw

            dxh1 = (self.W1i.T @ di1_raw + self.W1f.T @ df1_raw +
                    self.W1o.T @ do1_raw + self.W1g.T @ dg1_raw)

            dh1_next = dxh1[self.vocab_size:]
            dc1_next = f1 * dc1

        for k in grads:
            np.clip(grads[k], -5, 5, out=grads[k])

        return grads, total_loss

    def update(self):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for name in self._param_names():
            g = self.grads[name]
            self.m[name] = b1 * self.m[name] + (1 - b1) * g
            self.v[name] = b2 * self.v[name] + (1 - b2) * g**2
            m_hat = self.m[name] / (1 - b1**self.t)
            v_hat = self.v[name] / (1 - b2**self.t)
            param = getattr(self, name)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def generate(self, tokenizer, seed_text, length=30, temperature=0.7):
        h1 = np.zeros(self.hidden_size); c1 = np.zeros(self.hidden_size)
        h2 = np.zeros(self.hidden_size); c2 = np.zeros(self.hidden_size)

        encoded_seed = tokenizer.encode(seed_text)
        
        for idx in encoded_seed:
            x = np.zeros(self.vocab_size)
            x[idx] = 1.0
            logits_list, hs1, cs1, hs2, cs2, _ = self.forward([x], h1, c1, h2, c2)
            h1 = hs1[-1]; c1 = cs1[-1]
            h2 = hs2[-1]; c2 = cs2[-1]

        result_indices = []
        current_idx = encoded_seed[-1] if encoded_seed else 0

        for _ in range(length):
            x = np.zeros(self.vocab_size)
            x[current_idx] = 1.0

            logits_list, hs1, cs1, hs2, cs2, _ = self.forward([x], h1, c1, h2, c2)
            h1 = hs1[-1]; c1 = cs1[-1]
            h2 = hs2[-1]; c2 = cs2[-1]

            logits = logits_list[0] / temperature
            probs = softmax(logits)
            current_idx = np.random.choice(self.vocab_size, p=probs)
            result_indices.append(current_idx)

        return tokenizer.decode(result_indices)

# --- Step 4: Training with Learning Rate Scheduler ---
def train(model, encoded_data, epochs=10, seq_len=15, batch_size=16, max_lr=0.005):
    n = len(encoded_data)
    print(f"\n📚 Training 2-Layer LSTM on {n} tokens | {epochs} epochs\n")

    for epoch in range(1, epochs + 1):
        # Cosine Annealing Learning Rate Schedule
        # Starts at max_lr and smoothly drops to 0 by the last epoch
        current_lr = max_lr * 0.5 * (1 + np.cos(np.pi * (epoch - 1) / epochs))
        model.lr = current_lr
        
        total_loss = 0
        steps = 0
        h1 = np.zeros(model.hidden_size); c1 = np.zeros(model.hidden_size)
        h2 = np.zeros(model.hidden_size); c2 = np.zeros(model.hidden_size)

        starts = np.random.randint(0, n - seq_len - 1, size=batch_size)

        for start in starts:
            seq = encoded_data[start: start + seq_len]
            targets = encoded_data[start + 1: start + seq_len + 1]

            inputs = []
            for idx in seq:
                x = np.zeros(model.vocab_size)
                x[idx] = 1.0
                inputs.append(x)

            logits_list, hs1, cs1, hs2, cs2, cache = model.forward(inputs, h1, c1, h2, c2)
            grads, loss = model.backward(logits_list, targets, cache)
            model.grads = grads
            model.update()

            total_loss += loss
            steps += 1

        avg_loss = total_loss / steps
        print(f"Epoch {epoch:2d}/{epochs} | LR: {current_lr:.5f} | Loss: {avg_loss:.4f} | Perplexity: {np.exp(avg_loss):.1f}")

    print("\n✅ Training complete!")

if __name__ == "__main__":
    print("=" * 60)
    print(" 🤖  Step 3 & 4: 2-Layer Stacked LSTM + LR Scheduler  🤖")
    print("=" * 60)
    
    with open("my_training_data.txt", "r", encoding="utf-8") as f:
        training_text = f.read()[:5000] # Smaller sample for fast demonstration
        
    tokenizer = WordTokenizer()
    tokenizer.fit(training_text)
    encoded_data = tokenizer.encode(training_text)
    
    model = StackedLSTMModel(vocab_size=tokenizer.vocab_size, hidden_size=256)
    
    # Train for 8 epochs to see the learning rate decay in action
    train(model, encoded_data, epochs=8, seq_len=15, batch_size=8, max_lr=0.01)
    
    prompt = "def "
    print(f"\nGeneration Test -> Prompt: '{prompt}'")
    output = model.generate(tokenizer, seed_text=prompt, length=15, temperature=0.5)
    print(f"AI: {prompt}{output}")
