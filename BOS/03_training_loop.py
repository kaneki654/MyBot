import numpy as np

# --- 1. Softmax & Cross-Entropy Loss ---
def softmax(x):
    """Compute softmax values for each set of scores in x."""
    # Subtract max for numerical stability
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def cross_entropy_loss(logits, targets):
    """
    Computes the cross-entropy loss between logits and integer targets.
    logits: (batch_size, seq_len, vocab_size)
    targets: (batch_size, seq_len)
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    # Flatten logits and targets to 2D/1D for easier computation
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)
    
    # Get probabilities
    probs = softmax(logits_flat)
    
    # Get the probability of the true class
    # Use advanced indexing to pick the probability of the correct word
    correct_probs = probs[np.arange(len(targets_flat)), targets_flat]
    
    # Add small epsilon to prevent log(0)
    eps = 1e-12
    loss = -np.mean(np.log(correct_probs + eps))
    
    # Compute the gradient of the loss with respect to the logits (dLoss/dLogits)
    # The gradient of cross entropy with softmax is simply: probs - 1 (for the true class)
    d_logits = probs.copy()
    d_logits[np.arange(len(targets_flat)), targets_flat] -= 1
    d_logits = d_logits / len(targets_flat) # Average over batch
    
    # Reshape back to 3D
    d_logits = d_logits.reshape(batch_size, seq_len, vocab_size)
    
    return loss, d_logits

# --- 2. The Adam Optimizer ---
class AdamOptimizer:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params # List of parameter arrays
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        
        # Initialize moving averages
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        
    def step(self, grads):
        self.t += 1
        
        for i in range(len(self.params)):
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i] ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            self.params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# --- 3. Example Training Loop Setup ---
# Let's mock the parameters and gradients for a simple linear layer
# to demonstrate the loop. In a real scenario, you'd collect parameters
# and gradients recursively from your Transformer blocks.

# Dummy parameters for a single linear layer
vocab_size = 100
d_model = 64
W = np.random.randn(d_model, vocab_size) * 0.01
b = np.zeros(vocab_size)

# Collect parameters
params = [W, b]
optimizer = AdamOptimizer(params, lr=0.01)

# Dummy inputs (from the end of the transformer block)
batch_size = 32
seq_len = 10
X_block_out = np.random.randn(batch_size, seq_len, d_model)

# Dummy targets (next token IDs)
Y_targets = np.random.randint(0, vocab_size, size=(batch_size, seq_len))

epochs = 5
print("Starting Training Loop...")
for epoch in range(epochs):
    # --- Forward Pass ---
    # 1. Compute Logits: X @ W + b
    # X shape: (B, T, d_model), W shape: (d_model, V) -> Logits shape: (B, T, V)
    logits = np.dot(X_block_out, W) + b
    
    # 2. Compute Loss and Gradient of Loss wrt Logits
    loss, d_logits = cross_entropy_loss(logits, Y_targets)
    
    # --- Backward Pass (BPTT) ---
    # 3. Compute Gradients for Parameters (Chain Rule)
    # dLoss/dW = X^T @ dLoss/dLogits
    # Flatten batch and seq_len for matrix multiplication
    X_flat = X_block_out.reshape(-1, d_model) # (B*T, d_model)
    d_logits_flat = d_logits.reshape(-1, vocab_size) # (B*T, V)
    
    # dW shape: (d_model, V)
    dW = np.dot(X_flat.T, d_logits_flat)
    
    # db shape: (V,)
    db = np.sum(d_logits_flat, axis=0)
    
    # Optional: Gradient Clipping (prevents exploding gradients)
    clip_val = 1.0
    dW = np.clip(dW, -clip_val, clip_val)
    db = np.clip(db, -clip_val, clip_val)
    
    grads = [dW, db]
    
    # --- Optimizer Step ---
    optimizer.step(grads)
    
    print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f}")

print("\nPhase 3 complete! The model parameters (W, b) have been updated.")
