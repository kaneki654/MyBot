import numpy as np

# --- Helper functions ---
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

# Mock model that always predicts a distribution slightly favoring token ID 10
class MockLanguageModel:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        
    def forward(self, input_ids):
        # input_ids: (batch, seq_len)
        batch_size, seq_len = input_ids.shape
        # Logits shape: (batch, seq_len, vocab_size)
        logits = np.random.randn(batch_size, seq_len, self.vocab_size)
        # Add a bias to make token 10 slightly more likely so it's not complete noise
        logits[:, :, 10] += 2.0 
        return logits

# --- 1. Temperature & Top-K Sampling ---
def generate_text(model, prompt_ids, max_new_tokens, temperature=1.0, top_k=0):
    """
    Generates text by iteratively predicting the next token.
    prompt_ids: List of integer token IDs (shape: [batch=1, seq_len])
    temperature: > 1.0 = more random, < 1.0 = more deterministic, 1.0 = normal
    top_k: Only consider the top K most likely tokens. 0 = disable.
    """
    # Convert to NumPy array with batch dimension (1, seq_len)
    current_ids = np.array([prompt_ids], dtype=np.int32)
    
    for _ in range(max_new_tokens):
        # 1. Forward Pass
        # We only care about the very last token's predictions
        # logits shape: (1, seq_len, vocab_size)
        logits = model.forward(current_ids)
        next_token_logits = logits[0, -1, :] # Shape: (vocab_size,)
        
        # 2. Apply Temperature
        # Higher temperature squashes the distribution (more random)
        # Lower temperature sharpens it (more greedy)
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature
            
        # 3. Apply Top-K Filtering
        if top_k > 0:
            # Find the value of the k-th largest logit
            # np.partition is faster than full sort, it puts the top k elements at the end
            indices_to_remove = next_token_logits < np.partition(next_token_logits, -top_k)[-top_k]
            # Set the logits of the removed tokens to -infinity so they have 0 probability
            next_token_logits[indices_to_remove] = float('-inf')
            
        # 4. Convert Logits to Probabilities
        probs = softmax(next_token_logits)
        
        # 5. Sample the Next Token
        # Instead of just taking the argmax (greedy), we sample from the distribution
        next_token_id = np.random.choice(model.vocab_size, p=probs)
        
        # Append to the sequence for the next iteration
        current_ids = np.concatenate([current_ids, [[next_token_id]]], axis=1)
        
    return current_ids[0].tolist()

# --- 2. Beam Search (Simplified 1-step logic) ---
# Full beam search maintains multiple parallel "beams" of generated text.
# For brevity, this is the core logic of scoring a beam.
def beam_search_step(model, current_beams, beam_width):
    """
    Expands the current beams and keeps the top 'beam_width' scoring sequences.
    current_beams: List of tuples (sequence_of_ids, cumulative_log_prob)
    """
    all_candidates = []
    
    for seq, score in current_beams:
        # Predict next token for this beam
        input_ids = np.array([seq], dtype=np.int32)
        logits = model.forward(input_ids)
        next_token_logits = logits[0, -1, :]
        
        # Get log probabilities (log_softmax)
        probs = softmax(next_token_logits)
        log_probs = np.log(probs + 1e-10) # Add epsilon to prevent log(0)
        
        # We only need to consider the top 'beam_width' tokens for this specific beam
        # to avoid evaluating all V^B combinations
        top_k_indices = np.argsort(log_probs)[-beam_width:]
        
        for idx in top_k_indices:
            candidate_seq = seq + [idx]
            # Add log probability (multiplying probabilities = adding log probabilities)
            candidate_score = score + log_probs[idx]
            all_candidates.append((candidate_seq, candidate_score))
            
    # Sort all candidates by score (highest log prob is best)
    # and keep only the top 'beam_width' beams
    all_candidates.sort(key=lambda x: x[1], reverse=True)
    return all_candidates[:beam_width]

# --- 3. Quantization (Int8) ---
def quantize_weights(weights):
    """
    Converts float32 weights to int8 to save 75% memory.
    Returns the quantized weights and the scaling factor needed to dequantize.
    """
    # Find the maximum absolute value in the tensor
    max_val = np.max(np.abs(weights))
    
    # Calculate scale so that max_val maps to 127 (max value of int8)
    # 127.0 / max_val
    scale = 127.0 / max_val
    
    # Scale, round to nearest integer, and cast to int8
    quantized = np.round(weights * scale).astype(np.int8)
    
    return quantized, scale

def dequantize_weights(quantized, scale):
    """Converts int8 back to float32 for computation."""
    # float32 = int8 / scale
    return quantized.astype(np.float32) / scale

# --- Run Examples ---
print("Initializing Mock Model...")
vocab_size = 50
model = MockLanguageModel(vocab_size)
prompt = [5, 12] # e.g. "The cat"

print("\n1. Temperature & Top-K Sampling:")
# Greedy (Temp very low) - will likely output token 10 repeatedly
greedy_out = generate_text(model, prompt, max_new_tokens=5, temperature=0.1)
print(f"Greedy Output: {greedy_out}")

# Random (Temp high, Top-K=5)
random_out = generate_text(model, prompt, max_new_tokens=5, temperature=1.5, top_k=5)
print(f"Sampled Output (Top-K=5): {random_out}")

print("\n2. Beam Search Expansion:")
initial_beams = [(prompt, 0.0)] # (sequence, log_prob)
beam_width = 3
new_beams = beam_search_step(model, initial_beams, beam_width)
for i, (seq, score) in enumerate(new_beams):
    print(f"Beam {i+1}: Sequence={seq}, Score={score:.4f}")

print("\n3. Quantization:")
# Mock a massive weight matrix (e.g. 4096 x 4096 float32 = 64MB)
W_float32 = np.random.randn(1000, 1000).astype(np.float32) * 0.1
print(f"Original Float32 size: {W_float32.nbytes / 1024 / 1024:.2f} MB")

W_int8, scale = quantize_weights(W_float32)
print(f"Quantized Int8 size:   {W_int8.nbytes / 1024 / 1024:.2f} MB")

# Verify the error is small
W_restored = dequantize_weights(W_int8, scale)
max_error = np.max(np.abs(W_float32 - W_restored))
print(f"Max quantization error: {max_error:.6f}")
