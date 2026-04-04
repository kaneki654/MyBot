import numpy as np
import os
import json

# --- 3. Save & Load Checkpoints ---
def save_checkpoint(filepath, model_params, optimizer_state, epoch, loss):
    """
    Saves the state of the model and optimizer to a file.
    model_params: Dictionary of NumPy arrays containing weights
    optimizer_state: Dictionary of optimizer variables (e.g. moving averages)
    """
    print(f"Saving checkpoint to {filepath}...")
    checkpoint = {
        'epoch': epoch,
        'loss': loss,
        'model_state': {},
        'optimizer_state': {}
    }
    
    # Iterate and copy data out to standard Python lists for JSON compatibility
    # If saving to .npz, we would just dump the arrays directly, but we will serialize to a custom format
    
    # We will use np.savez for efficient binary storage of the arrays
    np_arrays = {}
    
    # Add model parameters
    for name, param in model_params.items():
        np_arrays[f'model_{name}'] = param
        
    # Add optimizer state
    for name, state in optimizer_state.items():
        if isinstance(state, list):
            for i, p in enumerate(state):
                 np_arrays[f'optim_{name}_{i}'] = p
        else:
            np_arrays[f'optim_{name}'] = state
            
    # Save the arrays
    np.savez(filepath, **np_arrays)
    
    # Save the metadata (epoch, loss) in a sidecar file or rely on naming
    meta_filepath = filepath.replace('.npz', '_meta.json')
    with open(meta_filepath, 'w') as f:
        json.dump({'epoch': epoch, 'loss': loss}, f)
        
    print(f"Checkpoint saved.")

def load_checkpoint(filepath, model_params, optimizer_state):
    """
    Restores the model and optimizer from a saved checkpoint.
    Updates the provided dictionaries in-place.
    """
    print(f"Loading checkpoint from {filepath}...")
    
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return 0, float('inf')
        
    # Load the arrays
    checkpoint_data = np.load(filepath)
    
    # Restore model parameters
    for name in model_params.keys():
        key = f'model_{name}'
        if key in checkpoint_data:
            model_params[name][:] = checkpoint_data[key] # In-place update
            
    # Restore optimizer state
    for name in optimizer_state.keys():
        if isinstance(optimizer_state[name], list):
             for i in range(len(optimizer_state[name])):
                  key = f'optim_{name}_{i}'
                  if key in checkpoint_data:
                       optimizer_state[name][i][:] = checkpoint_data[key]
        else:
             key = f'optim_{name}'
             if key in checkpoint_data:
                 optimizer_state[name] = float(checkpoint_data[key])
                 
    # Load metadata
    meta_filepath = filepath.replace('.npz', '_meta.json')
    epoch, loss = 0, float('inf')
    if os.path.exists(meta_filepath):
        with open(meta_filepath, 'r') as f:
            meta = json.loads(f.read())
            epoch = meta.get('epoch', 0)
            loss = meta.get('loss', float('inf'))
            
    print(f"Successfully loaded checkpoint from epoch {epoch} (loss: {loss:.4f})")
    return epoch, loss

# Mock Data
vocab_size = 100
d_model = 64
model_weights = {
    'W': np.random.randn(d_model, vocab_size) * 0.01,
    'b': np.zeros(vocab_size)
}

# We'll mock the optimizer state (Adam)
adam_m = [np.zeros_like(p) for p in model_weights.values()]
adam_v = [np.zeros_like(p) for p in model_weights.values()]
optim_state = {
    'm': adam_m,
    'v': adam_v,
    't': 100 # Current step
}

print("\nTesting Checkpoint Save/Load...")
checkpoint_file = 'model_checkpoint.npz'
save_checkpoint(checkpoint_file, model_weights, optim_state, epoch=5, loss=2.345)

# Clear out the state to ensure we actually load it
model_weights['W'][:] = 0
model_weights['b'][:] = 0

# Reload
loaded_epoch, loaded_loss = load_checkpoint(checkpoint_file, model_weights, optim_state)
print(f"W shape restored: {model_weights['W'].shape}, sum: {np.sum(model_weights['W']):.4f}")
