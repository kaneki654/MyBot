import numpy as np

# --- Learning Rate Scheduler ---
class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, base_lr, min_lr=0.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        
        # 1. Linear Warmup
        if self.current_step < self.warmup_steps:
            lr = self.base_lr * (self.current_step / self.warmup_steps)
            
        # 2. Cosine Decay
        else:
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            # Clip progress to [0, 1]
            progress = min(1.0, max(0.0, progress)) 
            # Cosine decay formula
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            lr = self.min_lr + (self.base_lr - self.min_lr) * cosine_decay
            
        # Update the optimizer's learning rate
        self.optimizer.lr = lr
        return lr

# --- Putting it Together ---
print("Setting up LR Scheduler...")
# Mock Optimizer
class MockOptimizer:
    def __init__(self, lr):
        self.lr = lr

opt = MockOptimizer(lr=0.0)
total_training_steps = 100
warmup_steps = 10

scheduler = CosineWarmupScheduler(
    optimizer=opt, 
    warmup_steps=warmup_steps, 
    total_steps=total_training_steps, 
    base_lr=1e-3, 
    min_lr=1e-5
)

lrs = []
for step in range(total_training_steps):
    current_lr = scheduler.step()
    lrs.append(current_lr)

print(f"Initial LR: {lrs[0]:.6f}")
print(f"Peak LR (Step {warmup_steps}): {lrs[warmup_steps-1]:.6f}")
print(f"Final LR (Step {total_training_steps}): {lrs[-1]:.6f}")
