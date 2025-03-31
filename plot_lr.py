import numpy as np
import matplotlib.pyplot as plt

def lr_lambda(current_epoch):
    # Parameters for scheduler
    warmup_epochs = 50
    maintain_epochs = 250
    decay_schedule = [(250, 1.0), (350, 0.7), (450, 0.4), (550, 0.2)]
    min_lr_factor = 0.1
    
    # Warm-up phase
    if current_epoch < warmup_epochs:
        return (float(current_epoch) + 1) / float(warmup_epochs)
    
    # Maintain phase
    if current_epoch <= maintain_epochs:
        return 1.0
    
    # Step decay phase
    for epoch_threshold, lr_factor in decay_schedule:
        if current_epoch <= epoch_threshold:
            return lr_factor
    
    # Final decay phase
    final_decay = max(
        min_lr_factor * decay_schedule[-1][1],
        decay_schedule[-1][1] - ((current_epoch - decay_schedule[-1][0]) / 100) * 0.1
    )
    return final_decay

# Generate data points
epochs = np.arange(0, 600)
lr_values = [lr_lambda(epoch) for epoch in epochs]

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(epochs, lr_values, 'b-', linewidth=2)
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate Factor')
plt.title('Learning Rate Schedule Over 600 Epochs')

# Add annotations for different phases
plt.axvline(x=50, color='r', linestyle='--', alpha=0.3)
plt.axvline(x=250, color='r', linestyle='--', alpha=0.3)
plt.axvline(x=350, color='r', linestyle='--', alpha=0.3)
plt.axvline(x=450, color='r', linestyle='--', alpha=0.3)
plt.axvline(x=550, color='r', linestyle='--', alpha=0.3)

plt.text(25, 0.5, 'Warmup', rotation=90)
plt.text(150, 1.1, 'Maintain Max LR')
plt.text(300, 0.8, 'Step\nDecay')
plt.text(560, 0.3, 'Final\nDecay')

# Save the plot
plt.savefig('lr_schedule.png')
print('Learning rate schedule visualization saved as lr_schedule.png') 