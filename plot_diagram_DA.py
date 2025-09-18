import matplotlib.pyplot as plt
import numpy as np

# Dictionary with 3 values for each key
values = {'ISLES15(DWI)':[62.2,60.0,59.1],'ISLES15(FLAIR)':[58.2,55.1,53.4],'WMH(FLAIR)':[76.6,70.0,69.6]}

# Extract keys and values
keys = list(values.keys())
val1 = [values[key][0] for key in keys]  # First value for each key
val2 = [values[key][1] for key in keys]  # Second value for each key
val3 = [values[key][2] for key in keys]  # Third value for each key

# Set up the plot
fig, ax = plt.subplots(figsize=(12, 8))

# Set the width of each bar and positions
bar_width = 0.25
x_pos = np.arange(len(keys))

# Create the bars with spacing between groups
bars1 = ax.bar(x_pos - bar_width, val1, bar_width, label='Fine-tune Pre trained Agnostic Path Model', alpha=0.8)
bars2 = ax.bar(x_pos, val2, bar_width, label='Randomly Initialise Agnostic Channel and Path and Finetune', alpha=0.8)
bars3 = ax.bar(x_pos + bar_width, val3, bar_width, label='Randomly Initialise Agnostic Channel and Finetune ', alpha=0.8)

# Customize the plot
ax.set_xlabel('Dataset (Modality in additional channel)', fontsize=25,fontweight = 'bold')
ax.set_ylabel('DICE (%)', fontsize=25,fontweight = 'bold')
ax.set_title('Finetuning with Additional Channels', fontsize=30, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(keys, rotation=45, ha='right', fontsize=20)
ax.legend(fontsize=15)

# Increase y-axis tick label font size
ax.tick_params(axis='y', labelsize=14)

# Set y-axis limit to 90
ax.set_ylim(0, 90)

# Add value labels on top of each bar
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{height:.1f}', ha='center', va='bottom', fontsize=14)

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{height:.1f}', ha='center', va='bottom', fontsize=14)

for bar in bars3:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{height:.1f}', ha='center', va='bottom', fontsize=14)

# Adjust layout and add grid for better readability
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()

# Save the diagram
plt.savefig('grouped_bar_plot.png', dpi=300, bbox_inches='tight')
print("Diagram saved as 'grouped_bar_plot.png'")
plt.show()