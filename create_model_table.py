import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Create figure and axis - adjusted for header visibility
fig, ax = plt.subplots(figsize=(16, 12))
ax.set_xlim(0, 12)
ax.set_ylim(0, 12)
ax.axis('off')

# Define colors
header_color = '#4a90e2'
highlight_color = '#f5f5dc'
standard_color = '#ffffff'
text_color = '#333333'
checkmark_color = '#28a745'
cross_color = '#dc3545'

# Table data
models = ['Standard', 'Shuffled', 'Single', 'Agn_Channel', 'Agn_Path']
mod_specific = [True, False, False, True, True]
mod_agnostic = [False, True, True, True, True]

# Title removed - clean table only

# Table dimensions - bigger header to fit column headings
table_x = 1.5
table_y = 3.5
cell_width = 3.0
cell_height = 1.4
header_height = 1.8

# Draw header row
header_rect = patches.Rectangle((table_x, table_y + len(models) * cell_height), 
                               cell_width * 3, header_height, 
                               linewidth=2, edgecolor='white', 
                               facecolor=header_color)
ax.add_patch(header_rect)

# Header text - Clear column headings with proper sizing
headers = ['Model', 'Modality\nSpecific', 'Modality\nAgnostic']
for i, header in enumerate(headers):
    plt.text(table_x + (i + 0.5) * cell_width, 
             table_y + len(models) * cell_height + header_height/2, 
             header, fontsize=22, fontweight='black', 
             ha='center', va='center', color='white', linespacing=1.2)

# Draw table rows
for i, (model, spec, agno) in enumerate(zip(models, mod_specific, mod_agnostic)):
    y_pos = table_y + (len(models) - 1 - i) * cell_height
    
    # Highlight special rows
    row_color = highlight_color if 'Agn_' in model else standard_color
    
    # Draw row cells
    for j in range(3):
        cell_rect = patches.Rectangle((table_x + j * cell_width, y_pos), 
                                    cell_width, cell_height, 
                                    linewidth=1, edgecolor='#cccccc', 
                                    facecolor=row_color)
        ax.add_patch(cell_rect)
    
    # Model name - ENORMOUS to completely fill the cell
    plt.text(table_x + 0.5 * cell_width, y_pos + cell_height/2, 
             model, fontsize=24, fontweight='black',
             ha='center', va='center', color=text_color)
    
    # Modality Specific - GIGANTIC symbols that COMPLETELY FILL the cell
    symbol = '✓' if spec else '✗'
    color = checkmark_color if spec else cross_color
    plt.text(table_x + 1.5 * cell_width, y_pos + cell_height/2, 
             symbol, fontsize=50, fontweight='black',
             ha='center', va='center', color=color)
    
    # Modality Agnostic - GIGANTIC symbols that COMPLETELY FILL the cell
    symbol = '✓' if agno else '✗'
    color = checkmark_color if agno else cross_color
    plt.text(table_x + 2.5 * cell_width, y_pos + cell_height/2, 
             symbol, fontsize=50, fontweight='black',
             ha='center', va='center', color=color)

# Add legend - POSTER SIZE
legend_y = 2.5
plt.text(6, legend_y, 'Legend: ✓ Supported  |  ✗ Not Supported', 
         fontsize=18, fontweight='black', ha='center', va='center', color='#222222')
plt.text(6, legend_y - 0.5, 'Highlighted rows indicate models with both capabilities', 
         fontsize=16, fontweight='bold', ha='center', va='center', color='#444444', style='italic')

# Add some styling elements
# Top border
top_line = patches.Rectangle((table_x - 0.1, table_y + len(models) * cell_height + header_height), 
                           cell_width * 3 + 0.2, 0.05, 
                           facecolor=header_color, edgecolor='none')
ax.add_patch(top_line)

# Bottom border
bottom_line = patches.Rectangle((table_x - 0.1, table_y - 0.05), 
                              cell_width * 3 + 0.2, 0.05, 
                              facecolor=header_color, edgecolor='none')
ax.add_patch(bottom_line)

plt.tight_layout()

# Save as high-quality image
plt.savefig('model_comparison_table.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('model_comparison_table.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none')

print("✅ Table saved as:")
print("   📊 model_comparison_table.png (high-resolution image)")
print("   📄 model_comparison_table.pdf (vector format)")
print("\n🎯 You can now view the PNG file or copy it to your local machine!")

plt.show()
