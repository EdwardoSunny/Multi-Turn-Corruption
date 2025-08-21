import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json

# Set style for cleaner plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load data from JSON file
with open('simple_results.json', 'r') as f:
    data = json.load(f)

# Prepare data for plotting
turns = [1, 3, 6, 9]
categories = list(data.keys())

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot lines for each category
for category in categories:
    values = [data[category][f"{turn}_turn" if turn == 1 else f"{turn}_turns"] for turn in turns]
    ax.plot(turns, values, marker='o', linewidth=2.5, markersize=6, 
            label=category.title(), alpha=0.8)

# Customize the plot
ax.set_xlabel('Number of Turns', fontsize=12, fontweight='bold')
ax.set_ylabel('Quality Score', fontsize=12, fontweight='bold')
ax.set_title('Output Quality vs Multi-turn Conversation Length', 
             fontsize=16, fontweight='bold', pad=20)

# Set x-axis ticks
ax.set_xticks(turns)
ax.set_xticklabels([f'{turn} turn{"s" if turn > 1 else ""}' for turn in turns])

# Set y-axis limits for better visualization
ax.set_ylim(6.0, 9.5)

# Add grid
ax.grid(True, alpha=0.3)

# Add legend
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

# Adjust layout to prevent legend cutoff
plt.tight_layout()

# Show the plot
# plt.show()

# Optional: Save the plot
plt.savefig('multiturn_quality.png', dpi=300, bbox_inches='tight')
