import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Define the models and their metrics
models = [
    "Logistic Regression",
    "XGBoost (Default)",
    "XGBoost (Default - Standard)",
    "XGBoost (Diverse)",
    "XGBoost (Diverse - Standard)",
    "XGBoost (Shifted)",
    "XGBoost (Shifted - Standard)",
    "XGBoost (Special Negatives)",
    "XGBoost (Special Negatives - Standard)"
]

# Validation metrics for each model
accuracy = [0.8621, 0.8621, 0.8621, 0.9677, 0.9597, 0.9583, 0.9514, 0.9862, 0.9655]
precision = [0.8824, 0.8824, 0.8824, 1.0000, 1.0000, 1.0000, 0.9722, 1.0000, 0.9737]
recall = [0.8824, 0.8824, 0.8824, 0.8182, 0.7727, 0.8537, 0.8537, 0.9512, 0.9024]
f1 = [0.8824, 0.8824, 0.8824, 0.9000, 0.8718, 0.9211, 0.9091, 0.9750, 0.9367]
roc_auc = [0.8995, 0.8995, 0.8775, 0.9893, 0.9739, 0.9927, 0.9873, 0.9923, 0.9909]

# Create a DataFrame
df = pd.DataFrame({
    'Model': models,
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1': f1,
    'ROC AUC': roc_auc
})

# Set the Model column as the index
df.set_index('Model', inplace=True)

# Create a figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Model Performance Comparison', fontsize=16)

# Plot 1: Accuracy and Precision
ax1 = axes[0, 0]
df[['Accuracy', 'Precision']].plot(kind='bar', ax=ax1, color=['#3498db', '#2ecc71'])
ax1.set_title('Accuracy and Precision')
ax1.set_ylabel('Score')
ax1.set_ylim(0.7, 1.05)
ax1.grid(axis='y', linestyle='--', alpha=0.7)
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

# Plot 2: Recall and F1
ax2 = axes[0, 1]
df[['Recall', 'F1']].plot(kind='bar', ax=ax2, color=['#e74c3c', '#f39c12'])
ax2.set_title('Recall and F1 Score')
ax2.set_ylabel('Score')
ax2.set_ylim(0.7, 1.05)
ax2.grid(axis='y', linestyle='--', alpha=0.7)
plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

# Plot 3: ROC AUC
ax3 = axes[1, 0]
df['ROC AUC'].plot(kind='bar', ax=ax3, color='#9b59b6')
ax3.set_title('ROC AUC Score')
ax3.set_ylabel('Score')
ax3.set_ylim(0.7, 1.05)
ax3.grid(axis='y', linestyle='--', alpha=0.7)
plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')

# Plot 4: All metrics combined
ax4 = axes[1, 1]
df.plot(kind='bar', ax=ax4, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'])
ax4.set_title('All Metrics Combined')
ax4.set_ylabel('Score')
ax4.set_ylim(0.7, 1.05)
ax4.grid(axis='y', linestyle='--', alpha=0.7)
plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')

# Adjust layout
plt.tight_layout(rect=(0, 0, 1, 0.96))

# Save the figure
output_dir = 'output/model'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'model_comparison_standard.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create a radar chart for the best model (XGBoost Special Negatives)
best_model = "XGBoost (Special Negatives)"
best_model_metrics = df.loc[best_model]

# Number of metrics
num_metrics = len(best_model_metrics)

# Compute angle for each axis
angles = [n / float(num_metrics) * 2 * np.pi for n in range(num_metrics)]
angles = np.concatenate([angles, [angles[0]]])  # Complete the circle

# Add the first value to the end to close the polygon
values = best_model_metrics.values.tolist()
values = np.concatenate([values, [values[0]]])  # Close the polygon

# Create the figure
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

# Plot the data
ax.plot(angles, values, linewidth=2, linestyle='solid', label=best_model)
ax.fill(angles, values, alpha=0.25)

# Set the labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(best_model_metrics.index)

# Set the title
plt.title(f'Performance Radar Chart: {best_model}', y=1.05)

# Save the figure
plt.savefig(os.path.join(output_dir, 'best_model_radar_standard.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create a comparison between the original and standard parameters for Special Negatives
comparison_models = ["XGBoost (Special Negatives)", "XGBoost (Special Negatives - Standard)"]
comparison_df = df.loc[comparison_models]

# Create a figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the data
comparison_df.plot(kind='bar', ax=ax, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'])
ax.set_title('Comparison: Original vs Standard Parameters')
ax.set_ylabel('Score')
ax.set_ylim(0.9, 1.05)
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.setp(ax.get_xticklabels(), rotation=0)

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig(os.path.join(output_dir, 'special_negatives_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"Comparison charts saved to {output_dir}") 