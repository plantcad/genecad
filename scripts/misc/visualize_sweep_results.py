import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np

# Parse the multiline string into a DataFrame
data = """
CDS F1	sweep-2__cfg_009__arch_classifier-only__frzn_no__lr_1e-04	0.7276001572608948
CDS F1	sweep-2__cfg_001__arch_encoder-only__frzn_yes__lr_1e-04	0.6359251737594604
CDS F1	sweep-2__cfg_010__arch_classifier-only__frzn_yes__lr_1e-03	0.39263486862182617
CDS F1	sweep-2__cfg_002__arch_encoder-only__frzn_yes__lr_1e-03	0.3635783791542053
CDS F1	sweep-2__cfg_000__arch_encoder-only__frzn_yes__lr_1e-05	0.20073825120925903
CDS F1	sweep-2__cfg_011__arch_classifier-only__frzn_no__lr_1e-03	0.18508309125900269
CDS F1	sweep-2__cfg_008__arch_classifier-only__frzn_yes__lr_1e-04	0.1821092963218689
CDS F1	sweep-2__cfg_007__arch_classifier-only__frzn_no__lr_1e-05	0.048799384385347366
CDS F1	sweep-2__cfg_004__arch_sequence-only__frzn_yes__lr_1e-04	0.00006118453165981919
CDS F1	sweep-2__cfg_006__arch_classifier-only__frzn_yes__lr_1e-05	0.0000075319353527447674
CDS F1	sweep-2__cfg_003__arch_sequence-only__frzn_yes__lr_1e-05	0
CDS F1	sweep-2__cfg_005__arch_sequence-only__frzn_yes__lr_1e-03	0
Token F1	sweep-2__cfg_009__arch_classifier-only__frzn_no__lr_1e-04	0.5443550944328308
Token F1	sweep-2__cfg_010__arch_classifier-only__frzn_yes__lr_1e-03	0.5199611186981201
Token F1	sweep-2__cfg_001__arch_encoder-only__frzn_yes__lr_1e-04	0.5061268210411072
Token F1	sweep-2__cfg_002__arch_encoder-only__frzn_yes__lr_1e-03	0.43929845094680786
Token F1	sweep-2__cfg_008__arch_classifier-only__frzn_yes__lr_1e-04	0.4151017963886261
Token F1	sweep-2__cfg_011__arch_classifier-only__frzn_no__lr_1e-03	0.39564773440361023
Token F1	sweep-2__cfg_000__arch_encoder-only__frzn_yes__lr_1e-05	0.34612560272216797
Token F1	sweep-2__cfg_007__arch_classifier-only__frzn_no__lr_1e-05	0.23986288905143738
Token F1	sweep-2__cfg_004__arch_sequence-only__frzn_yes__lr_1e-04	0.14854471385478973
Token F1	sweep-2__cfg_006__arch_classifier-only__frzn_yes__lr_1e-05	0.09972843527793884
Token F1	sweep-2__cfg_003__arch_sequence-only__frzn_yes__lr_1e-05	0.09293826669454575
Token F1	sweep-2__cfg_005__arch_sequence-only__frzn_yes__lr_1e-03	0.05061815679073334
Entity F1	sweep-2__cfg_009__arch_classifier-only__frzn_no__lr_1e-04	0.3075734078884125
Entity F1	sweep-2__cfg_001__arch_encoder-only__frzn_yes__lr_1e-04	0.254525750875473
Entity F1	sweep-2__cfg_002__arch_encoder-only__frzn_yes__lr_1e-03	0.16698308289051056
Entity F1	sweep-2__cfg_010__arch_classifier-only__frzn_yes__lr_1e-03	0.1102740690112114
Entity F1	sweep-2__cfg_011__arch_classifier-only__frzn_no__lr_1e-03	0.10926404595375061
Entity F1	sweep-2__cfg_000__arch_encoder-only__frzn_yes__lr_1e-05	0.0832553580403328
Entity F1	sweep-2__cfg_008__arch_classifier-only__frzn_yes__lr_1e-04	0.05649743974208832
Entity F1	sweep-2__cfg_007__arch_classifier-only__frzn_no__lr_1e-05	0.02153610810637474
Entity F1	sweep-2__cfg_004__arch_sequence-only__frzn_yes__lr_1e-04	0.00006038461287971586
Entity F1	sweep-2__cfg_003__arch_sequence-only__frzn_yes__lr_1e-05	0.00001590663850947749
Entity F1	sweep-2__cfg_006__arch_classifier-only__frzn_yes__lr_1e-05	0.0000012553225587907946
Entity F1	sweep-2__cfg_005__arch_sequence-only__frzn_yes__lr_1e-03	0
"""

lines = [line.strip() for line in data.strip().split('\n') if line.strip()]
rows = []

for line in lines:
    metric, config, value = line.split('\t')
    rows.append({
        'Metric': metric,
        'Config': config,
        'Value': float(value)
    })

df = pd.DataFrame(rows)

# Extract configuration details using regex
pattern = r'sweep-2__cfg_\d+__arch_([^_]+(?:-[^_]+)?)__frzn_([^_]+)__lr_(\d+e-\d+)'
df['Architecture'] = df['Config'].apply(lambda x: re.search(pattern, x).group(1))
df['Frozen'] = df['Config'].apply(lambda x: re.search(pattern, x).group(2))
df['Learning Rate'] = df['Config'].apply(lambda x: re.search(pattern, x).group(3))

# Create a more readable label for each configuration
label_mapping = {
    'encoder-only__frzn_yes': 'BILUO BERT + Frozen PCv2',
    'classifier-only__frzn_yes': 'MLP + Frozen PCv2',
    'classifier-only__frzn_no': 'MLP + Tuned PCv2',
    'sequence-only__frzn_yes': 'BILUO BERT Only'
}

df['Model'] = df.apply(
    lambda row: label_mapping.get(f"{row['Architecture']}__frzn_{row['Frozen']}", 
                                 f"{row['Architecture']} (Frozen: {row['Frozen']})"),
    axis=1
)

# Simplify learning rate labels and ensure they're properly ordered
df['LR Label'] = df['Learning Rate'].map({
    '1e-03': 'LR: 1e-3',
    '1e-04': 'LR: 1e-4',
    '1e-05': 'LR: 1e-5'
})

# Ensure learning rates are in the desired order (larger to smaller)
lr_order = {'1e-03': 0, '1e-04': 1, '1e-05': 2}
df['LR Order'] = df['Learning Rate'].map(lr_order)
df = df.sort_values('LR Order')

# Order metrics appropriately for consistent plotting
metric_order = ['CDS F1', 'Token F1', 'Entity F1']
df['Metric'] = pd.Categorical(df['Metric'], categories=metric_order, ordered=True)

# Calculate average score for each model to determine ordering
model_avg_scores = df.groupby('Model')['Value'].mean().reset_index()
model_avg_scores = model_avg_scores.sort_values('Value', ascending=False)
model_order = model_avg_scores['Model'].tolist()

# Define a pleasing color palette
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']  # Blue, Orange, Green, Red

# Create the figure with multiple subplots (one for each learning rate)
fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=True, constrained_layout=True)
fig.suptitle('Performance Comparison of Model Architectures', fontsize=18, fontweight='bold')
fig.text(0.5, 0.92, 'Across Different Learning Rates and Evaluation Metrics', 
         ha='center', fontsize=14)
fig.text(0.05, 0.5, 'F1 Score', va='center', rotation='vertical', fontsize=14, fontweight='bold')

# Plot each learning rate in a separate subplot
for i, lr in enumerate(['1e-03', '1e-04', '1e-05']):
    lr_data = df[df['Learning Rate'] == lr]
    ax = axes[i]
    
    # Get all model types for this learning rate, sorted by overall average score
    models_in_lr = lr_data['Model'].unique().tolist()
    # Sort models based on the global model_order while preserving only models that exist in this LR
    models = [model for model in model_order if model in models_in_lr]
    
    # Set up positions for grouped bars
    x = np.arange(len(metric_order))
    width = 0.75 / len(models)
    
    # Plot bars for each model
    for j, model in enumerate(models):
        model_data = lr_data[lr_data['Model'] == model]
        values = []
        for metric in metric_order:
            val = model_data[model_data['Metric'] == metric]['Value'].values
            values.append(val[0] if len(val) > 0 else 0)
        
        # Plot this model's bars
        offset = j - (len(models) - 1) / 2
        bars = ax.bar(x + offset * width, values, width, label=model if i == 0 else "", 
                 color=colors[j % len(colors)])
        
        # Add value labels on top of bars
        for b, v in zip(bars, values):
            if v >= 0.01:  # Only label bars with non-negligible values
                ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                      f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Set the title and other properties
    ax.set_title(f'Learning Rate: {lr}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_order, rotation=45, ha='right')
    ax.set_ylim(0, 0.85)  # Set y-limit with some padding
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Only add y-ticks to the leftmost subplot
    if i != 0:
        ax.set_yticklabels([])

# Add a legend at the bottom
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01),
          ncol=len(models), frameon=True, fontsize=12)

plt.tight_layout(rect=[0.05, 0.07, 0.95, 0.9])

plt.savefig('local/results/sweep_results_visualization.pdf', bbox_inches='tight')

plt.show()