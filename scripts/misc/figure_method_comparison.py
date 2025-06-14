import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from io import StringIO

data_str = """species	model	trained_on	base_recall	base_precision	base_f1	exon_recall	exon_precision	exon_f1	gene_recall	gene_precision	gene_f1
Zea mays B73 Chr1	Helixer	77 plant species	80.2	79.5	79.85	57.9	58.0	57.95	23.8	42.3	30.46
Zea mays B73 Chr1	Tiberius	37 mammals	46.9	8.7	14.68	23.5	12.9	16.66	9.8	3.8	5.48
Zea mays B73 Chr1	ANNEVO	Embryophyta	74.8	61.4	67.44	55.4	49.9	52.51	20.8	30.8	24.83
Zea mays B73 Chr1	PlantCad1 + 25 labels CRF	Ath chr1-4 & Osa chr1-11	82.1	27.0	40.64	55.7	21.6	31.13	24.2	6.6	10.37
Zea mays B73 Chr1	PlantCad2 (l24) + BILUO ModernBERT + GateMLP	Ath chr1-5 & Osa chr1-12 @ 90%	73.6	82.4	77.75	56.5	60.9	58.62	19.9	60.2	29.91
Zea mays B73 Chr1	PlantCad2 (l24) + BILUO ModernBERT + GateMLP + CRF	Ath chr1-5 & Osa chr1-12 @ 90%	74.5	86.2	79.92	58.9	73.8	65.51	21.2	61.5	31.53
Zea mays B73 Chr1	PlantCad2 (l24) + BILUO ModernBERT + GateMLP + CRF V2	Ath chr1-5 & Osa chr1-12 @ 97.5%	75.5	89.5	81.91	57.7	79.6	66.9	22.3	60.9	32.65
Zea mays B73 Chr1	PlantCad2 (l24) + 25 labels CRF	Ath chr1-4 & Osa chr1-11	84.7	42.1	56.24	56.9	32.4	41.29	21.4	13.5	16.56
Zea mays B73 Chr1	PlantCad2 (l24) + 25 labels CRF + ReelProtein	Ath chr1-4 & Osa chr1-11	71.4	71.8	71.6	45.2	60.1	51.6	20.5	35.2	25.91
Zea mays B73 Chr1	PlantCad2 (l24) + 25 labels CRF + ReelProtein V2	Ath chr1-4 & Osa chr1-12	50.6	49.1	49.84	32.5	38.1	35.08	14.1	19.0	16.19
Nicotiana tabacum Chr1	Helixer	77 plant species	66.5	69.7	68.06	51.7	55.7	53.63	20.2	29.9	24.11
Nicotiana tabacum Chr1	Tiberius	37 mammals	44.3	22.0	29.4	28.3	26.5	27.37	10.2	5.2	6.89
Nicotiana tabacum Chr1	ANNEVO	Embryophyta	67.6	51.9	58.72	52.0	45.0	48.25	20.3	18.9	19.58
Nicotiana tabacum Chr1	PlantCad1 + 25 labels CRF	Ath chr1-4 & Osa chr1-11	75.6	26.8	39.57	52.2	23.3	32.22	23.3	4.6	7.68
Nicotiana tabacum Chr1	transcript + PlantCad1 + 25 labels CRF	Ath chr1-4 & Osa chr1-11	75.6	26.8	39.57	52.2	23.3	32.22	23.3	4.6	7.68
Nicotiana tabacum Chr1	PlantCad2 (l24) + BILUO ModernBERT + GateMLP + CRF	Ath chr1-5 & Osa chr1-12 @ 90%	64.4	79.5	71.16	54.8	72.4	62.38	21.4	46.3	29.27
Nicotiana tabacum Chr1	PlantCad2 (l24) + BILUO ModernBERT + GateMLP + CRF V2	Ath chr1-5 & Osa chr1-12 @ 97.5%	63.8	78.0	70.19	52.1	71.7	60.35	21.5	46.7	29.44
Nicotiana tabacum Chr1	PlantCad2 (l24) + 25 labels CRF	Ath chr1-4 & Osa chr1-11	69.4	52.7	59.91	48.8	40.1	44.02	16.8	12.0	14.0
Nicotiana tabacum Chr1	PlantCad2 (l24) + 25 labels CRF + ReelProtein	Ath chr1-4 & Osa chr1-11	54.7	61.3	57.81	35.5	50.2	41.59	15.1	21.9	17.88
Nicotiana tabacum Chr1	PlantCad2 (l24) + 25 labels CRF + ReelProtein V2	Ath chr1-4 & Osa chr1-12	18.6	60.9	28.5	12.2	49.5	19.58	5.2	20.2	8.27
Nicotiana tabacum Chr1	PlantCad2 (l48) + 25 labels CRF	Ath chr1-4 & Osa chr1-11	71.1	27.3	39.45	44.6	18.4	26.05	11.3	2.8	4.49
Nicotiana tabacum Chr1	transcript + PlantCad2 (l24) + 25 labels CRF	Ath chr1-4 & Osa chr1-11	66.5	36.3	46.96	25.7	12.5	16.82	11.1	2.3	3.81
Nicotiana sylvestris Chr1	Helixer	77 plant species	66.3	81.4	73.08	61.7	63.3	62.49	21.2	38.8	27.42
Nicotiana sylvestris Chr1	Tiberius	37 mammals	34.4	19.1	24.56	25.1	22.3	23.62	7.1	4.4	5.43
Nicotiana sylvestris Chr1	ANNEVO	Embryophyta	63.0	56.2	59.41	58.6	49.0	53.37	20.2	23.3	21.64
Nicotiana sylvestris Chr1	PlantCad1 + 25 labels CRF	Ath chr1-4 & Osa chr1-11	68.0	25.0	36.56	59.8	25.7	35.95	23.1	5.5	8.88
Nicotiana sylvestris Chr1	transcript + PlantCad1 + 25 labels CRF	Ath chr1-4 & Osa chr1-11	63.5	38.5	47.94	29.8	13.6	18.68	10.9	2.7	4.33
Nicotiana sylvestris Chr1	PlantCad2 (l24) + BILUO ModernBERT + GateMLP + CRF	Ath chr1-5 & Osa chr1-12 @ 90%	62.4	90.9	74.0	61.6	78.7	69.11	23.2	61.6	33.71
Nicotiana sylvestris Chr1	PlantCad2 (l24) + BILUO ModernBERT + GateMLP + CRF V2	Ath chr1-5 & Osa chr1-12 @ 97.5%	63.4	91.9	75.03	61.3	81.2	69.86	23.5	62.2	34.11
Nicotiana sylvestris Chr1	PlantCad2 (l24) + 25 labels CRF	Ath chr1-4 & Osa chr1-11	71.1	64.2	67.47	58.7	46.6	51.95	17.1	15.2	16.09
Nicotiana sylvestris Chr1	PlantCad2 (l24) + 25 labels CRF + ReelProtein	Ath chr1-4 & Osa chr1-11	56.5	74.9	64.41	43.7	59.3	50.32	15.7	27.9	20.09
Nicotiana sylvestris Chr1	PlantCad2 (l24) + 25 labels CRF + ReelProtein V2	Ath chr1-4 & Osa chr1-12	20.5	75.8	32.27	16.0	58.7	25.15	5.8	27.3	9.57
Nicotiana sylvestris Chr1	PlantCad2 (l48) + 25 labels CRF	Ath chr1-4 & Osa chr1-11	65.9	29.1	40.37	51.2	19.6	28.35	11.0	3.3	5.08
Nicotiana sylvestris Chr1	transcript + PlantCad2 (l24) + 25 labels CRF	Ath chr1-4 & Osa chr1-11	63.3	40.2	49.17	29.8	13.9	18.96	10.6	2.7	4.3
Coffea arabica Chr1c	Helixer	77 plant species	83.8	67.0	74.46	64.6	49.7	56.18	27.4	26.6	26.99
Coffea arabica Chr1c	Tiberius	37 mammals	39.0	37.3	38.13	24.4	36.2	29.15	11.6	10.3	10.91
Coffea arabica Chr1c	ANNEVO	Embryophyta	78.3	64.3	70.61	60.9	47.6	53.43	26.3	25.0	25.63
Coffea arabica Chr1c	PlantCad2 (l24) + BILUO ModernBERT + GateMLP	Ath chr1-5 & Osa chr1-12 @ 90%	78.6	77.4	78.0	65.8	60.8	63.2	28.8	42.9	34.46
Coffea arabica Chr1c	PlantCad2 (l24) + BILUO ModernBERT + GateMLP + CRF	Ath chr1-5 & Osa chr1-12 @ 90%	79.5	77.4	78.44	67.1	67.9	67.5	30.3	43.2	35.62
Coffea arabica Chr1c	PlantCad2 (l24) + BILUO ModernBERT + GateMLP + CRF V2	Ath chr1-5 & Osa chr1-12 @ 97.5%	79.4	76.0	77.66	66.1	69.2	67.61	32.8	45.8	38.22
Juglans regia Chr1	Helixer	77 plant species	90.2	69.1	78.25	63.9	58.7	61.19	33.6	41.6	37.17
Juglans regia Chr1	Tiberius	37 mammals	51.6	66.7	58.19	29.6	52.7	37.91	12.0	17.8	14.34
Juglans regia Chr1	ANNEVO	Embryophyta	88.8	73.2	80.25	66.2	59.3	62.56	30.7	37.2	33.64
Juglans regia Chr1	PlantCad2 (l24) + BILUO ModernBERT + GateMLP	Ath chr1-5 & Osa chr1-12 @ 90%	88.7	75.8	81.74	70.7	66.8	68.69	33.7	53.4	41.32
Juglans regia Chr1	PlantCad2 (l24) + BILUO ModernBERT + GateMLP + CRF	Ath chr1-5 & Osa chr1-12 @ 90%	89.2	76.9	82.59	71.3	73.6	72.43	36.1	55.1	43.62
Juglans regia Chr1	PlantCad2 (l24) + BILUO ModernBERT + GateMLP + CRF V2	Ath chr1-5 & Osa chr1-12 @ 97.5%	88.4	75.2	81.27	67.7	72.4	69.97	37.0	56.6	44.75
Phaseolus vulgaris Chr1	Helixer	77 plant species	73.9	82.3	77.87	69.0	71.1	70.03	26.6	35.7	30.49
Phaseolus vulgaris Chr1	Tiberius	37 mammals	38.7	36.0	37.3	28.1	39.2	32.73	8.3	7.8	8.04
Phaseolus vulgaris Chr1	ANNEVO	Embryophyta	69.9	79.3	74.3	67.4	65.5	66.44	25.4	30.6	27.76
Phaseolus vulgaris Chr1	PlantCad2 (l24) + BILUO ModernBERT + GateMLP	Ath chr1-5 & Osa chr1-12 @ 90%	70.4	86.1	77.46	69.3	73.3	71.24	27.4	41.9	33.13
Phaseolus vulgaris Chr1	PlantCad2 (l24) + BILUO ModernBERT + GateMLP + CRF	Ath chr1-5 & Osa chr1-12 @ 90%	70.9	87.1	78.17	71.3	79.5	75.18	29.5	44.0	35.32
Phaseolus vulgaris Chr1	PlantCad2 (l24) + BILUO ModernBERT + GateMLP + CRF V2	Ath chr1-5 & Osa chr1-12 @ 97.5%	72.4	85.0	78.2	70.4	79.1	74.5	30.7	45.3	36.6"""

# Shared variables
SPECIES_MAP = {
    'Zea mays B73 Chr1': 'Z. mays',
    'Nicotiana tabacum Chr1': 'N. tabacum', 
    'Nicotiana sylvestris Chr1': 'N. sylvestris',
    'Coffea arabica Chr1c': 'C. arabica',
    'Juglans regia Chr1': 'J. regia',
    'Phaseolus vulgaris Chr1': 'P. vulgaris'
}

MODEL_MAP = {
    "PlantCad2 (l24) + BILUO ModernBERT + GateMLP + CRF V2": "ReelAnnote (Ours)",
    "Tiberius": "Tiberius",
    "Helixer": "Helixer", 
    "ANNEVO": "ANNEVO"
}

# Colors for tools (using colorblind-friendly palette)
MODEL_COLORS = {
    "ReelAnnote (Ours)": '#1f77b4',  # blue
    "Tiberius": '#9467bd',           # purple
    "Helixer": '#17becf',            # cyan
    "ANNEVO": '#2ca02c'              # green
}

# Shapes for species
SPECIES_MARKERS = {
    'C. arabica': 'o',      # circle
    'J. regia': 's',        # square  
    'N. sylvestris': '^',   # triangle up
    'N. tabacum': 'D',      # diamond
    'P. vulgaris': 'v',     # triangle down
    'Z. mays': 'h'          # hexagon
}

# Hatching patterns for tools (colorblind friendly)
MODEL_HATCHES = {
    "ReelAnnote (Ours)": '....',  # dots
    "Tiberius": 'ooo',            # little circles
    "Helixer": '///',             # forward slashes
    "ANNEVO": '\\\\\\'            # backward slashes
}

TARGET_MODELS = [
    "PlantCad2 (l24) + BILUO ModernBERT + GateMLP + CRF V2",
    "Tiberius", 
    "Helixer", 
    "ANNEVO"
]

# Output directory for all figures
OUTPUT_DIR = Path('local/scratch/figures')


def prepare_data():
    """Prepare and filter the data for plotting."""
    # Parse data
    df = pd.read_csv(StringIO(data_str), sep='\t')
    
    # Filter to target models
    df = df[df['model'].isin(TARGET_MODELS)]
    
    # Add species short labels
    df['species_short'] = df['species'].map(SPECIES_MAP)
    
    # Add model display names
    df['model_display'] = df['model'].map(MODEL_MAP)
    
    return df


def create_tool_comparison_figure():
    """Create a precision-recall plot comparing gene annotation tools."""
    
    df = prepare_data()
    
    # Setup plot
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot F1 isocurves
    recall_range = np.linspace(5, 40, 1000)  # Extend to right edge of axis
    f1_values = [10, 20, 30, 40, 50]  # F1 values 0.1 to 0.5 by 0.1 increments
    
    for f1 in f1_values:
        # F1 = 2PR/(P+R), solve for P: P = F1*R/(2*R - F1)
        # Use 0-1 scale for calculation
        f1_norm = f1 / 100.0
        recall_norm = recall_range / 100.0
        
        # Avoid division by zero: only calculate where 2*R > F1
        valid_recall = recall_norm > f1_norm / 2
        recall_subset = recall_norm[valid_recall]
        
        if len(recall_subset) > 0:
            precision_norm = (f1_norm * recall_subset) / (2 * recall_subset - f1_norm)
            precision_subset = precision_norm * 100
            recall_plot = recall_subset * 100
            
            # Plot all mathematically valid values within axis range
            plot_mask = (precision_subset > 0) & (recall_plot >= 5) & (recall_plot <= 40)
            
            if np.any(plot_mask):
                ax.plot(recall_plot[plot_mask], precision_subset[plot_mask], 
                       'k--', alpha=0.4, linewidth=1, zorder=1)
                
                # Add F1 labels aligned vertically at x=34
                if len(recall_plot[plot_mask]) > 10:
                    # Find the point on the curve closest to x=34
                    target_x = 34
                    distances = np.abs(recall_plot[plot_mask] - target_x)
                    closest_idx = np.argmin(distances)
                    
                    x_pos = target_x  # Use fixed x position for alignment
                    y_pos = precision_subset[plot_mask][closest_idx]
                    
                    if y_pos <= 60 and recall_plot[plot_mask][closest_idx] >= 25:  # Only if curve reaches this x position
                        # Add extra offset for F1=0.4 to avoid overlap
                        y_offset = 4 if f1_norm == 0.4 else 2
                        ax.text(x_pos, y_pos + y_offset, f'F1={f1_norm:.1f}', 
                               fontsize=12, alpha=0.8, ha='center',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='none'))
    
    # Plot data points
    for species in df['species_short'].unique():
        species_data = df[df['species_short'] == species]
        
        for model in species_data['model_display'].unique():
            model_data = species_data[species_data['model_display'] == model]
            
            # Set marker size based on tool
            base_size = 360
            size_multipliers = {
                "ReelAnnote (Ours)": 1.0,  # 100%
                "Tiberius": 0.8,           # 80%
                "ANNEVO": 0.8,             # 80%
                "Helixer": 0.8             # 80%
            }
            marker_size = int(base_size * size_multipliers[model])
            
            # Add black border for all markers (thicker for ReelAnnote)
            edge_color = 'black'
            edge_width = 2.5 if model == "ReelAnnote (Ours)" else .8
            
            ax.scatter(model_data['gene_recall'], model_data['gene_precision'],
                      c=MODEL_COLORS[model], marker=SPECIES_MARKERS[species],
                      s=marker_size, alpha=0.7, edgecolors=edge_color, linewidth=edge_width,
                      zorder=3, label=f'{species} - {model}' if species == 'C. arabica' else '')
            
            # Add tool-specific hatching overlay
            ax.scatter(model_data['gene_recall'], model_data['gene_precision'],
                      facecolors='none', marker=SPECIES_MARKERS[species],
                      s=marker_size, alpha=0.5, edgecolors='gray', linewidth=0.8,
                      hatch=MODEL_HATCHES[model], zorder=-1)
    
    # Customize plot with larger fonts
    ax.set_xlabel('Recall (%)', fontsize=16)
    ax.set_ylabel('Precision (%)', fontsize=16)
    ax.set_title('Gene Annotation Performance by Method', fontsize=18, pad=20)
    
    # Make tick labels larger
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Set axis limits to focus on the data range
    ax.set_xlim(5, 40)
    ax.set_ylim(0, 65)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Create legends
    # Species legend (shapes)
    species_handles = []
    for species, marker in SPECIES_MARKERS.items():
        # Use 3x larger size for legend markers too
        legend_size = 300
        handle = plt.scatter([], [], c='black', marker=marker, s=legend_size, 
                           edgecolors='white', linewidth=1.5, label=species)
        species_handles.append(handle)
    
    # Model legend (colors) - use wider rectangles with hatching
    from matplotlib.patches import Rectangle
    model_handles = []
    for model, color in MODEL_COLORS.items():
        # Add black border for all tools in legend (thicker for ReelAnnote)
        edge_color = 'black'
        edge_width = 2.5 if model == "ReelAnnote (Ours)" else 1.2
        
        # Create marker with tool-specific hatching
        from matplotlib.patches import Patch
        handle = Patch(facecolor=color, edgecolor=edge_color, linewidth=edge_width,
                      hatch=MODEL_HATCHES[model], alpha=0.7, label=model)
        model_handles.append(handle)
    
    # Position legends below the figure with more spacing
    species_legend = ax.legend(handles=species_handles, title='Species', 
                              loc='upper center', bbox_to_anchor=(0.26, -0.13), ncol=3,
                              labelspacing=.5, columnspacing=1.5, title_fontsize=14, fontsize=12,
                              handleheight=2.0)
    ax.add_artist(species_legend)
    
    model_legend = ax.legend(handles=model_handles, title='Method',
                            loc='upper center', bbox_to_anchor=(0.77, -0.13), ncol=2,
                            labelspacing=.5, columnspacing=1.5, title_fontsize=14, fontsize=12,
                            handlelength=3.5, handletextpad=0.8, handleheight=2.0)
    
    # Style legends
    for legend in [species_legend, model_legend]:
        legend.get_frame().set_alpha(0.9)
        legend.get_frame().set_edgecolor('black')
        # legend.get_title().set_fontweight('bold')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.35)  # Make even more room for legends with extra spacing
    
    # Save figure
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save as both PDF and PNG
    pdf_path = OUTPUT_DIR / 'figure_method_scatterplot.pdf'
    png_path = OUTPUT_DIR / 'figure_method_scatterplot.png'
    
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    print(f"Scatterplot saved to: {pdf_path} and {png_path}")
    
    plt.show()


def create_radar_plots():
    """Create radar plots comparing F1 scores across gene, exon, and base levels."""
    df = prepare_data()
    
    # Define the metrics to plot (in order: gene, exon, base)
    metrics = ['gene_f1', 'exon_f1', 'base_f1']
    metric_titles = ['Gene F1', 'Exon F1', 'Base F1']
    
    # Create figure with 1:4 aspect ratio (tall and skinny), tighter spacing
    fig, axes = plt.subplots(3, 1, figsize=(4, 16), subplot_kw=dict(projection='polar'))
    
    # Get species and models - only the 3 main tools for radar plots
    species_list = list(SPECIES_MARKERS.keys())
    models = ["ANNEVO", "Helixer", "ReelAnnote (Ours)"]  # Only these 3 tools for radar plots
    
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[i]
        
        # Number of species (radar spokes)
        N = len(species_list)
        
        # Compute angle for each spoke
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Collect all values for this metric to determine range
        all_values = []
        model_data = {}
        
        # Plot each model
        for model in models:
            # Get F1 scores for this model across all species
            values = []
            for species in species_list:
                model_species_data = df[(df['model_display'] == model) & (df['species_short'] == species)]
                if not model_species_data.empty:
                    val = model_species_data[metric].iloc[0]
                    values.append(val)
                    all_values.append(val)
                else:
                    values.append(0)  # If no data, use 0
                    all_values.append(0)
            
            values += values[:1]  # Complete the circle
            model_data[model] = values
        
        # Calculate relative y-axis range
        min_val = min(all_values)
        max_val = max(all_values)
        range_padding = (max_val - min_val) * 0.1  # 10% padding
        y_min = max(0, min_val - range_padding)  # Don't go below 0
        y_max = max_val + range_padding
        
        # Plot each model with calculated data
        for model in models:
            values = model_data[model]
            
            # Plot the radar line
            ax.plot(angles, values, 'o-', linewidth=4, alpha=0.7,
                   color=MODEL_COLORS[model], markersize=4)
        
        # Customize each subplot
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([])  # Remove tick labels
        
        # Add title on the left side, rotated 90 degrees, inside the plot
        ax.text(0.15, 0.5, title, transform=ax.transAxes, fontsize=20, 
                rotation=90, ha='center', va='center')
        
        # Set y-axis (radial) limits and labels relative to data
        ax.set_ylim(y_min, y_max)
        
        # Create 4 evenly spaced ticks between min and max
        tick_values = np.linspace(y_min, y_max, 5)[1:-1]  # Skip first and last to avoid clutter
        ax.set_yticks(tick_values)
        ax.set_yticklabels([f'{val:.0f}' for val in tick_values], fontsize=16, zorder=10,
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))
        
        # Add species markers outside the plot using text positioning (only for first plot)
        if i == 0:  # Only show species markers on the first plot (Gene F1)
            # Map matplotlib markers to Unicode shape symbols
            marker_symbols = {
                'o': '●',      # circle
                's': '■',      # square  
                '^': '▲',      # triangle up
                'D': '◆',      # diamond
                'v': '▼',      # triangle down
                'h': '⬢'       # hexagon
            }
            
            for j, (angle, species) in enumerate(zip(angles[:-1], species_list)):
                # Convert polar angle to x,y position outside the axes
                # Increase distance to place shapes well outside the plot rings
                distance = 0.57  # Increased distance to place outside the rings
                # Use the exact same angle as the radar spokes (no additional rotation)
                x_offset = distance * np.cos(angle)
                y_offset = distance * np.sin(angle)
                
                # Use axes coordinates (0.5, 0.5) is center, add offset to place outside
                x_pos = 0.5 + x_offset
                y_pos = 0.5 + y_offset
                
                # Get the shape symbol for this species
                shape_symbol = marker_symbols[SPECIES_MARKERS[species]]
                
                # Add text with shape symbol using axes coordinates
                ax.text(x_pos, y_pos, shape_symbol, 
                       transform=ax.transAxes, fontsize=24, ha='center', va='center',
                       color='black', weight='bold')
        
        # Add grid
        ax.grid(True, alpha=0.3)
    
    # Remove vertical spacing between subplots
    plt.subplots_adjust(hspace=-0.45)
    
    # Save figure
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save as both PDF and PNG
    pdf_path = OUTPUT_DIR / 'figure_method_radar.pdf'
    png_path = OUTPUT_DIR / 'figure_method_radar.png'
    
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    print(f"Radar plots saved to: {pdf_path} and {png_path}")
    
    plt.show()


def combine_png_plots():
    """Combine the scatterplot and radar plot PNGs into a single PDF figure."""
    # Paths to the individual PNG files
    scatter_png = OUTPUT_DIR / 'figure_method_scatterplot.png'
    radar_png = OUTPUT_DIR / 'figure_method_radar.png'
    
    # Load PNGs using matplotlib
    scatter_img = plt.imread(scatter_png)
    radar_img = plt.imread(radar_png)
    
    # Create combined figure with custom width ratios and smaller overall size
    fig = plt.figure(figsize=(14, 8))
    
    # Use gridspec for more control - give more space to scatterplot, less to radar
    gs = fig.add_gridspec(1, 2, width_ratios=[3.7, 1.0], wspace=-.018)
    
    # Create subplots
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # Display images
    ax1.imshow(scatter_img)
    ax1.axis('off')
    ax1.set_title('', pad=0)
    
    ax2.imshow(radar_img)
    ax2.axis('off')
    ax2.set_title('', pad=0)
    
    # Push radar plots down by adjusting their position
    ax2_pos = ax2.get_position()
    ax2.set_position([ax2_pos.x0, ax2_pos.y0 - 0.02, ax2_pos.width, ax2_pos.height])
    
    # Save combined figure
    combined_pdf_path = OUTPUT_DIR / 'figure_method_comparison_combined.pdf'
    combined_png_path = OUTPUT_DIR / 'figure_method_comparison_combined.png'
    
    plt.savefig(combined_pdf_path, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.savefig(combined_png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
    print(f"Combined figure saved to: {combined_pdf_path} and {combined_png_path}")
    
    plt.show()


if __name__ == "__main__":
    create_tool_comparison_figure()
    create_radar_plots()
    combine_png_plots()
