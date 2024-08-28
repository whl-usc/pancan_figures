"""
contact:    wlee9829@gmail.com
date:       2024_08_06
python:     python3.10
script:     dge_boxwhisker_multi.py

This Python script compares the distribution of normalized data
between up to three genes by plotting and performing significance tests.
"""

# Define version
__version__ = "1.3.0"

# Version notes
__update_notes__ = """
2.0.0
    -   Extended the script to handle up to four input datasets.
    -   Revised significance star placement to be staggered at the top.
    -   Centered significance brackets based on boxplot positions.

1.1.0
    -   Incorporated significance calculations between all pairs of genes.
    -   Adjusted file handling to derive gene names from file paths.
    -   Retained argparse logic with positional arguments for file paths.

1.0.0
    -   Initial commit, basic plotting logic.
"""

# Import packages
from datetime import datetime
from scipy.stats import mannwhitneyu
from scipy import stats
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import textwrap

###########################################################################
# 1. Set up functions

def read_input(file_path1, file_path2, file_path3=None, file_path4=None, placeholder_value=-9999):
    """
    Read the data for RNA expression counts for up to four genes from
    separate CSV files.

    Args:
        file_path1 (str): CSV file containing expression data for the first gene.
        file_path2 (str): CSV file containing expression data for the second gene.
        file_path3 (str): CSV file containing expression data for the third gene (optional).
        file_path4 (str): CSV file containing expression data for the fourth gene (optional).
        placeholder_value (float): Placeholder value for missing data.

    Returns:
        tuple: DataFrames with filtered tumor columns for each gene.
    """
    def filter_tumor_columns(data):
        data.columns = [col.strip() for col in data.columns]
        tumor_columns = [col for col in data.columns if col.endswith('Tumor')]
        filtered_data = data[tumor_columns] if tumor_columns else pd.DataFrame()
        return filtered_data

    data1 = pd.read_csv(file_path1).fillna(placeholder_value)
    data2 = pd.read_csv(file_path2).fillna(placeholder_value)
    data3 = pd.read_csv(file_path3).fillna(placeholder_value) if file_path3 else pd.DataFrame()
    data4 = pd.read_csv(file_path4).fillna(placeholder_value) if file_path4 else pd.DataFrame()

    return (filter_tumor_columns(data1), 
            filter_tumor_columns(data2), 
            filter_tumor_columns(data3), 
            filter_tumor_columns(data4))


def plot(dataframe1, dataframe2, dataframe3=None, dataframe4=None, 
         gene1_name=None, gene2_name=None, gene3_name=None, gene4_name=None, 
         placeholder_value=-9999):
    # Combine dataframes
    if dataframe4 is not None:
        dataframe1.columns = [f'{gene1_name}_{col}' for col in dataframe1.columns]
        dataframe2.columns = [f'{gene2_name}_{col}' for col in dataframe2.columns]
        dataframe3.columns = [f'{gene3_name}_{col}' for col in dataframe3.columns]
        dataframe4.columns = [f'{gene4_name}_{col}' for col in dataframe4.columns]
        combined_df = pd.concat([dataframe1, dataframe2, dataframe3, dataframe4], axis=1)
    elif dataframe3 is not None:
        dataframe1.columns = [f'{gene1_name}_{col}' for col in dataframe1.columns]
        dataframe2.columns = [f'{gene2_name}_{col}' for col in dataframe2.columns]
        dataframe3.columns = [f'{gene3_name}_{col}' for col in dataframe3.columns]
        combined_df = pd.concat([dataframe1, dataframe2, dataframe3], axis=1)
    else:
        dataframe1.columns = [f'{gene1_name}_{col}' for col in dataframe1.columns]
        dataframe2.columns = [f'{gene2_name}_{col}' for col in dataframe2.columns]
        combined_df = pd.concat([dataframe1, dataframe2], axis=1)

    # Melt and preprocess dataframe
    df = pd.melt(combined_df, var_name='tissue_type', value_name='expression')
    df['gene'] = df['tissue_type'].apply(lambda x: x.split('_')[0])
    df['tissue_type'] = df['tissue_type'].apply(lambda x: x.split('_', 1)[1])
    df = df[df['expression'] != placeholder_value]

    # Define color palette
    color_palette = {
        gene1_name: 'blue',
        gene2_name: 'red',
        gene3_name: 'green',
        gene4_name: 'purple'
    }

    plt.figure(figsize=(16, 10))

    # Plot the data
    ax = sns.boxplot(
        x='tissue_type', y='expression', data=df, hue='gene', 
        palette=color_palette, whis=[0, 100], linewidth=2, fliersize=0.5, showcaps=True,
        boxprops={'facecolor': 'none', 'edgecolor': 'black', 'linewidth': 0.75},
        whiskerprops={'color': 'black', 'linewidth': 0.75},
        medianprops={'color': 'black', 'linewidth': 0.75},
        capprops={'color': 'gray', 'linewidth': 0.5}
    )
    sns.stripplot(
        x='tissue_type', y='expression', data=df, hue='gene', 
        palette=color_palette, jitter=True, edgecolor='black', size=5, alpha=0.25, ax=ax, dodge=True, legend=False
    )

    ax.set_xlabel('', fontsize=8, fontweight='bold')
    ax.set_ylabel('Gene Expression', fontsize=8, fontweight='bold')

    # Get tissue types for x-axis labels
    tissue_types = df['tissue_type'].unique()
    x_labels = []
    for tissue_type in tissue_types:
        counts = [
            f"{gene1_name}: n={df[(df['tissue_type'] == tissue_type) & (df['gene'] == gene1_name)].shape[0]}",
            f"{gene2_name}: n={df[(df['tissue_type'] == tissue_type) & (df['gene'] == gene2_name)].shape[0]}"
        ]
        if gene3_name:
            counts.append(f"{gene3_name}: n={df[(df['tissue_type'] == tissue_type) & (df['gene'] == gene3_name)].shape[0]}")
        if gene4_name:
            counts.append(f"{gene4_name}: n={df[(df['tissue_type'] == tissue_type) & (df['gene'] == gene4_name)].shape[0]}")
        x_labels.append(f"{tissue_type} ({', '.join(counts)})")

    ax.set_xticks(range(len(tissue_types)))
    ax.set_xticklabels(x_labels, rotation=45, rotation_mode='anchor', ha='right', fontsize=8)
    ax.set_xlim(-0.75, len(tissue_types) + 1.5)

    # Add alternating background colors
    colors = ['lightgray', 'white']
    for i, tissue_type in enumerate(tissue_types):
        start_idx = i - 0.5
        end_idx = i + 0.5
        color = colors[i % 2]
        ax.axvspan(start_idx, end_idx, alpha=0.15, color=color, zorder=-1)

    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    def add_significance(ax, x1, x2, y, star, color='black'):
        """Add significance stars to the plot with brackets spanning the gene comparisons."""
        ax.plot([x1, x1, x2, x2], [y, y + 0.1, y + 0.1, y], color=color, lw=0.75)
        ax.text((x1 + x2) / 2, y, star, ha='center', va='bottom', color=color, fontsize=8)

    # Extract positions and labels from the x-axis
    positions = ax.get_xticks()
    ticklabels = [ticklabel.get_text() for ticklabel in ax.get_xticklabels()]
    
    # Create a mappping from tissue type to positions
    tissue_to_pos = {}
    for label, pos in zip(ticklabels, positions):
        for tissue_type in df['tissue_type'].unique():
            if tissue_type in label:
                tissue_to_pos[tissue_type] = pos
                break

    # Create a mapping for gene positions within each tissue type
    gene_positions = {}
    for tissue_type in df['tissue_type'].unique():
        position = tissue_to_pos[tissue_type]
        for i, gene in enumerate(df[df['tissue_type'] == tissue_type]['gene'].unique()):
            gene_positions[(tissue_type, gene)] = position + (i - len(df[df['tissue_type'] == tissue_type]['gene'].unique()) / 2) * 0.2

    # Plot significance
    y_max_per_tissue = {tissue_type: df[df['tissue_type'] == tissue_type]['expression'].max() + 0.5
                    for tissue_type in tissue_types}
    y_increment = 0.55

    for i, (g1, g2) in enumerate([(gene1_name, gene2_name), 
                                  (gene1_name, gene3_name), 
                                  (gene1_name, gene4_name),
                                  (gene2_name, gene3_name), 
                                  (gene2_name, gene4_name), 
                                  (gene3_name, gene4_name)]):
        for tissue_type in tissue_types:
            if (tissue_type, g1) in gene_positions and (tissue_type, g2) in gene_positions:
                pos1 = gene_positions[(tissue_type, g1)]
                pos2 = gene_positions[(tissue_type, g2)]

                # Calculate statistical significance
                stat, pval = mannwhitneyu(
                    df[(df['gene'] == g1) & (df['tissue_type'] == tissue_type)]['expression'],
                    df[(df['gene'] == g2) & (df['tissue_type'] == tissue_type)]['expression'],
                    alternative='two-sided'
                )
                if pval < 0.05:
                    stars = '*' if pval < 0.05 else ''
                    stars = '**' if pval < 0.01 else stars
                    stars = '***' if pval < 0.001 else stars
                    add_significance(ax, pos1 - 0.1, pos2 + 0.1, y_max_per_tissue[tissue_type], stars)
                    y_max_per_tissue[tissue_type] += y_increment

    # Add custom legend
    if gene3_name and gene4_name:
        legend_labels = [gene1_name, gene2_name, gene3_name, gene4_name]
        colors = [color_palette[gene1_name], color_palette[gene2_name], color_palette.get(gene3_name, 'gray'), color_palette.get(gene4_name, 'gray')]
    elif gene3_name:
        legend_labels = [gene1_name, gene2_name, gene3_name]
        colors = [color_palette[gene1_name], color_palette[gene2_name], color_palette.get(gene3_name, 'gray')]
    else:
        legend_labels = [gene1_name, gene2_name]
        colors = [color_palette[gene1_name], color_palette[gene2_name]]

    legend_handles = [plt.Line2D([0], [0], 
        marker='o', color='w', markerfacecolor=color, markersize=5) 
        for color in colors]
    ax.legend(
            legend_handles, 
            legend_labels, 
            loc='upper right',
            frameon=True,
            handletextpad=0.1,
            labelspacing=0.2,
            borderpad=0.5
            )

    plt.tight_layout()
    plt.savefig(f"{gene1_name}_{gene2_name}{'_'+gene3_name if gene3_name else ''}{'_'+gene4_name if gene4_name else ''}_boxplot.png", dpi=400)
    plt.savefig(f"{gene1_name}_{gene2_name}{'_'+gene3_name if gene3_name else ''}{'_'+gene4_name if gene4_name else ''}_boxplot.svg", dpi=400)
    plt.close()

    time = str(datetime.now())[:-7]
    print(f"Plot saved on {time}.")

###########################################################################
# 2. argparse options

def main():
    # Set up argparse
    parser = argparse.ArgumentParser(
        prog='dge_boxwhisker_multi.py',
        formatter_class=argparse.RawTextHelpFormatter,
        description=textwrap.dedent('''\
        This script compares the distribution of RNA expression data between up to four genes,
        generating a boxplot with significance markers for pairwise comparisons.
        Example usage:
            python dge_boxwhisker_multi.py gene1.csv gene2.csv gene3.csv gene4.csv
        ''')
    )

    parser.add_argument(
        'input_files',
        metavar='input_files',
        type=str,
        nargs='+',
        help='CSV files containing RNA expression data for genes (up to four files).'
    )

    args = parser.parse_args()

    # Input data from up to four files
    if len(args.input_files) == 2:
        gene1_df, gene2_df, gene3_df, gene4_df = read_input(args.input_files[0], args.input_files[1])
    elif len(args.input_files) == 3:
        gene1_df, gene2_df, gene3_df, gene4_df = read_input(args.input_files[0], args.input_files[1], args.input_files[2])
    elif len(args.input_files) == 4:
        gene1_df, gene2_df, gene3_df, gene4_df = read_input(args.input_files[0], args.input_files[1], args.input_files[2], args.input_files[3])
    else:
        print("Please provide at least two and up to four CSV files.")
        sys.exit(1)

    # Derive gene names from file names
    gene1_name = args.input_files[0].split('/')[-1].replace('.csv', '').replace('_RSEM', '').replace('_TPM', '')
    gene2_name = args.input_files[1].split('/')[-1].replace('.csv', '').replace('_RSEM', '').replace('_TPM', '')
    gene3_name = args.input_files[2].split('/')[-1].replace('.csv', '').replace('_RSEM', '').replace('_TPM', '') if len(args.input_files) > 2 else None
    gene4_name = args.input_files[3].split('/')[-1].replace('.csv', '').replace('_RSEM', '').replace('_TPM', '') if len(args.input_files) > 3 else None

    # Plotting function
    plot(gene1_df, gene2_df, gene3_df, gene4_df, gene1_name, gene2_name, gene3_name, gene4_name)

if __name__ == "__main__":
    main()