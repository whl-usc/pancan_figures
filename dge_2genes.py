"""
contact:    wlee9829@gmail.com
date:       2024_08_06
python:     python3.10
script:     dge_2genes.py

This Python script compares the distribution of normalized data
between two genes by plotting.
"""

# Define version
__version__ = "1.1.0"

# Version notes
__update_notes__ = """
1.1.0
    -   Edited for docstrings, styling of plots.

1.0.0
    -   Initial commit, set up outline of logic and functions.
"""

# Import packages
from datetime import datetime
from scipy import stats
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
import textwrap

###########################################################################
# 1. Set up functions

def read_input(file_path1, file_path2, placeholder_value=-9999):
    """
    Read the data for RNA expression counts for two specified genes from
    separate CSV files.

    Args:
        file_path1 (str): CSV file containing expression data for first gene.
        file_path2 (str): CSV file containing expression data for second gene.
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

    return filter_tumor_columns(data1), filter_tumor_columns(data2)

def calc_significance(df, gene1_name, gene2_name, placeholder_value=-9999):
    """
    Calculate p-values and significance levels between two genes.
    
    Args:
        df (pd.DataFrame): DataFrame containing gene expression data.
        gene1_name (str): Name of the first gene.
        gene2_name (str): Name of the second gene.
        placeholder_value (float): Placeholder value for missing data.

    Returns:
        tuple: Dictionaries of p-values and significance levels by tissue type.
    """
    p_values, significance_levels = {}, {}    
    tissue_types = df['tissue_type'].unique()

    for tissue in tissue_types:
        gene1_expr = (df[(df['tissue_type'] == tissue) & 
            (df['gene'] == gene1_name)]['expression'])
        gene2_expr = (df[(df['tissue_type'] == tissue) & 
            (df['gene'] == gene2_name)]['expression'])
        
        gene1_expr = gene1_expr[gene1_expr != placeholder_value]
        gene2_expr = gene2_expr[gene2_expr != placeholder_value]
        
        if len(gene1_expr) > 1 and len(gene2_expr) > 1:
            _, p_value = stats.mannwhitneyu(gene1_expr, gene2_expr, 
                alternative='two-sided')
            p_values[tissue] = p_value
            if p_value < 0.001:
                significance_levels[tissue] = '***'
            elif p_value < 0.01:
                significance_levels[tissue] = '**'
            elif p_value < 0.05:
                significance_levels[tissue] = '*'
            else:
                significance_levels[tissue] = ''
        else:
            significance_levels[tissue] = ''
    
    return p_values, significance_levels

def plot(dataframe1, dataframe2, gene1_name, gene2_name, output_prefix='',
    placeholder_value=-9999):
    """
    Generate and save plots comparing gene expression data.
    
    Args:
        dataframe1 (pd.DataFrame): Data for the first gene.
        dataframe2 (pd.DataFrame): Data for the second gene.
        gene1_name (str): Name of the first gene.
        gene2_name (str): Name of the second gene.
        output_prefix (str): Prefix for output file names.
        placeholder_value (float): Placeholder value for missing data.
    """
    dataframe1.columns = [f'{gene1_name}_{col}' for col in dataframe1.columns]
    dataframe2.columns = [f'{gene2_name}_{col}' for col in dataframe2.columns]
    combined_df = pd.concat([dataframe1, dataframe2], axis=1)
    
    df = pd.melt(combined_df, var_name='tissue_type', value_name='expression')
    df['gene'] = df['tissue_type'].apply(lambda x: x.split('_')[0])
    df['tissue_type'] = df['tissue_type'].apply(lambda x: x.split('_', 1)[1])
    df = df[df['expression'] != placeholder_value]

    gene_palette = {gene: 'blue' if gene == gene1_name else 'red' 
        for gene in df['gene'].unique()}
    
    plt.figure(figsize=(16, 10))

    ax = sns.boxplot(
        x='tissue_type', y='expression', data=df, hue='gene', 
            palette=gene_palette,
        whis=[0, 100], linewidth=2, fliersize=0.5, showcaps=True,
        boxprops={'facecolor': 'none', 'edgecolor': 'black', 'linewidth': 0.75},
        whiskerprops={'color': 'black', 'linewidth': 0.75},
        medianprops={'color': 'black', 'linewidth': 0.75},
        capprops={'color': 'gray', 'linewidth': 0.5}
    )
    sns.stripplot(
        x='tissue_type', y='expression', data=df, hue='gene', 
            palette=gene_palette,
        jitter=True, edgecolor='black', size=5, alpha=0.25, ax=ax, dodge=True
    )

    ax.set_xlabel('', fontsize=8, fontweight='bold')
    ax.set_ylabel(f'Gene Expression', 
        fontsize=8, fontweight='bold')
    ax.set_title(f'{gene1_name} and {gene2_name} Expression by Tissue Type',
        fontsize=12, fontweight='bold')

    tissue_types = df['tissue_type'].unique()

    x_labels = [
        f"{tissue_type} ({gene1_name}: n={df[(df['tissue_type'] == tissue_type) & (df['gene'] == gene1_name)].shape[0]}, {gene2_name}: n={df[(df['tissue_type'] == tissue_type) & (df['gene'] == gene2_name)].shape[0]})"
        for tissue_type in tissue_types
    ]

    ax.set_xticklabels(x_labels, rotation=45, rotation_mode='anchor', 
        ha='right', fontsize=8)
    ax.set_xlim(-0.75, len(tissue_types) + 1.25)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    print(f"Summary of datasets:\n")
    unique_tissue_types = df['tissue_type'].unique()
    print(f"\t\t     Unique tissue types : {len(unique_tissue_types)}")
    total_count = df.shape[0]
    print(f"\t\tTotal number of data points : {total_count}\n")
    for tissue_type in unique_tissue_types:
        count = df[df['tissue_type'] == tissue_type].shape[0]
        print(f"{tissue_type.rjust(40)} : {count}")
    print("-" * 60)

    p_values, significance_levels = calc_significance(df, gene1_name, 
        gene2_name, placeholder_value)
    significance_annotations = {
        '***': {'color': 'black', 'alpha': 1.0},
        '**': {'color': 'black', 'alpha': 1.0},
        '*': {'color': 'black', 'alpha': 1.0},
        '': {'color': 'black', 'alpha': 0.0}
    }

    for tissue_type, significance in significance_levels.items():
        if significance:
            tissue_index = list(tissue_types).index(tissue_type)
            y_coord = df[df['tissue_type'] == tissue_type]['expression'].max()
            ax.annotate(
                significance, (tissue_index, y_coord), xytext=(0, 10),
                    textcoords='offset points',
                ha='center', va='center', 
                    color=significance_annotations[significance]['color'],
                fontsize=8, fontweight='bold', backgroundcolor='none', 
                    alpha=significance_annotations[significance]['alpha']
            )
            ax.axvspan(
                tissue_index - 0.5, tissue_index + 0.5, alpha=0.1, 
                edgecolor='black',
                linewidth=0.5, facecolor='gray', zorder=-1
            )

    legend_labels = [f'{gene1_name}', f'{gene2_name}']
    legend_handles = [plt.Line2D([0], [0], 
        marker='o', 
        color='w',
        markerfacecolor=color,
        markersize=5) 
        for color in ['blue', 'red']]

    ax.legend(
        legend_handles, 
        legend_labels, 
        loc='upper right', 
        fontsize=8, 
        frameon=True, 
        handletextpad=0.1,
        labelspacing=0.2,
        borderpad=0.5
    )

    for i in range(len(tissue_types)):
        if i % 2 ==0:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.05, 
                color='gray', zorder=-1)

    output_file = f"{output_prefix}{gene1_name}_{gene2_name}_plot"
    plt.tight_layout(pad=5.0)
    plt.savefig(f"{output_file}.png", dpi=400)
    plt.savefig(f"{output_file}.svg", dpi=400)
    plt.close()

    print(f"Plot saved as {output_file} on {datetime.now():%Y-%m-%d %H:%M:%S}.")

def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="%(prog)s",
        formatter_class=argparse.RawTextHelpFormatter,
        description=textwrap.dedent("""\
###########################################################################
NOTE: The input strings are positional.

1. file_path1:      Input file (.csv) with tissue types as columns and 
                    expression counts as rows for every gene.
2. file_path2:      See above.
3. -o, --output-prefix  Prefix for the output file name.
###########################################################################
"""),
        usage="\npython3 %(prog)s file_path1 file_path2 [-o output_prefix]"
    )
    parser.add_argument('file_path1', type=str, 
        help='Path to the CSV file for the first gene.')
    parser.add_argument('file_path2', type=str, 
        help='Path to the CSV file for the second gene.')
    parser.add_argument('-o', '--output', type=str, 
        help='Prefix for output plot files.')
    return parser.parse_args()

def main(args):
    """
    Main function to execute the script.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    gene1_file = args.file_path1
    gene2_file = args.file_path2
    gene1 = gene1_file.replace(".csv", "").replace("_TPM", "").replace("_RSEM", "")
    gene2 = gene2_file.replace(".csv", "").replace("_TPM", "").replace("_RSEM", "")

    tumor_data1, tumor_data2 = read_input(gene1_file, gene2_file)
    plot(tumor_data1, tumor_data2, gene1, gene2, 
        output_prefix=args.output or '')

if __name__ == "__main__":
    args = parse_args()
    main(args)
    sys.exit()
