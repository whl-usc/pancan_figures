"""
contact:    wlee9829@gmail.com
date:       2024_10_02
python:     python3.10
script:     multigene_boxwhisker.py

This Python script plots the distribution of normalized gene expression 
counts downloaded from the UCSC Xena web platform.
"""
# Define version
__version__ = "2.0.0"

# Verison notes
__update_notes__ = """
2.0.0
    -   Automatically calculates and prints the statistics, removes 'other' tissues
        from comparison.

1.0.0
    -   Initial commit, set up ouline of logic and function.
"""

# Import Packages
import argparse
from datetime import datetime
import gzip
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.stats import mannwhitneyu
import seaborn as sns
import sys
import textwrap

###########################################################################
# 1. Set up functions

def calc_significance(dataframe):
    """
    Perform Mann-Whitney U tests between columns with matching names 
    when stripped of "Normal" or "Tumor". Prints the p-value for each matched 
    pair and summarizes significance based on the p-values.

    Args:
        dataframe (pd.DataFrame): A DataFrame containing gene expression data.

    Returns:
        dict: A dictionary containing p-values.
    """
    p_values = {}
    significance_levels = {}

    # Iterate over unique base names (stripped of "Normal" and "Tumor")
    for base_name in dataframe.columns.str.replace('Normal|Tumor', '', 
        regex=True).unique():
        normal_col = f"{base_name}Normal"
        tumor_col = f"{base_name}Tumor"
        
        # Check if both Normal and Tumor columns exist
        if normal_col in dataframe.columns and tumor_col in dataframe.columns:
            normal_values = dataframe[normal_col].dropna()
            tumor_values = dataframe[tumor_col].dropna()
            
            if len(normal_values) > 0 and len(tumor_values) > 0:
                stat, p_value = mannwhitneyu(normal_values, tumor_values,
                    alternative='two-sided')
                
                # Define significance level
                if p_value < 0.001:
                    significance = '***'
                elif p_value < 0.01:
                    significance = '**'
                elif p_value < 0.05:
                    significance = '*'
                else:
                    significance = ''

                p_values[base_name] = p_value
                significance_levels[base_name] = significance

    # Summarize significance findings if any comparisons were made
    if p_values:
        sorted_p_values = sorted(p_values.items(), key=lambda x: x[1])
        print(f"Summary of significance calculations:\n")
        for base_name, p_value in sorted_p_values:
            significance = significance_levels[base_name]
            print(f"{base_name.rjust(40)}: (p = {p_value:.2e}) "
                f"{significance} ")
        print("-" * 60)

    return p_values, significance_levels

def plot(dataframe, output_prefix=''):
    """
    Generates and saves a plots of the log2 transformed normalized_counts,
    separated by phenotype (tissue types). This function creates a boxplot
    and a strip plot to visualize the expression levels of a specified gene 
    across different tissue types. It includes options to exclude columns with
    'Other' in their names. The plot is saved as a PNG file.

    Args:
        dataframe (pd.DataFrame): A DataFrame containing gene expression data.
        gene_name (str): The name of the gene to be plotted.
        output_prefix (str, optional): A prefix to be added to the output file 
            name. Defaults to ''.
        exclude_other (bool, optional): If True, exclude columns containing 
            'Other' from the plot. Defaults to False.
        stats (bool, optional): If True, includes statistics on the columns that
            share a common name by varying condition.
        tissue (str or None, optional): Comma-separated list of tissue types 
            (case sensitive) to include. Defaults to None (all tissues).

    Returns:
        None. The plot is saved as a PNG file.
    """
    dataframe = dataframe.loc[:, ~dataframe.columns.str.contains('Other')]
    dataframe = dataframe.loc[:, dataframe.count() >= 5]
    filtered_df = dataframe.melt(var_name='gene_tissue', value_name='expression')
    filtered_df['tissue_type'] = filtered_df['gene_tissue'].apply(lambda x: 'Normal' if 'Normal' in x else 'Tumor')
    filtered_df.dropna(subset=['expression'], inplace=True)

    # Ensure counts_dict is defined before use
    counts_dict = filtered_df['gene_tissue'].value_counts().to_dict()

    # Set up the plot
    num_tissues = len(filtered_df['gene_tissue'].unique())
    fig_width = min(num_tissues, 16)
    plt.figure(figsize=(fig_width, 8))

    # Define color palette
    colors = {'Normal': 'blue', 'Tumor': 'red'}
    
    # Define boxplot variables 
    ax = sns.boxplot(
            x='gene_tissue', 
            y='expression', 
            data=filtered_df, 
            hue='tissue_type', 
            palette=colors, 
            whis=[0, 100],
            linewidth=2, 
            fliersize=0.5, 
            showcaps=True, 
            boxprops={'facecolor':'none', 
                'edgecolor':'black', 
                'linewidth': 0.75}, 
            whiskerprops={'color':'black', 'linewidth': 0.75}, 
            medianprops={'color':'black', 'linewidth': 0.75}, 
            capprops={'color':'gray', 'linewidth': 0.5}
        )

    # Define stripplot variables
    sns.stripplot(  
            x='gene_tissue', 
            y='expression',
            data=filtered_df, 
            hue='tissue_type', 
            palette=colors, 
            jitter=True, 
            edgecolor='black', 
            size=5, 
            alpha=0.25,
            ax=ax
        )

    # Plot labels and title for figure
    ax.set_xlabel('', fontsize=8, fontweight='bold')
    ax.set_ylabel(f'Gene Expression (log2(norm_count+1))',
        fontsize=8, fontweight='bold')

    # Add counts to x-axis labels
    x_labels = filtered_df['gene_tissue'].unique()
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels([f"{label}" 
        for label in x_labels], rotation=45, rotation_mode='anchor', 
            ha='right', fontsize=8)
    ax.set_xlim(-0.50, num_tissues)

    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Print statistics on tissue types, and number of counts for each.
    print(f"Summary of datasets:\n")
    tissue_types = filtered_df['gene_tissue'].nunique()
    print(f"\t\t     Unique sample types : {tissue_types}")

    total_count = filtered_df.shape[0]
    print(f"\t\tTotal number of datasets : {total_count}\n")
    for tissue_type, count in counts_dict.items():
        print(f"{tissue_type.rjust(40)} : {count}")
    print("-" * 60)

    # Add significance annotations above the boxplot groups
    p_values, significance_levels = calc_significance(dataframe)
    significance_annotations = {
        '***': {'color': 'black', 'alpha': 1.0},
        '**': {'color': 'black', 'alpha': 1.0},
        '*': {'color': 'black', 'alpha': 1.0},
        '': {'color': 'black', 'alpha': 0.0}
    }

    # Calculate mid-points for annotation
    for base_name, significance in significance_levels.items():
        normal_label = f"{base_name}Normal"
        tumor_label = f"{base_name}Tumor"
        if normal_label in x_labels and tumor_label in x_labels:
            normal_index = x_labels.tolist().index(normal_label)
            tumor_index = x_labels.tolist().index(tumor_label)
            mid_index = (normal_index + tumor_index) / 2
            y_coord = max(
                filtered_df.loc[filtered_df['gene_tissue'] == normal_label,
                    'expression'].max(),
                filtered_df.loc[filtered_df['gene_tissue'] == tumor_label, 
                    'expression'].max()
            )

            ax.annotate(
                significance,
                (mid_index, y_coord),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                va='center',
                color=significance_annotations[significance]['color'],
                fontsize=8,
                fontweight='bold',
                backgroundcolor='none',
                alpha=significance_annotations[significance]['alpha']
            )
            
    # Add alternating background colors for every two samples
    tissue_types = filtered_df['gene_tissue'].unique()
    num_tissues = len(tissue_types)
    colors = ['lightgray', 'white']

    for i in range(0, num_tissues, 2):
        start_idx = i - 0.5
        end_idx = min(i + 1.5, num_tissues - 0.5)
        color = colors[(i // 2) % 2]
        ax.axvspan(
            start_idx, 
            end_idx, 
            alpha=0.1, 
            color=color,
            zorder=-1
        )

    # Add custom legend
    legend_labels = ['Normal', 'Tumor']
    colors = ['blue', 'red']
    legend_handles = [plt.Line2D([0], [0], 
        marker='o', color='w', markerfacecolor=color, markersize=5) 
        for color in colors]
    legend = ax.legend(
        legend_handles, 
        legend_labels, 
        loc='upper right',
        frameon=True,
        handletextpad=0.1,
        labelspacing=0.2,
        borderpad=0.5
        )

    # Save the plot as PNG
    output_file = f"{output_prefix}_plot"
    plt.subplots_adjust(bottom=0.1)
    plt.tight_layout()
    plt.savefig(f"{output_file}.png", dpi=400)
    # plt.savefig(f"{output_file}.svg", dpi=400)
    plt.close()

    time = str(datetime.now())[:-7]
    print(f"Plot saved as {output_file} on {time}.")

###########################################################################
# 2. argparse options
def parse_args():
    """
    Main function to set up the argument parser, handle input arguments, 
    and call the relevant functions for processing and plotting gene 
    expression data.


   Args:
        -csv, --csv-file (str, optional): Path to a CSV file to read data from.
        input_file (str, optional): Path to the gene expression count data file.
        gene_name (str): The name of the gene to extract expression data for.
        -o, --output-prefix (str, optional): Prefix for the output file names.
        -x, --exclude (bool, optional): Flag to exclude columns containing 
        "Other" from plotting.
        -s, --stats (bool, optional): Flag to print statistics on tissue types 
        and counts.

    Raises:
        argparse.ArgumentError: If the required arguments are not provided 
        when --csv-file is not used.
    """
    parser = argparse.ArgumentParser(
    prog="dge_plot.py",
    formatter_class=argparse.RawTextHelpFormatter,
    description=textwrap.dedent("""\
###########################################################################
Generate box-whisker plots for gene expression data.

1. input:               Curated csv file containing the tissue types as
                        separate columns and the expression count value as row.
                        Highly recommended to increase plotting speed when a
                        gene can be extracted.

3. -o, --output         Prefix for output files.

6. -t, --tissue         Specifies which tissue types to isolate plots for.
                        Case sensitive and must match the tissue types available
                        in the non-specific plot.

7. -V, --version        Prints version and version updates.

###########################################################################
"""),
    usage=
"""    \npython3 %(prog)s input_file gene_name

    Usage examples: 

        %(prog)s GAPDH.csv GAPDH -x -s -o=GAPDH_expression 
""")
    parser.add_argument(
        'input', type=str, nargs='?',
        help=('Required CSV file path that contains'
            ' gene expression count data file for processing.'))

    parser.add_argument(
        '-o', '--output-prefix', type=str, default='expression',
        help='Optional. Prefix for the output file names. Defaults to None.')

    parser.add_argument(
        '-t', '--tissue', type=str, required=True,
            help='Specify a tissue types to isolate (case sensitive).')

    parser.add_argument('-V', '--version', action='version', 
        version=f'%(prog)s: {__version__}\n{__update_notes__}\n', 
        help='Prints version and update notes.')

    return parser.parse_args()

def main(args):
    """
    Main function.
    """
    input_files = args.input.split(',')
    output_prefix = args.output_prefix
    tissue = args.tissue.strip()

    combined_df = pd.DataFrame()

    for file in input_files:
        df = pd.read_csv(file)
        print(f"Processing {file} for expression data.")
        print("-" * 60)

        base_name = os.path.splitext(os.path.basename(file))[0]
        df.columns = [f"{base_name} {col}" for col in df.columns]

        filtered_tissue = [col for col in df.columns if tissue in col]
        if not filtered_tissue:
            print(f"No columns found for tissue: '{tissue}' in file.")
            continue

        filtered_df = df[filtered_tissue]
        combined_df = pd.concat([combined_df, filtered_df], axis=0, ignore_index=True)

    p_values, significance_levels = calc_significance(combined_df)            
    plot(combined_df, output_prefix)

if __name__ == "__main__":
    args = parse_args()
    main(args)
    sys.exit()

"""
Tumor: High
ACSL4.csv,AIFM2.csv,ALOX15.csv,FTH1.csv,GCLC.csv,GPX1.csv,GPX4.csv,SLC7A11.csv

GLS2.csv,LPCAT3.csv,NFE2L2.csv,NFKB1.csv,PRKAA1.csv,PTGS2.csv,
"""