"""
contact:    wlee9829@gmail.com
date:       2024_08_08
python:     python3.10
script:     violin_plots.py

This Python script plots the pathological stage data and determines the 
Kruskal-Wallis and Mann-Whitney U tests for data retrieved from the UCSC Xena web platform.
"""
# Define version
__version__ = "2.0.1"

# Version notes
__update_notes__ = """
2.0.1
    -   Removed statannotations and replaced it with manual significance annotations.
    -   Kruskal-Wallis and Mann-Whitney U tests for pairwise comparisons still included.

2.0.0
    -   Replaced ANOVA with Kruskal-Wallis test.
    -   Added Mann-Whitney U tests for pairwise comparisons.
    -   Included significance stars and brackets for comparisons.

1.0.0
    -   Initial commit, set up outline of logic and functions.
"""

from datetime import datetime
import argparse
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns

def read_input(file_path):
    """
    Read gene expression and pathologic stage data from a TSV file.

    Args:
        file_path (str): Path to the TSV file containing the data.

    Returns:
        pd.DataFrame: DataFrame containing the pathologic stage and gene expression data.
    """
    data = pd.read_csv(file_path, sep='\t')
    if 'ENSG00000112164.5' in data.columns:
        data.rename(columns={'ENSG00000112164.5': 'GLP1R'}, inplace=True)
    data['GLP1R'] = pd.to_numeric(data['GLP1R'], errors='coerce')

    return data

def violinplot(data, filename):
    # Map specific stages to general stages (I, II, III, IV)
    stage_mapping = {
        'Stage I': 'Stage I', 
        'Stage IA': 'Stage I', 
        'Stage IB': 'Stage I',
        'Stage IC': 'Stage I',
        'Stage II': 'Stage II', 
        'Stage IIA': 'Stage II', 
        'Stage IIB': 'Stage II',
        'Stage IIC': 'Stage II',
        'Stage III': 'Stage III', 
        'Stage IIIA': 'Stage III', 
        'Stage IIIB': 'Stage III',
        'Stage IIIC': 'Stage III',
        'Stage IIIC1': 'Stage III',
        'Stage IIIC2': 'Stage III',
        'Stage IV': 'Stage IV', 
        'Stage IVA': 'Stage IV', 
        'Stage IVB': 'Stage IV',
        'Stage IVC': 'Stage IV'
    }

    # Initialize variables for x_col and data_cleaned
    x_col = None
    data_cleaned = None
    stage_column = None

    if 'pathologic_stage' in data.columns:
        stage_column = 'pathologic_stage'
    elif 'clinical_stage' in data.columns:
        stage_column = 'clinical_stage'

    if stage_column:
        # Map stages to general stages
        data['general_stage'] = data[stage_column].map(stage_mapping)

        # Drop NaN values and set x_col
        x_col = 'general_stage'
        data_cleaned = data.dropna(subset=['general_stage', 'GLP1R'])

        # Define full stage order and filter out unavailable ones
        available_stages = sorted(data_cleaned['general_stage'].unique())
        full_stage_order = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
        stage_order = [stage for stage in full_stage_order if stage in available_stages]

    elif 'histological_type' in data.columns:
        # Drop NaN values and set x_col
        x_col = 'histological_type'
        data_cleaned = data.dropna(subset=['histological_type', 'GLP1R'])
        stage_order = sorted(data_cleaned[x_col].unique())

    if data_cleaned is not None:
        # Drop any remaining NaN values for the chosen x_col and 'GLP1R'
        data_cleaned = data_cleaned.dropna(subset=[x_col, 'GLP1R'])
        data_cleaned[x_col] = data_cleaned[x_col].astype(str)

        # Ensure there's at least one unique value in x_col
        if data_cleaned[x_col].nunique() < 2:
            print(f"Not enough unique values in {x_col} to perform Kruskal-Wallis test.")
            return

        unique_stages = sorted(data_cleaned[x_col].unique())
        grouped_data_simplified = [data_cleaned[data_cleaned[x_col] == stage]['GLP1R'] for stage in unique_stages]
        
        # Perform Kruskal-Wallis test
        kw_stat, p_value_kw = stats.kruskal(*grouped_data_simplified)

        # Dynamically create pairwise comparisons based on the available stages
        predefined_comparisons = list(itertools.combinations(range(len(unique_stages)), 2))
        pairwise_results = []

        # Perform pairwise Mann-Whitney U tests for predefined comparisons
        for idx1, idx2 in predefined_comparisons:
            group1_data = data_cleaned[data_cleaned[x_col] == unique_stages[idx1]]['GLP1R']
            group2_data = data_cleaned[data_cleaned[x_col] == unique_stages[idx2]]['GLP1R']
            u_stat, p_value_mw = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
            pairwise_results.append((unique_stages[idx1], unique_stages[idx2], p_value_mw))

        # Apply Bonferroni correction
        num_comparisons = len(pairwise_results)
        alpha = 0.05
        alpha_corrected = alpha / num_comparisons

        # Create a violin plot with the simplified stages
        plt.figure(figsize=(4, 4))

        sns.violinplot(
            x=x_col, 
            y='GLP1R', 
            data=data_cleaned, 
            color='red', 
            inner='box', 
            linewidth=0.5,
            order=stage_order
        )

        # Overlay IQR and median manually
        for stage in stage_order:
            stage_data = data[data[x_col] == stage]['GLP1R'].dropna()
            q1 = stage_data.quantile(0.25)
            median = stage_data.median()
            q3 = stage_data.quantile(0.75)
            iqr = q3 - q1

            # Add IQR line
            plt.plot([stage_order.index(stage)] * 2, [q1, q3], color='black', lw=2.5)
            plt.plot([stage_order.index(stage)] * 2, [q1 - 1.5 * iqr, q3 + 1.5 * iqr], color='black', lw=1)

            # Add median as a white dot
            plt.scatter(stage_order.index(stage), median, color='white', s=10, zorder=3, edgecolor=None)

        # Annotate the plot with the Kruskal-Wallis result
        plt.text(0.95, 0.95, f'H-Statistic: {kw_stat:.2f}\np-value: {p_value_kw:.3e}',
                 horizontalalignment='right', 
                 verticalalignment='top', 
                 transform=plt.gca().transAxes, 
                 fontsize=8, 
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))

        def significance_stars(p):
            """
            Return the significance stars based on p-value.
            """
            return '***' if p <= 0.001 else '**' if p <= 0.01 else '*' if p <= 0.05 else ''

        # Add significance brackets manually f dynamic pairwise comparisons
        y_offset_increment = 0.25
        current_y_max = 0

        for (group1, group2, p_value_mw) in pairwise_results:
            # Only plot if the p-value is significant (≤ alpha_corrected)
            if p_value_mw <= alpha_corrected:
                # Get the max y-value for both groups and add an offset
                y_max = max(data_cleaned[data_cleaned[x_col] == group1]['GLP1R'].max(), 
                            data_cleaned[data_cleaned[x_col] == group2]['GLP1R'].max()) + 0.2
                # Stagger the brackets vertically
                current_y_max = y_max + (pairwise_results.index((group1, group2, p_value_mw)) * y_offset_increment)

                x1, x2 = stage_order.index(group1), stage_order.index(group2)

                # Plot the bracket
                plt.plot([x1, x1, x2, x2], [current_y_max, current_y_max + 0.1, current_y_max + 0.1, current_y_max], 
                         lw=0.75, color='black')

                # Add the significance stars
                plt.text((x1 + x2) * 0.5, current_y_max + 0.15, significance_stars(p_value_mw), 
                         ha='center', va='bottom', fontsize=8)

        # Set plot title and labels
        plt.xticks(fontsize=8)
        plt.xlabel(None)
        plt.ylabel(f'GLP1R Expression log\u2082(norm_count+1)', fontsize=8)
        plt.ylim(0, None)

        # Show the plot
        plt.savefig(f"{filename}.png", dpi=400)
        plt.savefig(f"{filename}.svg", dpi=400)
        plt.close()

        time = str(datetime.now())[:-7]
        print(f"Plot saved on {time}.")

def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="violin_plots.py",
        description="Violin plots based on expression and pathologic stage with Kruskal-Wallis and Mann-Whitney tests.")
    parser.add_argument(
        'file_path', type=str,
        help='Path to the TSV file containing pathologic stage and gene expression.')
    
    return parser.parse_args()

def main(args):
    """
    Main function to execute the script.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    filename = (args.file_path).replace(".tsv", "")
    data = read_input(args.file_path)
    violinplot(data, filename)

if __name__ == "__main__":
    args = parse_args()
    main(args)


#!/bin/bash

# Loop through all .tsv files in the current directory.
for file in *.tsv; do
    echo "Processing $file"
    python3 violin_plots.py "$file"
done