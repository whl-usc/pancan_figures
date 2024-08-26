"""
contact:    wlee9829@gmail.com
date:       2024_08_08
python:     python3.10
script:     violin_plots.py

This Python script plots the pathological stage data and determines the 
F-value and associated statistics for data retrieved from the UCSC Xena web platform.
"""
# Define version
__version__ = "1.0.0"

# Version notes
__update_notes__ = """
1.0.0
    -   Initial commit, set up outline of logic and functions.
"""

from datetime import datetime
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns

def read_input(file_path):
    """
    Read gene expression and expression data from a TSV file.

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
            print(f"Not enough unique values in {x_col} to perform ANOVA.")
            return

        unique_stages = sorted(data_cleaned[x_col].unique())
        grouped_data_simplified = [data_cleaned[data_cleaned[x_col] == stage]['GLP1R'] for stage in unique_stages]
        f_value_simplified, p_value_simplified = stats.f_oneway(*grouped_data_simplified)

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

    # Annotate the plot with the F-value and p-value
    plt.text(0.95, 0.95, f'F-value: {f_value_simplified:.2f}\np-value: {p_value_simplified:.3e}',
             horizontalalignment='right', 
             verticalalignment='top', 
             transform=plt.gca().transAxes, 
             fontsize=8, 
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.75))

    # Set plot title and labels
    plt.xticks(fontsize=8)
    plt.xlabel(None)
    plt.ylabel(f'GLP1R Expression log\u2082(norm_count+1)',
        fontsize=8)
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
        description="Violon plots based on expression and pathologic stage.")
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
