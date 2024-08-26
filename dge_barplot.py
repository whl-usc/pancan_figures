"""
contact:    wlee9829@gmail.com
date:       2024_08_02
python:     python3.10
script:     dge_barplot.py

This Python script summarizes fold change significance of normalized gene
expression counts downloaded from the UCSC Xena web platform.
"""

# Define version
__version__ = "1.3.0"

# Version notes
__update_notes__ = """
1.3.0
    -   Updated the data comprehension to include adjusting for normalization.
    
1.2.1
    -   Statistical comparisons switched to a Mann-Whitney U Test.
    -   Removed the background coloring for easier viewing.

1.2.0
    -   Broke sections into functions.
    -   Added argument parsing for output file names.

1.1.0 
    -   Removed unnecessary comments.
    -   Replaced hard-coded paths with argparse logic.

1.0.0
    -   Initial commit; set up function logic and styling.
"""

# Import Packages
from datetime import datetime
from matplotlib.colors import Normalize
from scipy.stats import mannwhitneyu
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import textwrap

###########################################################################
# 1. Set up functions

def read_input(file_path):
    """
    Read gene expression data from a CSV file and check for columns with
    '0' values exceeding 50% of total counts.

    Args:
        file_path (str): Path to the CSV file containing gene expression data.

    Returns:
        pd.DataFrame: DataFrame containing the gene expression data.
    """
    data = pd.read_csv(file_path)
    
    # Count the number of zeros for each column
    counts_per_column = data.notna().sum()
    zero_counts = data.apply(lambda col: (col == 0).sum())
    zero_proportions = (zero_counts / counts_per_column) * 100
    high_zero = zero_proportions[zero_proportions > 50.0]
    
    if not high_zero.empty:
        zero_counts_df = pd.DataFrame(high_zero, 
            columns=['Zero Proportion (%)'])
        zero_counts_df['Zero Proportion (%)'] = (
            zero_counts_df['Zero Proportion (%)'].map('{:.2f}'.format))
        print(f"Warning: These tissue types have '0' values exceeding 50% of"
            "total counts:")
        print(zero_counts_df)

    return data

def data_cleanup(dataframe, tissue_types, norm_source):
    """
    Clean and process the gene expression data to calculate fold change and
    significance for each tissue type.

    Args:
        dataframe (pd.DataFrame): DataFrame containing the gene expression data.
        tissue_types (list of str): List of tissue types to process.
        norm_source (str): The normalization method used ('RSEM' or 'TPM').

    Returns:
        tuple: A tuple containing the DataFrame with fold change, significance,
               colormap, and normalization instance for fold change.
    """
    relative_expression = []

    for tissue_type in tissue_types:
        # Extract normal and tumor values
        normal_values = dataframe[f'{tissue_type} Normal']
        tumor_values = dataframe[f'{tissue_type} Tumor']

        # Combined dataframe columns
        combined_data = pd.DataFrame({'Normal': normal_values, 
            'Tumor': tumor_values})

        normal_cleaned = combined_data['Normal'].dropna()
        tumor_cleaned = combined_data['Tumor'].dropna()

        # Convert log2 values back to original scale
        if norm_source == "TPM":
            normal_original = (2**normal_cleaned - 0.001)
            tumor_original = (2**tumor_cleaned - 0.001)
        elif norm_source == "RSEM":
            normal_original = (2**normal_cleaned - 1)
            tumor_original = (2**tumor_cleaned - 1)
        else:
            raise ValueError(f"Unknown normalization method: {norm_source}")

        # Calculate means and log2 fold change
        mean_normal = np.mean(normal_original)
        mean_tumor = np.mean(tumor_original)

        if mean_normal == 0 or mean_tumor == 0:
            fold_change_log2 = np.nan
            p_value = 1.0
        else:
            fold_change_log2 = np.log2(mean_tumor / mean_normal)

            # Perform Mann-Whitney U test
            _, p_value = mannwhitneyu(normal_original, tumor_original, 
                alternative='two-sided')

        # Calculate -log10(p-value)
        neg_log_p_value = -np.log10(p_value) if p_value <= 0.05 else 0
  
        # Determine significance stars
        if p_value <= 0.001:
            significance = '***'
        elif p_value <= 0.01:
            significance = '**'
        elif p_value <= 0.05:
            significance = '*'
        else: 
            significance = ''

        # Append results
        relative_expression.append([tissue_type, fold_change_log2,
            neg_log_p_value, significance])

    # Convert to DataFrame for plotting
    expression_data = pd.DataFrame(relative_expression, columns=['Tissue Type',
        'Fold Change (log2)', 'Significance', 'Stars'])
    expression_data = expression_data.set_index('Tissue Type')

    # Ensure x-axis values are categorical
    expression_data.reset_index(inplace=True)
    expression_data['Tissue Type'] = pd.Categorical(
        expression_data['Tissue Type'], categories=tissue_types)

    # Define color map and norm for fold change
    cmap = plt.get_cmap('coolwarm')
    norm = Normalize(vmin=-5, vmax=5, clip=True)
    
    return expression_data, cmap, norm

def plot(expression_data, cmap, norm, output_file):
    """
    Generate and save a bar plot summarizing fold change and significance.

    Args:
        expression_data (pd.DataFrame): DataFrame containing fold change,
            significance data.
        cmap (matplotlib.colors.Colormap): Colormap for the plot.
        norm (matplotlib.colors.Normalize): Normalization instance for the plot.
        output_file (str): Base name for saving the plot images.
    """
    # Set up figure and axes
    fig, ax = plt.subplots(figsize=(12, 3))

    # Plot small boxes with color representing fold change
    sns.scatterplot(
        x='Tissue Type',
        y=0,
        hue='Fold Change (log2)',
        palette=cmap,
        data=expression_data,
        s=600,
        marker='s',
        edgecolor=None,
        legend=None,
        ax=ax
    )

    # Add color bar for fold change
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, 
        ax=ax, 
        orientation='vertical', 
        fraction=0.005, 
        pad=0)
    cbar.set_label('Fold Change (log2)')

    # Annotate significance stars above each marker
    for index, row in expression_data.iterrows():
        ax.text(
            index, 
            0.05, 
            row['Stars'], 
            ha='center', 
            va='bottom', 
            fontsize=12, 
            color='black')

    # Plot labels and title for figure
    ax.set_xlabel('Tissue Type')
    ax.set_ylabel('-log$_{10}$(p)')
    ax.set_ylim(-0.2, 0.2)
    ax.set_title('Tumor vs. Normal (Fold Change)', 
        fontsize=10, fontweight='bold')

    # Set x-tick labels with counts
    x_labels = expression_data['Tissue Type']
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, rotation_mode='anchor',
        ha='right', fontsize=8)

    # Remove y-axis ticks and spines
    ax.yaxis.set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plt.tight_layout()

    # Save and show plot
    plt.savefig(f"{output_file}.png", dpi=400)
    plt.savefig(f"{output_file}.svg", dpi=400)

    plt.show()

    time = str(datetime.now())[:-7]
    print(f"Plot saved as {output_file} on {time}.")

def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="dge_barplot.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description=textwrap.dedent("""\
###########################################################################
NOTE: The input and output strings are positional.

1. input_file:      Input file (.csv) structured with tissue types as columns
                    and expression count values as new rows for every gene.

2. -n, --normalization  Normalization method used on the raw data. Required. 
                         Accepts "RSEM" or "TPM".

3. -o, --output     An output file name, defaults to "significance.png".
###########################################################################
"""),
        usage="\npython3 %(prog)s input_file -normalization -output"
    )
    parser.add_argument(
        'file_path', type=str,
        help='Path to the CSV file containing gene expression data.')
    parser.add_argument(
        '-n', '--normalization', required=True,
        help='Normalization method used on the raw data. "RSEM" or "TPM".')
    parser.add_argument(
        '-o', '--output', type=str, 
        default='significance.png',
        help='Filename for saving the plot (default: "significance.png").')

    return parser.parse_args()

def main(args):
    """
    Main function to execute the script.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    tissue_types = [
        'Adrenal Gland', 'Bile Duct', 'Bladder', 'Brain', 'Breast',
        'Cervix', 'Colon', 'Esophagus', 'Head And Neck', 'Kidney',
        'Liver', 'Lung', 'Ovary', 'Pancreas', 'Prostate', 'Rectum',
        'Skin', 'Stomach', 'Testis', 'Thyroid', 'Uterus'
    ]

    input_file = args.file_path
    normalization = args.normalization
    output = args.output

    data = read_input(input_file)
    expression_data, cmap, norm = data_cleanup(data, tissue_types, 
        normalization)
    plot(expression_data, cmap, norm, output)

if __name__ == "__main__":
    args = parse_args()
    main(args)