"""
contact:    wlee9829@gmail.com
date:       2024_06_27
python:     python3.10
script:     dge_boxwhisker.py

This Python script plots the distribution of normalized gene expression 
counts downloaded from the UCSC Xena web platform.
"""
# Define version
__version__ = "2.0.0"

# Version notes
__update_notes__ = """
2.1.0
    -   Relabeling y-axis for different normalization methods.

2.0.0
    -   Edited script to exclude tissues or samples that have no normal to
        tumor comparison. Removed the following tissues:

        Adipose Tissue Normal
        Blood Normal
        Blood Vessel Normal 
        Eye Tumor
        Fallopian Tube Normal
        Heart Normal
        Lining of Body Cavities Tumor
        Lymphatic Tissue Tumor
        Muscle Normal
        Nerve Normal
        Paraganglia Tumor
        Pituitary Normal
        Salivary Gland Normal
        Small Intestine Normal
        Soft Tissue or Bone Tumor
        Spleen Normal
        Nervous System Tumor
        Thymus Tumor
        Vagina Normal

    -   Combined tissue types from same source.

        Uterus Normal + Endometrium Normal = Uterus Normal
        Uterus Tumor + Endometrium Tumor = Uterus Tumor
        Thyroid Normal + Thyroid Gland Normal = Thyroid Normal

    -   Renamed certain tissue types for consistent style.

        Cervix Uteri Normal: Cervix Normal
        Head And Neck Region Normal: Head And Neck Normal
        Head And Neck Region Tumor: Head And Neck Tumor
        Thyroid Gland Tumor: Thyroid Tumor

    -   Note on "Normal" tissues.
        TCGA matched normal - solid tissue taken from tissue adjacent to tumor
        GTEx normal - normal tissue of individuals who do not have cancer

1.4.1
    -   Rotate x-axis labels for readability and spacing.

1.4.0
    -   Added flag to isolate and generate plots based on specified tissue type 
        (-t, --tissue). Case sensitive and works with  multiple tissue types.
    -   Dynamically changing plot width for datasets.

1.3.1
    -   Added significance value annotations and background that spans the 
        tissue types in comparison. 
    -   Added logic when excluding 'Other' tissue types, to warn about
        statistical test being performed on only 'Normal' vs. 'Tumor'.

1.3.0
    -   Add the Mann-Whitney U test on columns with paired "Normal" and "Tumor"
        (calc_significance).
    -   Added flag to exclude 'Other' tissue type (-x, --exclude)
    -   Added flag to name output figure (-o, --output-prefix)
    -   Added flag for statistics printout and annotating figures (-s, --stats)

1.2.0
    -   Added boxplot and stripplot function, prints tissue_type count 
        for each of the categories. 

1.1.0
    -   Add function to skip processing input files and use CSV instead
        (-csv, --csv-file). 
    -   Added CSV output function to increse processing speed.

1.0.1
    -   Combined input file reading functions to convert sample_ids to tissue
        type by phenotype file.

1.0.0   
    -   Initial commit, set up outline of logic and functions.
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

def read_input(input_file, phenotype_file, gene_name):
    """
    Processes the input gene expression and phenotype data files to generate 
    a combined DataFrame with normalized counts and renamed columns.

    This function reads the input expression file and phenotype description 
    file, processes the data to extract and rename columns based on tissue 
    types and sample categories, and returns a structured DataFrame. It 
    supports both plain text and gzip-compressed input files.

    Args:
        input_file (str): Path to the gene expression count data file.
        phenotype_file (str): Path to the phenotype description file.
        gene_name (str): The name of the gene to extract expression data for.

    Returns:
        pd.DataFrame: A transformed DataFrame with normalized counts and 
        renamed columns based on tissue types and sample categories.
    
    The phenotype file is expected to have the following columns:
        - sample
        - detailed_category
        - primary disease or tissue
        - _primary_site
        - _sample_type
        - _gender
        - _study
    """
    # Read input expression file
    if "gz" in str(input_file):
        with gzip.open(input_file, 'rt') as file:
            count_df = pd.read_csv(file, sep='\t')
    else:
        with open(input_file, 'rt') as file:
            count_df = pd.read_csv(file, sep='\t')

    # Check DataFrame shape, search for specified gene
    if count_df.shape[0] == 1 and count_df.iloc[0, 0] == gene_name:
        print(f"Processing data for {gene_name}.")
        norm_count = count_df
    else:
        print(f"Searching data for {gene_name}.")
        norm_count = count_df[count_df.iloc[:, 0] == gene_name]

    norm_count = norm_count.transpose().reset_index()
    norm_count.columns = ['identifier', 'Expression']
    expression = norm_count[1:]

    # Read phenotype description file, process information
    with gzip.open(phenotype_file, 'rt', errors='replace') as file:
        phenotype_df = pd.read_csv(file, sep='\t', encoding='utf-8-sig')

    phenotype_df['_primary_site'] = phenotype_df['_primary_site'].str.title()

    # Merge and update identifier based on phenotype information
    merged_count = pd.merge(expression, phenotype_df, 
        left_on='identifier', right_on='sample', how='left')

    def get_sample_type(sample_type):
        if pd.isna(sample_type):
            return 'Other'
        elif 'Normal' in sample_type:
            return 'Normal'
        elif 'Tumor' in sample_type:
            return 'Tumor'
        else:
            return 'Other'

    merged_count['identifier'] = merged_count.apply(
        lambda row: 
        f"{row['_primary_site']} "
        f"{get_sample_type(row['_sample_type'])}", axis=1)

    # Select and restructure columns
    merged_count = merged_count[['identifier', 'Expression']]
    transformed_df = (merged_count
        .set_index('identifier')
        .stack()
        .groupby(level=0)
        .apply(list)
        .apply(pd.Series)
        .T)
    transformed_df.columns = [
        'Sympathetic Nervous System Other' if 'sympathetic' in col.lower() \
            and 'other' in col.lower() else 
        'Sympathetic Nervous System Tumor' if 'sympathetic' in col.lower() \
            and 'tumor' in col.lower() else 
        'Soft Tissue or Bone Normal' if 'soft tissue,bone' in col.lower() \
            and 'normal' in col.lower() else
        'Soft Tissue or Bone Tumor' if 'soft tissue,bone' in col.lower() \
            and 'tumor' in col.lower() else
        col
        for col in transformed_df.columns
        ]
    
    transformed_df = transformed_df.drop(
        columns=['nan Normal', 'nan Other'],axis=1)

    # Print the transformed DataFrame
    # print(transformed_df)
    return transformed_df

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

def plot(dataframe, gene_name, output_prefix='', 
    exclude_other=False, stats=False, tissue=None):
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
    # Set up the data.
    if tissue:
        tissue_list = [t.strip() for t in tissue.split(',')]
        filtered_tissue = [col for col in dataframe.columns
            if any(t in col for t in tissue_list)]

        # Check for tissue types that may include typo.
        missing_tissues = [t for t in tissue_list if not any(t in col 
            for col in dataframe.columns)]

        if missing_tissues:
            # Error if there are no tissues found with specified type.
            if not filtered_tissue:
                print(f"ERROR: None of the specified tissue(s) were found in"
                 f" the dataset. Check to see if it exists here:\n")
                print("\n".join(dataframe.columns))
                sys.exit()

            print(f"WARNING: The following tissues could not be located"
                f" within the dataset: ")
            print(f"\n{', '.join(missing_tissues)}")
            print(f"\nOmitting and continuing...")
            print("-" * 60)

        dataframe = dataframe[filtered_tissue]

    if exclude_other:
        dataframe = dataframe.loc[:, ~dataframe.columns.str.contains('Other')]
    else:
        if stats:
            print(f"NOTE: Statistical analysis ignores 'Other' tissue types"
                f" and compares between 'Normal' and 'Tumor'.\n")

    # Lower this number if too strict
    dataframe = dataframe.loc[:, dataframe.count() >= 5]

    df = pd.melt(dataframe, var_name='tissue_type', value_name='expression')
    filtered_df = df[df['expression'].notna()]
    counts_dict = filtered_df.groupby('tissue_type').size().to_dict()

    # Color mapping for tissue types.
    def set_color(column_name):
        if 'Normal' in column_name:
            return 'blue'
        elif 'Tumor' in column_name:
            return 'red'
        else:
            return 'gray'

    colors = [set_color(col) for col in dataframe.columns 
        if col != 'tissue_type']

    # Set figure parameters
    num_tissues = len(filtered_df['tissue_type'].unique())
    fig_width = min(num_tissues, 16)
    plt.figure(figsize=(fig_width, 8))

    # Define boxplot variables 
    ax = sns.boxplot(
            x='tissue_type', 
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
            x='tissue_type', 
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
    ax.set_ylabel(f'Gene Expression',
        fontsize=8, fontweight='bold')
    ax.set_title(f'{gene_name} Expression by Tissue Type', fontsize=12,
        fontweight='bold')

    # Add counts to x-axis labels
    x_labels = filtered_df['tissue_type'].unique()
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels([f"{label} (n={counts_dict.get(label, 0)})" 
        for label in x_labels], rotation=45, rotation_mode='anchor', 
            ha='right', fontsize=8)
    ax.set_xlim(-0.50, num_tissues + 1.5)

    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if stats:
        # Print statistics on tissue types, and number of counts for each.
        print(f"Summary of datasets:\n")
        tissue_types = filtered_df['tissue_type'].nunique()
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
                    filtered_df.loc[filtered_df['tissue_type'] == normal_label,
                        'expression'].max(),
                    filtered_df.loc[filtered_df['tissue_type'] == tumor_label, 
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

                # Calculate span of gray background
                span_start = min(normal_index, tumor_index) - 0.5
                span_end = max(normal_index, tumor_index) + 0.5

                # Add gray background behind the annotation
                ax.axvspan(
                    span_start, 
                    span_end, 
                    alpha=0.05,
                    edgecolor='black',
                    linewidth=0.5, 
                    facecolor='gray', 
                    zorder=-1)

    # Add custom legend
    if exclude_other:
        legend_labels = ['Normal', 'Tumor']
        colors = ['blue', 'red']
    else:
        legend_labels = ['Normal', 'Tumor', 'Other']
        colors = ['blue', 'red', 'gray']

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

    # Add alternating background colors for every two samples
    tissue_types = filtered_df['tissue_type'].unique()
    num_tissues = len(tissue_types)
    colors = ['lightgray', 'white']

    for i in range(0, num_tissues, 2):
        span_start = i - 0.5
        span_end = min(i + 1.5, num_tissues - 0.5)
        color = colors[(i // 2) % 2]
        ax.axvspan(
            span_start, 
            span_end, 
            alpha=0.1, 
            color=color,
            zorder=-1
        )
        
    # Save the plot as PNG
    output_file = f"{output_prefix}{gene_name}_plot"
    plt.subplots_adjust(bottom=0.1)
    plt.tight_layout()
    plt.savefig(f"{output_file}.png", dpi=400)
    plt.savefig(f"{output_file}.svg", dpi=400)
    plt.close()

    time = str(datetime.now())[:-7]
    print(f"Plot saved as {output_file} on {time}.")

def parse_args():
    """
    Main function to set up the argument parser, handle input arguments, 
    and call the relevant functions for processing and plotting gene 
    expression data.

    This function supports the following functionalities:
    - Reading gene expression data from a CSV file or input file.
    - Extracting gene expression data for a specific gene.
    - Excluding columns containing "Other" from plotting.
    - Setting a prefix for the output file names.
    - Printing statistics on tissue types and counts.

    The function parses command-line arguments, reads the input data, 
    calculates statistical significance, and generates plots for gene 
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

NOTE: Only the -csv/input_file and gene_name arguments are positional.

1a. -csv, --csv-file    Curated csv file containing the tissue types as
                        separate columns and the expression count value as row.
                        Highly recommended to increase plotting speed when a
                        gene can be extracted.

1b. input_file:         Input file (.txt or .gz) structured with tissue 
                        types as columns and expression count values as new
                        rows for every gene.

2. gene_name:           Gene to search data for. Case sensitive.

3. -o, --output-prefix  An output name preceding 'gene_name'_plot.png.

4. -x --exclude         Excludes the tissue types that are not defined as either
                        'Normal' or 'Tumor'. Optionally input a list of tissue
                        or sample types to exclude.

5. -s, --stats          Calculates statistitical difference between pairs of 
                        tissue 'Normal' vs 'Tumor' and adds significance 
                        annotations to the plots.

6. -t, --tissue         Specifies which tissue types to isolate plots for.
                        Case sensitive and must match the tissue types available
                        in the non-specific plot.

7. -V, --version        Prints version and version updates.

###########################################################################
"""),
    usage=
"""
    \npython3 %(prog)s input_file gene_name

    Usage examples: 

        %(prog)s TcgaTargetGtex_RSEM_Hugo_norm_count.gz GLP1R [-x, -s, -o]

        %(prog)s -csv=glp1r_counts.csv GLP1R -x -s -o=GLP1R_expression 
""")

    parser.add_argument(
        '-csv', '--csv-file', type=str,
        help='Optional. Path to CSV file. Reduces memory use if specified.',
        default=None)

    parser.add_argument(
        'input_file', type=str, nargs='?',
        help=('Required if CSV file (-csv, --csv-file) is not provided. Path to'
            ' gene expression count data file for processing.'))

    parser.add_argument(
        'gene_name', type=str,
        help='The name of the gene to extract expression data for.')

    parser.add_argument(
        '-o', '--output-prefix', type=str, default='',
        help='Optional. Prefix for the output file names. Defaults to None.')

    parser.add_argument(
        '-x', '--exclude', action='store_true',
        help='Optional. Exclude columns containing "Other" from plotting. '
        'Can include comma separated list of tissues/sample names to exclude')

    parser.add_argument(
        '-s', '--stats', action='store_true',
        help='Optional. Prints statistics on the tissue types and counts.')

    parser.add_argument(
        '-t', '--tissue', type=str, default='',
            help='Specify tissue types to isolate (comma-separated,' 
            ' case sensitive).')

    parser.add_argument('-V', '--version', action='version', 
        version=f'%(prog)s: {__version__}\n{__update_notes__}\n', 
        help='Prints version and update notes.')

    return parser.parse_args()

def main(args):
    """
    Main function.
    """
    csv_file = args.csv_file
    gene_name = args.gene_name
    output_prefix = args.output_prefix
    exclude_other = args.exclude
    stats = args.stats
    tissue = args.tissue

    if output_prefix:
        output_prefix = str(output_prefix+"_")
    else:
        output_prefix = ""

    if csv_file:
        combined_df = pd.read_csv(csv_file)
        print(f"Searching {csv_file} for {gene_name} data.")
        print("-" * 60)

    else:
        if not args.input_file or not args.gene_name:
            parser.error('You must provide input_file and gene_name' \
                ' if csv-file is not provided.')
            sys.exit()
        input_file = args.input_file

        if os.path.exists('TcgaTargetGTEX_phenotype.txt.gz'):
            phenotype_file = 'TcgaTargetGTEX_phenotype.txt.gz'
        else:
            phenotype_file = '*_phenotype.txt.gz'
            print(f"Attempting to use file {phenotype_file} as identifier.")
            
        count_df = read_input(input_file, phenotype_file, gene_name)
        combined_df = count_df
        print("-" * 60)

        # Data cleanup - columns without comparison.
        exclude_tissues = [
            "Adipose Tissue Normal",
            "Blood Normal",
            "Blood Vessel Normal",
            "Eye Tumor",
            "Fallopian Tube Normal",
            "Heart Normal",
            "Lining Of Body Cavities Tumor",
            "Lymphatic Tissue Tumor",
            "Muscle Normal",
            "Nerve Normal",
            "Paraganglia Tumor",
            "Pituitary Normal",
            "Salivary Gland Normal",
            "Small Intestine Normal",
            "Soft Tissue or Bone Tumor",
            "Spleen Normal",
            "Nervous System Tumor",
            "Thymus Tumor",
            "Vagina Normal"
        ]

        combined_df = combined_df.drop(
            columns=[col for col in combined_df.columns if any(
            exclude in col for exclude in exclude_tissues)]
        )

        # Data cleanup - naming conventions fix.
        combined_df.rename(columns={
            'Cervix Normal' : 'Cervix Normal 1',
            'Head And Neck Region Normal': 'Head And Neck Normal',
            'Head And Neck Region Tumor': 'Head And Neck Tumor',
            'Thyroid Gland Tumor': 'Thyroid Tumor'
        }, inplace=True)

        # Data cleanup - concatenating same tissue types 
        def concatenate_columns(df, columns_to_concat, new_column_name):
            concatenated_list = [item for col in columns_to_concat 
            for item in df[col].tolist()] 

            return pd.DataFrame({new_column_name: concatenated_list})

        uterus_endometrium_normal = concatenate_columns(
            combined_df, 
            ['Uterus Normal', 'Endometrium Normal'], 
            'Uterus Normal'
        )
        uterus_endometrium_tumor = concatenate_columns(
            combined_df, 
            ['Uterus Tumor', 'Endometrium Tumor'], 
            'Uterus Tumor'
        )
        thyroid_gland_normal = concatenate_columns(
            combined_df, 
            ['Thyroid Normal', 'Thyroid Gland Normal'], 
            'Thyroid Normal'
        )
        
        cervix_normal = concatenate_columns(
            combined_df, 
            ['Cervix Normal 1', 'Cervix Uteri Normal'], 
            'Cervix Normal'
        )

        combined_df = combined_df.drop(columns=[
            'Cervix Normal 1',
            'Cervix Uteri Normal',
            'Endometrium Normal',
            'Endometrium Tumor',
            'Thyroid Gland Normal',
            'Thyroid Normal',
            'Uterus Normal', 
            'Uterus Tumor',
        ])

        combined_df = pd.concat([
            combined_df, 
            cervix_normal,
            thyroid_gland_normal,
            uterus_endometrium_normal, 
            uterus_endometrium_tumor,
        ], axis=1)

        combined_df.to_csv(f'{output_prefix}{gene_name}.csv', index=False)

    combined_df = combined_df.sort_index(axis=1)
    plot(combined_df, gene_name, output_prefix, exclude_other, stats, tissue)

if __name__ == "__main__":
    args = parse_args()
    main(args)
    sys.exit()