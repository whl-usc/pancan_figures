"""
contact:    wlee9829@gmail.com
date:       2024_08_07
python:     python3.10
script:     survival_curves.py

This Python script plots the survival curve data and determines the 
p-value and Log-rank test statistics for data retrieved from the UCSC Xena web platform.
"""
# Define version
__version__ = "2.0.0"

# Version notes
__update_notes__ = """
2.0.0
    -   Added plotting survival curve function.

1.0.0
    -   Initial commit, set up outline of logic and functions.
"""

# Import Packages
from datetime import datetime
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from matplotlib.colors import Normalize
import argparse
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os
import pandas as pd
import seaborn as sns

def read_input(file_path):
    """
    Read gene expression and survival data from a TSV file.

    Args:
        file_path (str): Path to the TSV file containing the data.

    Returns:
        pd.DataFrame: DataFrame containing the gene expression and survival data.
    """
    data = pd.read_csv(file_path, sep='\t')
    # Rename columns based on the mapping
    data.columns = ['col_0', 'col_1', 'OS', 'OS.time', 'median_group', 'data']
    data[['OS.time', 'OS', 'data']] = data[['OS.time', 'OS', 'data']].apply(pd.to_numeric, errors='coerce')

    return data

def calculate_hazard_ratios(data):
    """
    Calculate Hazard Ratios using Cox Proportional-Hazards Model, ensuring 'high' is always compared to 'low'.

    Args:
        data (pd.DataFrame): DataFrame containing survival and gene expression data.

    Returns:
        tuple: Hazard ratio and p-value, or (None, None) if computation is not possible.
    """
    
    # Check if 'high' and 'low' groups are present
    if 'high' not in data['median_group'].values or 'low' not in data['median_group'].values:
        print("Dataset must contain both 'high' and 'low' groups for comparison.")
        return None, None
    
    # Encode 'median_group' with 'low' as the reference category
    data_encoded = pd.get_dummies(data['median_group'], drop_first=False, prefix='median_group')
    data_encoded = data.join(data_encoded)
    
    cph = CoxPHFitter()
    
    try:
        cph.fit(data_encoded[['OS.time', 'OS', 'median_group_high']], duration_col='OS.time', event_col='OS')
        summary = cph.summary
        
        # Extract hazard ratios and p-values
        hazard_ratio = summary.loc['median_group_high', 'exp(coef)']
        hr_p_values = summary.loc['median_group_high', 'p']
        
    except KeyError as e:
        print(f"Error fitting the model: {e}")
        return None, None

    return hazard_ratio, hr_p_values

def plot_kaplan_meier_curve(data, hazard_ratio, hr_p_values, filename):
    """
    Plot Kaplan-Meier survival curves based on gene expression levels.

    Args:
        data (pd.DataFrame): DataFrame containing survival and gene expression data.
    """
    kmf = KaplanMeierFitter()
    colors = {'high': 'red', 'low': 'blue'}

    def plot_group(group, label, color):
        kmf.fit(group['OS.time'], event_observed=group['OS'])
        kmf.plot_survival_function(ci_show=False, color=color, linestyle='-', label=label)
        plt.plot(kmf.survival_function_.index, kmf.confidence_interval_['KM_estimate_upper_0.95'], color=color, linestyle=':', alpha=0.25)
        plt.plot(kmf.survival_function_.index, kmf.confidence_interval_['KM_estimate_lower_0.95'], color=color, linestyle=':', alpha=0.25)
        add_vertical_bars(group, color, kmf)
    
    def add_vertical_bars(group, color, kmf):
        # Extract unique event times
        event_times = group['OS.time'].dropna().unique()
        survival_probs = kmf.survival_function_.reindex(event_times).fillna(1.0)
        
        for time in event_times:
            if time in survival_probs.index:
                survival_prob = survival_probs.loc[time].values[0]
                tick_length = 0.025
                plt.vlines(time, ymin=survival_prob - tick_length, ymax=survival_prob + tick_length,
                           color=color, linestyle='-', alpha=1, linewidth=0.5)

    plt.figure(figsize=(4, 4))
    
    high_group = data[data.iloc[:, 4] == 'high']
    low_group = data[data.iloc[:, 4] == 'low']

    # Initialize variables to store results
    plot_high = high_group.shape[0] > 0
    plot_low = low_group.shape[0] > 0

    if plot_high:
        plot_group(high_group, 'High Expression', colors['high'])
        high_line = mlines.Line2D([], [], color=colors['high'], linestyle='-', linewidth=1, label='High Expression')
    
    if plot_low:
        plot_group(low_group, 'Low Expression', colors['low'])
        low_line = mlines.Line2D([], [], color=colors['low'], linestyle='-', linewidth=1, label='Low Expression')

    if plot_low and plot_high:
        results = logrank_test(high_group['OS.time'], low_group['OS.time'],
                               event_observed_A=high_group['OS'],
                               event_observed_B=low_group['OS'])
        p_value = results.p_value
        test_statistic = results.test_statistic
    else:
        p_value = float('nan')
        test_statistic = float('nan')

    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.xlabel('Days')
    plt.ylabel('Percent Survival')

    if plot_low:
        legend_title = (
            f'Log-rank (p={p_value:.4f})\n'
            f'HR: {hazard_ratio:.2f} (p={hr_p_values:.4f})\n'
            f'High Expression (n={high_group.shape[0]})\n'
            f'Low Expression (n={low_group.shape[0]})'
        )
        plt.legend(handles=[high_line, low_line], loc='upper right', frameon=False, fontsize='8', bbox_to_anchor=(1.0, 1.0))
    else:
        legend_title = (
            f'\nHigh Expression (n={high_group.shape[0]})'
        )
        plt.legend(handles=[high_line], loc='upper right', frameon=False, fontsize='8', bbox_to_anchor=(1.0, 1.0))

    # Add custom text for legend
    plt.draw()
    legend_bbox = plt.gca().get_legend().get_window_extent().transformed(plt.gcf().transFigure.inverted())
    plt.text(legend_bbox.x0 + 0.355, legend_bbox.y0 + 0.08,
             legend_title,
             fontsize='8', linespacing=1.5,
             verticalalignment='top', horizontalalignment='right',
             transform=plt.gcf().transFigure)

    plt.tight_layout()
    plt.savefig(f"{filename}.png", dpi=400)
    plt.savefig(f"{filename}.svg", dpi=400)
    plt.close()

    time = str(datetime.now())[:-7]
    print(f"Plot saved as '{filename}' on {time}.")

def plot_survival_map(data, filename):
    """
    Plot a scatter plot of cancer types based on hazard ratio and significance.

    Args:
        data (pd.DataFrame): DataFrame containing survival data with columns
                             'cancer', 'hazard_ratio', and 'p-value'.
        filename (str): Output file name for the plot.
    """
    data['cancer_type'] = data['cancer_type'].str.upper()

    # Calculate -log10(p-value) and assign significance stars
    data['log_hazard_ratio'] = np.log10(data['hazard_ratio'])
    data['significance'] = data['p_value'].apply(
        lambda p: '***' if p <= 0.001 else '**' if p <= 0.01 else '*' if p <= 0.05 else ''
    )

    # Define color map and normalization for hazard ratio
    cmap = plt.get_cmap('coolwarm')
    norm = Normalize(vmin=-1.5, vmax=1.5, clip=True)

    # Set up figure and axes
    fig, ax = plt.subplots(figsize=(10, 3))

    # Plot scatter plot with color representing hazard ratio
    sns.scatterplot(
        x='cancer_type',
        y=0,
        hue='hazard_ratio',
        palette=cmap,
        data=data,
        s=600,
        marker='s',
        edgecolor=None,
        legend=None,
        ax=ax
    )

    # Add color bar for hazard ratio
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.005, pad=0)
    cbar.set_label('Hazard Ratio (HR)')

    # Annotate significance stars above each marker
    for index, row in data.iterrows():
        ax.text(
            index,
            0.05,
            row['significance'],
            ha='center',
            va='bottom',
            fontsize=12,
            color='black'
        )

    # Plot labels and title for the figure
    ax.set_xlabel('Cancer Type')
    ax.set_ylabel('log$_{10}$(HR)')
    ax.set_ylim(-0.2, 0.2)
    ax.set_title('Survival Map by Cancer Code', fontsize=10, fontweight='bold')

    # Set x-tick labels with counts
    x_labels = data['cancer_type']
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, rotation_mode='anchor', ha='right', fontsize=8)

    # Remove y-axis ticks and spines
    ax.yaxis.set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plt.tight_layout()

    # Save and show plot
    plt.savefig(f"{filename}.png", dpi=400)
    plt.savefig(f"{filename}.svg", dpi=400)
    plt.show()

    time = str(datetime.now())[:-7]
    print(f"Plot saved as {filename} on {time}.")

    return data

def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="kaplan_meier_analysis.py",
        description="Plot Kaplan-Meier curves and perform log-rank test on survival data.")
    parser.add_argument(
        'file_path', type=str,
        help='Path to the TSV file containing gene expression and survival data.')
    parser.add_argument(
        '-map', '--survival_map',
        action='store_true',
        help='Flag to call on the survival mapping function based on p-values.')

    return parser.parse_args()

def main(args):
    """
    Main function to execute the script.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    filename = (args.file_path).replace(".tsv", "")

    if not args.survival_map:
        data = read_input(args.file_path)
        hazard_ratio, hr_p_values = calculate_hazard_ratios(data)
        
        df = pd.DataFrame({
            'cancer_type': [filename],
            'p_value': [hr_p_values],
            'hazard_ratio': [hazard_ratio]
        })
        
        # Check if file exists and append or create
        summary_file = "survival_summary.tsv"
        if os.path.exists(summary_file):
            df.to_csv(summary_file, mode='a', header=False, index=False, sep='\t')
        else:
            df.to_csv(summary_file, mode='w', header=True, index=False, sep='\t')

        plot_kaplan_meier_curve(data, hazard_ratio, hr_p_values, filename)

    else:
        data = pd.read_csv(args.file_path, sep='\t')
        plot_survival_map(data, filename)

if __name__ == "__main__":
    args = parse_args()
    main(args)
