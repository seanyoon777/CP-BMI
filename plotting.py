import numpy as np
from scipy import stats 
import matplotlib.pyplot as plt
import tensorflow as tf

color_scheme = [
    '#C55B34',  # Orange
    '#316AAE',  # Blue 
    '#7FA450',  # Light Green
    '#6FBAE8',  # Light Blue
    '#E2B147',  # Bright orange
    '#8C252E',  # Wine
    '#A38424',  # Gold
    '#6A2A7D',  # Purple
    '#128888']  # Robin

x_positions = np.array([0, -2, -1, 0, 1, 2, -3, -2, -1, 0, 1, 2, 3, -2, -1, 0, 1, 2, -1, 0, 1, 0])
y_positions = np.array([2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -2, -2, -2, -3])

def nan_aware_trim_mean(data, proportiontocut):
    """
    Calculates the trimmed mean for each column in a 2D array, ignoring NaN values.

    Args:
    - data (np.ndarray): 2D input data array.
    - proportiontocut (float): Proportion of data to trim from each end of each column.

    Returns:
    - np.ndarray: 1D array of trimmed means for each column.
    """
    n_rows, n_cols = data.shape
    trim_counts = int(proportiontocut * n_rows)
    trimmed_means = np.zeros(n_cols)
    
    for col in range(n_cols):
        col_data = data[:, col]
        sorted_col_data = np.sort(col_data[~np.isnan(col_data)])
        if trim_counts < len(sorted_col_data):
            trimmed_data = sorted_col_data[trim_counts:-trim_counts]
            trimmed_means[col] = np.mean(trimmed_data)
        else:
            trimmed_means[col] = np.nan  # If there's not enough data to trim, return NaN
    
    return trimmed_means

def compute_ci_mean(data, proportiontocut=1/9+0.1, confidence_level=0.95):
    """
    Computes the confidence interval of the mean for each column in a 2D array,
    considering trimming and ignoring NaN values.

    Args:
    - data (np.ndarray): Input data array of shape (n_samples, n_features).
    - proportiontocut (float): Proportion of data to trim.
    - confidence_level (float): Desired confidence level for the CI.

    Returns:
    - Tuple of np.ndarray: Trimmed means, lower CI bounds, upper CI bounds for each column.
    """
    trimmed_mean = nan_aware_trim_mean(data, proportiontocut)
    
    n_rows, n_cols = data.shape
    sems = np.zeros(n_cols)
    ci_lower = np.zeros(n_cols)
    ci_upper = np.zeros(n_cols)
    
    for col in range(n_cols):
        col_data = data[:, col]
        non_nan_col_data = col_data[~np.isnan(col_data)]
        if len(non_nan_col_data) > 1:
            trimmed_data = stats.trimboth(non_nan_col_data, proportiontocut)
            sd = np.nanstd(trimmed_data, ddof=1)
            n = len(trimmed_data)
            sem = sd / np.sqrt(n)
            df = n - 1
            t_crit = np.abs(stats.t.ppf((1 - confidence_level) / 2, df))
            ci_lower[col] = trimmed_mean[col] - t_crit * sem
            ci_upper[col] = trimmed_mean[col] + t_crit * sem
            sems[col] = sem
        else:
            ci_lower[col] = np.nan
            ci_upper[col] = np.nan
            sems[col] = np.nan
    
    return trimmed_mean, ci_lower, ci_upper


def compute_ci_median(data, confidence_level=0.95): 
    # Calculate the trimmed mean
    median = np.median(data[~np.isnan(data)], axis=0)
    sem = stats.sem(data[~np.isnan(data)], axis=0)
    
    df = 9 - 1  # degrees of freedom
    t_crit = np.abs(stats.t.ppf((1-confidence_level)/2, df))  # t-critical value for 95% CI

    # Calculate the CI for the trimmed mean
    ci_lower = median - t_crit * sem
    ci_upper = median + t_crit * sem
    return median, ci_lower, ci_upper

def plot_average(Y, plot_type: str, baseline_accuracy=[], ylim0=False, ylim100=False, n_iters=1): 
    all_alphas = np.arange(0.01, 0.99, 0.01)
    plt.rcParams['font.family'] = 'Arial'
    plot_idx = np.logical_and(1-all_alphas <= 0.91, 1-all_alphas >= .5)
    fig = plt.figure(figsize=(7, 6))
    
    imp_out = []
    avg_out = []
    ci_lower_out = []
    ci_upper_out = []
    
    for i in range(len(Y)): 
        plot_alphas = (1-all_alphas)[plot_idx]
        Y_slice = 100 * Y[i][:,plot_idx]
        if plot_type == 'accuracies': 
            baseline_accuracy_plot = np.repeat(baseline_accuracy[:,None], n_iters)[:,None]
            y_imp, _, _ = compute_ci_mean(Y_slice - 100 * baseline_accuracy_plot)
            imp_out.append(y_imp)
            Y_slice = (Y_slice - 100 * baseline_accuracy_plot) / (1 - baseline_accuracy_plot)
        avg, ci_lower, ci_upper = compute_ci_mean(Y_slice)
        plt.plot(plot_alphas, avg, color=color_scheme[i], linewidth=3)
        if Y.shape[0] == 1: 
            plt.fill_between(plot_alphas, ci_lower, ci_upper, color='gray', alpha=.2, linewidth=0)
        else: 
            plt.fill_between(plot_alphas, ci_lower, ci_upper, color=color_scheme[i], alpha=.2, linewidth=0)
        avg_out.append(avg)
        ci_lower_out.append(ci_lower)
        ci_upper_out.append(ci_upper)
    if ylim100: 
        plt.ylim(top=100)
    if ylim0: 
        plt.ylim(bottom=0)
    
    plt.tick_params(axis='both',          # Applies to both x and y axis
                    which='both',         # Applies to both major and minor ticks
                    direction='in',       # Sets ticks to the inside
                    bottom=True,          # Enables bottom ticks
                    top=False,             # Enables top ticks
                    left=True,            # Enables left ticks
                    right=False)           # Enables right ticks
    plt.tick_params(axis='both', length=5, width=3)

    for spine in plt.gca().spines.values():
        spine.set_linewidth(3)  # Set the thickness here

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20) 

    plt.xticks(list(np.arange(0.5, 0.91, 0.1)), labels=['0.5', '0.6', '0.7', '0.8', '0.9'], fontsize=20) 
    plt.locator_params(axis='y', nbins=6)
    plt.xlabel(r'Target Coverage ($1-\alpha$)', fontsize=22, labelpad=8, fontdict=dict(fontstretch = 'condensed'))
    if plot_type == 'prediction_rates': 
        plt.ylabel('Prediction rate (%)', fontsize=22, labelpad=10, fontdict=dict(fontstretch = 'condensed'))#, fontdict=dict(weight='bold'))
    if plot_type == 'accuracies':
        plt.ylabel('Accuracy increase (%)', fontsize=22, labelpad=10, fontdict=dict(fontstretch = 'condensed'))#, fontdict=dict(weight='bold'))
    if plot_type == 'abstention_accuracies': 
       plt.ylabel('Abstention accuracy (%)', fontsize=22, labelpad=10, fontdict=dict(fontstretch = 'condensed'))#, fontdict=dict(weight='bold'))
    plt.show()
    
    if plot_type == 'accuracies': 
        return np.vstack(y_imp), np.vstack(avg_out), np.vstack(ci_lower_out), np.vstack(ci_upper_out)
    return np.vstack(avg_out), np.vstack(ci_lower_out), np.vstack(ci_upper_out)



def plot_average_with_optimum(avg, ci_lower, ci_upper, plot_type: str, max_alpha_idx, x_adj=0, y_adj=0, ylim0=False, ylim100=False): 
    all_alphas = np.arange(0.01, 0.99, 0.01)
    plt.rcParams['font.family'] = 'Arial'
    plot_idx = np.logical_and(1-all_alphas <= 0.91, 1-all_alphas >= .5)
    fig = plt.figure(figsize=(7, 6))
    plot_alphas = (1-all_alphas)[plot_idx]

    x_max = plot_alphas[max_alpha_idx]
    alpha_max = plot_alphas[max_alpha_idx]
    
    for i in range(len(avg)): 
        y_max = avg[i, max_alpha_idx]
        plt.plot(plot_alphas, avg[i], color=color_scheme[i], linewidth=3)
        plt.scatter(x_max, y_max, marker='o', s=150, color=color_scheme[i], linewidths=3, facecolors='none')
        plt.text(x_max+x_adj, y_max+y_adj, rf'$\alpha=${1-alpha_max:.2f}', verticalalignment='top', horizontalalignment='right', fontsize=20, color=color_scheme[i])
 
        if avg.shape[0] == 1: 
            plt.fill_between(plot_alphas, ci_lower[i], ci_upper[i], color=color_scheme[i], alpha=.2, linewidth=0)
    
        else: 
            plt.fill_between(plot_alphas, ci_lower[i], ci_upper[i], color=color_scheme[i], alpha=.2, linewidth=0)
        plt.plot([x_max, x_max], [0, y_max], '--', linewidth=3, color=color_scheme[i], alpha=.35)
        plt.plot([np.min(plot_alphas), alpha_max], [y_max, y_max], '--', linewidth=3, color=color_scheme[i], alpha=.35)

    if ylim100: 
        plt.ylim(top=100)
    if ylim0: 
        plt.ylim(bottom=0)
    
    plt.tick_params(axis='both',          # Applies to both x and y axis
                    which='both',         # Applies to both major and minor ticks
                    direction='in',       # Sets ticks to the inside
                    bottom=True,          # Enables bottom ticks
                    top=False,             # Enables top ticks
                    left=True,            # Enables left ticks
                    right=False)           # Enables right ticks
    plt.tick_params(axis='both', length=5, width=3)

    for spine in plt.gca().spines.values():
        spine.set_linewidth(3)  # Set the thickness here

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20) 

    plt.xticks(list(np.arange(0.5, 0.91, 0.1)), labels=['0.5', '0.6', '0.7', '0.8', '0.9'], fontsize=20) 
    plt.locator_params(axis='y', nbins=6)
    plt.xlabel(r'Target Coverage ($1-\alpha$)', fontsize=22, labelpad=8, fontdict=dict(fontstretch = 'condensed'))
    if plot_type == 'prediction_rates': 
        plt.ylabel('Prediction rate (%)', fontsize=22, labelpad=10, fontdict=dict(fontstretch = 'condensed'))#, fontdict=dict(weight='bold'))
    if plot_type == 'accuracies':
        plt.ylabel('Accuracy increase (%)', fontsize=22, labelpad=10, fontdict=dict(fontstretch = 'condensed'))#, fontdict=dict(weight='bold'))
    if plot_type == 'abstention_accuracies': 
       plt.ylabel('Abstention accuracy (%)', fontsize=22, labelpad=10, fontdict=dict(fontstretch = 'condensed'))#, fontdict=dict(weight='bold'))
    plt.show()

    
def plot_average_pval(pvalues, x_low=0.01, x_high=1.01, x_int=0.01): 
    x = np.arange(x_low, x_high, x_int)
    fig = plt.figure(figsize=(12, 6))
    plt.rcParams['font.family'] = 'Arial'

    avg, ci_lower, ci_upper = compute_ci_mean(pvalues)
    plt.plot(100 * x, [0.05] * len(x), 'r--', linewidth=3)
    plt.plot(100 * x, avg, color=color_scheme[0], linewidth=3)
    plt.fill_between(100 * x, ci_lower, ci_upper, color=color_scheme[0], alpha=.2, linewidth=0)
    plt.grid(False)

    plt.tick_params(axis='both',          # Applies to both x and y axis
                    which='both',         # Applies to both major and minor ticks
                    direction='in',       # Sets ticks to the inside
                    bottom=True,          # Enables bottom ticks
                    top=False,             # Enables top ticks
                    left=True,            # Enables left ticks
                    right=False)           # Enables right ticks
    plt.tick_params(axis='both', length=5, width=3)

    for spine in plt.gca().spines.values():
        spine.set_linewidth(3)  # Set the thickness here

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20) 

    #plt.xticks(x, labels=['0', '20', '40', '60', '80', '100'], fontsize=20) 

    plt.xlabel('Percent of data (%)', fontsize=22, labelpad=8, fontdict=dict(fontstretch = 'condensed'))
    plt.ylabel('P-value', fontsize=22, labelpad=10, fontdict=dict(fontstretch = 'condensed'))#, fontdict=dict(weight='bold'))
    plt.ylim(0, 1.05)

    plt.text(65.5, 0.95, rf'Does not reach p < 0.05', fontsize=19.5, fontdict=dict(fontstretch = 'condensed'))
    plt.plot([57.5, 63.5], [0.971, 0.971], color=color_scheme[0], linewidth=3)
    first_percentage = np.argmax(avg <= 0.05)
    if first_percentage == 0: 
        print('Does not reach p < 0.05')
    else: 
        print(f'Reaches p < 0.05 at {x[first_percentage]}')
    plt.show()
    
def plot_all_accuracies(Y, baseline_accuracy): 
    all_alphas = np.arange(0.01, 0.99, 0.01)
    fig = plt.figure(figsize=(8, 6))
    plt.rcParams['font.family'] = 'Arial'
    plot_idx = np.logical_and(1-all_alphas <= 0.91, 1-all_alphas >= .5)
    baseline_plot_idx = np.logical_and(1-all_alphas <= 0.91, 1-all_alphas >= .4)
    plot_alphas = (1-all_alphas)[plot_idx]
    
    plot_abstaining_accuarcy = Y[:,plot_idx]

    x_baseline = [.4] * 9

    for i in range(9):
        plt.plot(plot_alphas, 100 * plot_abstaining_accuarcy[i,:], color=color_scheme[i], linewidth=3)
    plt.grid(False)

    for i, (x, y) in enumerate(zip(x_baseline, baseline_accuracy)):
        plt.plot(x, 100 * y, 'o', color=color_scheme[i], markersize=8)
        plt.plot((1-all_alphas)[baseline_plot_idx], [100 * y] * np.sum(baseline_plot_idx), '--', linewidth=3, alpha=.2, color=color_scheme[i])

    plt.tick_params(axis='both',          # Applies to both x and y axis
                    which='both',         # Applies to both major and minor ticks
                    direction='in',       # Sets ticks to the inside
                    bottom=True,          # Enables bottom ticks
                    top=False,             # Enables top ticks
                    left=True,            # Enables left ticks
                    right=False)           # Enables right ticks
    plt.tick_params(axis='both', length=5, width=3)

    for spine in plt.gca().spines.values():
        spine.set_linewidth(3)  # Set the thickness here

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20) 

    plt.xticks([x_baseline[0]] + list(np.arange(0.5, 0.91, 0.1)), labels=['Baseline', '0.5', '0.6', '0.7', '0.8', '0.9'])
    plt.xlim(0.33, 0.95)  # Set the x-axis limits to make space for the baseline label
    #plt.xticks(xticks, [f'{x:.1f}' for x in xticks], fontsize=20) 

    plt.xlabel(r'Target Coverage ($1-\alpha$)', fontsize=22, labelpad=8, fontdict=dict(fontstretch = 'condensed'))
    plt.ylabel('Accuracy (%)', fontsize=22, labelpad=10, fontdict=dict(fontstretch = 'condensed'))#, fontdict=dict(weight='bold'))

    plt.show()
    
def plot_all_prediction_rates(Y): 
    all_alphas = np.arange(0.01, 0.99, 0.01)
    fig = plt.figure(figsize=(7, 6))
    plt.rcParams['font.family'] = 'Arial'
    plot_idx = np.logical_and(1-all_alphas <= 0.91, 1-all_alphas >= .5)
    plot_alphas = (1-all_alphas)[plot_idx]

    for i in range(9):
        plt.plot(plot_alphas, 100 * Y[i,plot_idx], color=color_scheme[i], linewidth=3)
    plt.grid(False)

    plt.tick_params(axis='both',          # Applies to both x and y axis
                    which='both',         # Applies to both major and minor ticks
                    direction='in',       # Sets ticks to the inside
                    bottom=True,          # Enables bottom ticks
                    top=False,             # Enables top ticks
                    left=True,            # Enables left ticks
                    right=False)           # Enables right ticks
    plt.tick_params(axis='both', length=5, width=3)

    for spine in plt.gca().spines.values():
        spine.set_linewidth(3)  # Set the thickness here

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20) 

    plt.xticks(list(np.arange(0.5, 0.91, 0.1)), labels=['0.5', '0.6', '0.7', '0.8', '0.9'], fontsize=20) 

    plt.xlabel(r'Target Coverage ($1-\alpha$)', fontsize=22, labelpad=8, fontdict=dict(fontstretch = 'condensed'))
    plt.ylabel('Prediction rate (%)', fontsize=22, labelpad=10, fontdict=dict(fontstretch = 'condensed'))#, fontdict=dict(weight='bold'))

    plt.show()

def plot_all_abstention_accuracies(Y): 
    all_alphas = np.arange(0.01, 0.99, 0.01)
    fig = plt.figure(figsize=(7, 6))
    plt.rcParams['font.family'] = 'Arial'
    plot_idx = np.logical_and(1-all_alphas <= 0.91, 1-all_alphas >= .5)
    plot_alphas = (1-all_alphas)[plot_idx]

    for i in range(9):
        plt.plot(plot_alphas, 100 * Y[i,plot_idx], color=color_scheme[i], linewidth=3)
    plt.grid(False)

    plt.tick_params(axis='both',          # Applies to both x and y axis
                    which='both',         # Applies to both major and minor ticks
                    direction='in',       # Sets ticks to the inside
                    bottom=True,          # Enables bottom ticks
                    top=False,             # Enables top ticks
                    left=True,            # Enables left ticks
                    right=False)           # Enables right ticks
    plt.tick_params(axis='both', length=5, width=3)

    for spine in plt.gca().spines.values():
        spine.set_linewidth(3)  # Set the thickness here

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20) 

    plt.xticks(list(np.arange(0.5, 0.91, 0.1)), labels=['0.5', '0.6', '0.7', '0.8', '0.9'], fontsize=20) 

    plt.xlabel(r'Target Coverage ($1-\alpha$)', fontsize=22, labelpad=8, fontdict=dict(fontstretch = 'condensed'))
    plt.ylabel('Abstention accuracies (%)', fontsize=22, labelpad=10, fontdict=dict(fontstretch = 'condensed'))#, fontdict=dict(weight='bold'))

    plt.show()
    
def plot_all_coverages(empirical_coverages): 
    all_alphas = np.arange(0.01, 0.99, 0.01)
    fig = plt.figure(figsize=(6, 6))
    plt.rcParams['font.family'] = 'Arial'
    # Plot 1: Target Coverage (x) & Empirical Coverage
    for i in range(empirical_coverages.shape[0]):
        plt.plot(1-all_alphas, empirical_coverages[i,:], color=color_scheme[i], linewidth=3)
    plt.plot(all_alphas, all_alphas, 'r--', label='y=x', linewidth=2)  # 'k--' is for black dotted line
    plt.grid(False)

    plt.tick_params(axis='both',          # Applies to both x and y axis
                    which='both',         # Applies to both major and minor ticks
                    direction='in',       # Sets ticks to the inside
                    bottom=True,          # Enables bottom ticks
                    top=False,             # Enables top ticks
                    left=True,            # Enables left ticks
                    right=False)           # Enables right ticks
    plt.tick_params(axis='both', length=5, width=3)

    for spine in plt.gca().spines.values():
        spine.set_linewidth(3)  # Set the thickness here

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.xlabel(r'Target Coverage ($1-\alpha$)', fontsize=22, labelpad=8, fontdict=dict(fontstretch = 'condensed'))
    plt.ylabel('Empirical Coverage', fontsize=22, labelpad=10)#, fontdict=dict(weight='bold'))

    plt.show()

def plot_average_coverage(empirical_coverages): 
    all_alphas = np.arange(0.01, 0.99, 0.01)
    fig = plt.figure(figsize=(6, 6))
    plt.rcParams['font.family'] = 'Arial'
    avg, ci_lower, ci_upper = compute_ci_mean(empirical_coverages)
    plt.plot(1-all_alphas, avg, color=color_scheme[0], linewidth=3)
    plt.fill_between(1-all_alphas, ci_lower, ci_upper, color='gray', alpha=.2, linewidth=0)

    pvalue = stats.kstest(1-all_alphas, avg, alternative='two-sided')[1]
    low_pvalue = stats.kstest(1-all_alphas, ci_lower, alternative='two-sided')[1]
    if pvalue > 0.999: 
        pvalue = 0.999
    plt.text(0.38, 0.01, fr'P-value: {pvalue:.3f}$\pm${low_pvalue-pvalue:.3f}', fontsize=20, fontdict=dict(fontstretch = 'condensed'))
    
    plt.plot(all_alphas, all_alphas, 'r--', label='y=x', linewidth=2)  # 'k--' is for black dotted line
    plt.grid(False)

    plt.tick_params(axis='both',          # Applies to both x and y axis
                    which='both',         # Applies to both major and minor ticks
                    direction='in',       # Sets ticks to the inside
                    bottom=True,          # Enables bottom ticks
                    top=False,             # Enables top ticks
                    left=True,            # Enables left ticks
                    right=False)           # Enables right ticks
    plt.tick_params(axis='both', length=5, width=3)

    for spine in plt.gca().spines.values():
        spine.set_linewidth(3)  # Set the thickness here

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.xlabel(r'Target Coverage ($1-\alpha$)', fontsize=22, labelpad=8, fontdict=dict(fontstretch = 'condensed'))
    plt.ylabel('Empirical Coverage', fontsize=22, labelpad=10)#, fontdict=dict(weight='bold'))

    plt.show()
    
def plot_all_pvalues(pvalues): 
    fig = plt.figure(figsize=(12, 6))
    plt.rcParams['font.family'] = 'Arial'
    x = np.arange(0.01, 1.01, 0.01)

    plt.plot(100 * x, [0.05] * len(x), 'r--', linewidth=3)
    for i in range(9): 
        plt.plot(100 * x, pvalues[i,:], color=color_scheme[i], linewidth=3)
    plt.grid(False)

    plt.tick_params(axis='both',          # Applies to both x and y axis
                    which='both',         # Applies to both major and minor ticks
                    direction='in',       # Sets ticks to the inside
                    bottom=True,          # Enables bottom ticks
                    top=False,             # Enables top ticks
                    left=True,            # Enables left ticks
                    right=False)           # Enables right ticks
    plt.tick_params(axis='both', length=5, width=3)

    for spine in plt.gca().spines.values():
        spine.set_linewidth(3)  # Set the thickness here

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20) 

    #plt.xticks(x, labels=['0', '20', '40', '60', '80', '100'], fontsize=20) 

    plt.xlabel('Calibration set size (%)', fontsize=22, labelpad=8, fontdict=dict(fontstretch = 'condensed'))
    plt.ylabel('P-value', fontsize=22, labelpad=10, fontdict=dict(fontstretch = 'condensed'))#, fontdict=dict(weight='bold'))

    plt.show()
    
def get_coverage_stats(empirical_coverages): 
    all_alphas = np.arange(0.01, 0.99, 0.01)
    print("KS test for equality of distribution of empirical coverage and target coverage:")
    for i in range(9): 
        print(stats.kstest(1-all_alphas, empirical_coverages[i,:], alternative='two-sided')[1])
        
        
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def plot_scalp_topography(channel_importance, x_positions, y_positions, title='Channel Contribution'):
    """
    Plot a scalp topography map of channel contributions.
    
    Parameters:
    - channel_importance: Array of channel importance scores.
    - x_positions, y_positions: Arrays of x and y positions of the channels on the scalp.
    - title: Title of the plot.
    """
    # Define grid to interpolate
    grid_x, grid_y = np.mgrid[min(x_positions):max(x_positions):100j, min(y_positions):max(y_positions):100j]
    
    # Interpolate channel importance scores onto the grid
    grid_z = griddata((x_positions, y_positions), channel_importance, (grid_x, grid_y), method='cubic')
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.imshow(grid_z.T, extent=(min(x_positions), max(x_positions), min(y_positions), max(y_positions)), origin='lower')
    plt.colorbar(label='Importance')
    plt.scatter(x_positions, y_positions, c='k', s=5)
    plt.title(title)
    plt.show()

def compute_gradients_for_class(model, inputs, labels, n_classes=4, conformal=False):
    """
    Compute the average gradient of the model's prediction for a specific target class
    with respect to the inputs that actually belong to that class.
    
    Parameters:
    - model: The Keras model.
    - inputs: Input data of shape (#samples, 1, #channels, #timestamps).
    - labels: True labels for each input, one-hot encoded.
    - target_class: The index of the target class for which to compute gradients.
    
    Returns:
    - Average gradients of shape (#channels, #timestamps), representing the importance
      of each channel for predicting the target class, averaged over inputs that
      actually belong to the target class.
    """
    n_channels = inputs.shape[2]
    all_gradients = np.zeros((n_classes, n_channels))
    for target_class in range(n_classes): 
        # Filter inputs and labels for the target class
        class_inputs = inputs[np.argmax(labels, axis=1) == target_class]
        
        # Initialize gradients sum
        gradients_sum = np.zeros((class_inputs.shape[2],))
        count = 0
        
        for sample in class_inputs:
            with tf.GradientTape() as tape:
                sample_tensor = tf.convert_to_tensor(sample[None, ...], dtype=tf.float32)
                tape.watch(sample_tensor)
                prediction = model(sample_tensor)[0, target_class]
            # Compute gradients for the target class prediction
            gradients = tape.gradient(prediction, sample_tensor)
            gradients_sum += np.average(gradients.numpy().squeeze(), axis=1)
            count += 1
        
        # Average the gradients
        all_gradients[target_class,:] = gradients_sum / max(count, 1)
    
    return all_gradients

def average_gradients(gradients):
    """
    Average gradients across samples and timestamps while retaining the channel dimension.
    
    Parameters:
    - gradients: Gradients of shape (#samples, 1, #channels, #timestamps).
    
    Returns:
    - Averaged gradients of shape (#channels).
    """
    # Taking absolute value to consider overall magnitude of gradients
    abs_gradients = np.abs(gradients)
    mean_gradients = np.mean(abs_gradients, axis=(0, 1, 3))  # Averaging over samples and timestamps
    return mean_gradients.squeeze()  # Removing singleton dimensions for plotting

def plot_scalp_topography_for_class(channel_importance, x_positions, y_positions, title_prefix='Channel Contribution for Class: '):
    """
    Plot a scalp topography map of channel contributions for a specific class.
    
    Parameters:
    - channel_importance: Array of channel importance scores.
    - x_positions, y_positions: Arrays of x and y positions of the channels on the scalp.
    - class_label: The label or index of the class for the topography.
    - title_prefix: Prefix for the title of the plot.
    """

    plot_scalp_topography(channel_importance, x_positions, y_positions)
    
from matplotlib.patches import Ellipse

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from matplotlib.patches import Circle, Ellipse, Polygon


def plot_scalp_topography(channel_importance, x_positions, y_positions, title='Channel Contribution'):
    grid_size = 8
    half_grid = grid_size / 2
    # Create a grid
    grid_x, grid_y = np.mgrid[-half_grid:half_grid:80j, -half_grid:half_grid:80j]

    # Find the convex hull of the electrode positions
    points = np.column_stack((x_positions, y_positions))
    hull = ConvexHull(points)
    
    # Inflate the convex hull by a distance of 1 unit
    hull_points = points[hull.vertices]
    hull_center = np.mean(hull_points, axis=0)
    inflated_hull_points = hull_points + ((hull_points - hull_center) * (1 / np.linalg.norm(hull_points - hull_center, axis=1))[:, np.newaxis])

    # Interpolate the values inside the inflated convex hull
    grid_z_inside = griddata(
        (x_positions, y_positions), channel_importance,
        (grid_x, grid_y), method='cubic', fill_value=np.nan
    )
    
    # Set points more than 1 unit away from any electrode to zero
    X_flat = grid_x.flatten()
    Y_flat = grid_y.flatten()
    for i, point in enumerate(zip(X_flat, Y_flat)):
        distances = np.sqrt((x_positions - point[0]) ** 2 + (y_positions - point[1]) ** 2)
        if np.all(distances > 1):
            grid_z_inside.flat[i] = 0
    
    points_to_interpolate = np.isnan(grid_z_inside)

    # Known points will be used to interpolate, these should be non-NaN and inside boundary
    known_x = grid_x[~points_to_interpolate]
    known_y = grid_y[~points_to_interpolate]
    known_z = grid_z_inside[~points_to_interpolate]

    # Perform the interpolation for points inside the boundary radius that are NaN
    # Use only the known (non-NaN) values to interpolate
    grid_z_inside[points_to_interpolate] = griddata(
        (known_x, known_y),  # Points with known values
        known_z,             # Known values
        (grid_x[points_to_interpolate], grid_y[points_to_interpolate]),  # Points to interpolate
        method='cubic'
    )
    sigma = 1  # Standard deviation for Gaussian kernel
    grid_z_inside = gaussian_filter(grid_z_inside, sigma=sigma)

    boundary_radius = 4
    outside_boundary_mask = np.sqrt(grid_x**2 + grid_y**2) >= boundary_radius
    grid_z_inside[outside_boundary_mask] = np.nan
    
    # Plotting the results
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    cmap = plt.get_cmap('RdBu_r')
    
    circle = Circle((0, 0), 4, edgecolor='black', facecolor='#132F5E', linewidth=3)
    ax.add_patch(circle)
    
    # Create an ellipse
    ellipse = Ellipse((-4, 0), 1, 2, edgecolor='black', facecolor='none', linewidth=3)
    ax.add_patch(ellipse)
    ellipse2 = Ellipse((4, 0), 1, 2, edgecolor='black', facecolor='none', linewidth=3)
    ax.add_patch(ellipse2)

    # Create a right triangle (right-angled triangle)
    right_triangle = Polygon([(0.5, 3.8), (-0.5, 3.8), (0, 4.5)], closed=True, edgecolor='black', linewidth=3, facecolor='none')
    ax.add_patch(right_triangle)
    
    # Plot the interpolated data
    contour = ax.contourf(grid_x, grid_y, grid_z_inside, 200, cmap=cmap)
    
    # Plot the electrode positions
    ax.scatter(x_positions, y_positions, c='k', s=20)
    
    # Set plot limits
    ax.set_xlim([-half_grid-1.5, half_grid+2])
    ax.set_ylim([-half_grid-1.5, half_grid+2])
    # Remove axis labels and ticks
    ax.axis('off')
    
    plt.show()

def compute_matrix_quantities(true_positives, true_negatives, false_positives, false_negatives): 
    n1, n2, _ = true_positives.shape
    tpr = np.zeros((n1, n2))
    fpr = np.zeros((n1, n2))
    precision = np.zeros((n1, n2))
    for subject_num in range(n1): 
        tpr[subject_num,:] = np.mean(true_positives[subject_num] / (true_positives[subject_num] + false_negatives[subject_num]), axis=1)
        fpr[subject_num,:] = np.mean(false_positives[subject_num] / (true_negatives[subject_num] + false_positives[subject_num]), axis=1)
        precision_array = true_positives[subject_num] / (true_positives[subject_num] + false_positives[subject_num])
        precision_array[np.isnan(precision_array)] = 1
        precision[subject_num,:] = np.mean(precision_array, axis=1)
    return tpr, fpr, precision
    
def plot_prcurve(tpr, precision, baseline_accuracies=None, x_adj=0.0215, y_adj=-0.0115, print_area=True): 
    tpr_mean, ci_lower_tpr, ci_upper_tpr = compute_ci_mean(tpr)
    precision_mean, ci_lower_pre, ci_upper_pre = compute_ci_mean(precision)

    sort_index = np.argsort(tpr_mean)
    f1_scores = 2 * (precision_mean * tpr_mean) / (precision_mean + tpr_mean)
    # Find the threshold that maximizes the F1 score
    max_f1_index = np.argmax(f1_scores)
    alpha_max = (1 - np.arange(0.01, 1.00, 0.01))[sort_index][max_f1_index]
    precision_max = precision_mean[max_f1_index]
    tpr_max = tpr_mean[max_f1_index]
    

    tpr_sorted = tpr_mean[sort_index]
    ci_lower_tpr = ci_lower_tpr[sort_index]
    ci_upper_tpr = ci_upper_tpr[sort_index]
    precision_sorted = precision_mean[sort_index]
    ci_lower_pre = ci_lower_pre[sort_index]
    ci_upper_pre = ci_upper_pre[sort_index]
    area = np.trapz(precision_sorted, tpr_sorted)
    area_lower = np.trapz(ci_lower_pre, tpr_sorted)
    
    if baseline_accuracies is not None: 
        naive_precision = np.mean(1-baseline_accuracies)
    
    fig = plt.figure(figsize=(6, 6))
    plt.rcParams['font.family'] = 'Arial'
    # Plot 1: Target Coverage (x) & Empirical Coverage
    plt.plot(tpr_sorted, precision_sorted, color=color_scheme[0], linewidth=3)
    plt.fill_between(tpr_sorted, ci_lower_pre, ci_upper_pre, color=color_scheme[0], alpha=.2, linewidth=0)
    plt.text(tpr_max+x_adj, precision_max+y_adj, rf'$\alpha=${alpha_max:.2f}', verticalalignment='top', horizontalalignment='right', fontsize=20, color=color_scheme[0])
    plt.scatter(tpr_max, precision_max, marker='o', s=150, color=color_scheme[0], linewidths=3, facecolors='none')
    
    plt.plot([tpr_max, tpr_max], [0, precision_max], '--', linewidth=3, color=color_scheme[0], alpha=.35)
    plt.plot([np.min(tpr_sorted), tpr_max], [precision_max, precision_max], '--', linewidth=3, color=color_scheme[0], alpha=.35)
    
    plt.grid(False)

    plt.tick_params(axis='both',          # Applies to both x and y axis
                    which='both',         # Applies to both major and minor ticks
                    direction='in',       # Sets ticks to the inside
                    bottom=True,          # Enables bottom ticks
                    top=False,             # Enables top ticks
                    left=True,            # Enables left ticks
                    right=False)           # Enables right ticks
    plt.tick_params(axis='both', length=5, width=3)

    for spine in plt.gca().spines.values():
        spine.set_linewidth(3)  # Set the thickness here

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(0, 1.05)
    plt.xlabel('Recall', fontsize=22, labelpad=8, fontdict=dict(fontstretch = 'condensed'))
    plt.ylabel('Precision', fontsize=22, labelpad=10)#, fontdict=dict(weight='bold'))
    if print_area: 
        plt.text(0.56, 0.13, rf'Baseline: {naive_precision:.3f}', fontsize=20, fontdict=dict(fontstretch = 'condensed'))
        plt.text(0.45, 0.05, rf'Area: {area:.3f}$\pm${(area-area_lower):.3f}', fontsize=20, fontdict=dict(fontstretch = 'condensed'))
        plt.plot([0.32, 0.42], [0.068, 0.068], color=color_scheme[0], linewidth=3)
    plt.show()
    
def plot_roccurve(fpr, tpr, x_adj=0.0215, y_adj=-0.0115, alpha_max=None): 
    fpr_mean, ci_lower_fpr, ci_upper_fpr = compute_ci_mean(fpr)
    tpr_mean, ci_lower_tpr, ci_upper_tpr = compute_ci_mean(tpr)

    sort_index = np.argsort(fpr_mean)
    tpr_sorted = np.insert(np.append(tpr_mean[sort_index], 1), 0, 0)
    fpr_sorted = np.insert(np.append(fpr_mean[sort_index], 1), 0, 0)
    ci_lower_fpr = ci_lower_fpr[sort_index]
    ci_upper_fpr = ci_upper_fpr[sort_index]
    ci_lower_tpr = ci_lower_tpr[sort_index]
    ci_upper_tpr = ci_upper_tpr[sort_index]

    j_index = tpr_mean - fpr_mean  # Youden's J statistic
    max_j_index = np.argmax(j_index[50:]) + 50
    if alpha_max == None: 
        alpha_max = (1 - np.arange(0.01, 1.00, 0.01))[max_j_index]
    tpr_max = tpr_mean[max_j_index]
    fpr_max = fpr_mean[max_j_index]
    
    area = np.trapz(tpr_sorted[1:-1], fpr_sorted[1:-1])
    area_upper = np.trapz(ci_upper_tpr, fpr_sorted[1:-1])
    area_lower = np.trapz(ci_lower_tpr, fpr_sorted[1:-1])
    print(f"Area under the curve: {area} $\plusminus$ {area_upper-area}")

    fig = plt.figure(figsize=(6, 6))
    plt.rcParams['font.family'] = 'Arial'
    plt.plot(fpr_sorted, tpr_sorted, color=color_scheme[0], linewidth=3)
    plt.fill_between(fpr_sorted[1:-1], ci_lower_tpr, ci_upper_tpr, color=color_scheme[0], alpha=.2, linewidth=0)
    #plt.fill_betweenx(tpr_sorted[1:-1], ci_lower_fpr, ci_upper_fpr, color='gray', alpha=.2, linewidth=0)
    plt.plot([0, 1], [0, 1], 'r--', label='y=x', linewidth=2)  # 'k--' is for black dotted line
    
    plt.text(fpr_max+x_adj, tpr_max+y_adj, rf'$\alpha=${alpha_max:.2f}', verticalalignment='top', horizontalalignment='right', fontsize=20, color=color_scheme[0])
    plt.scatter(fpr_max, tpr_max, marker='o', s=150, color=color_scheme[0], linewidths=3, facecolors='none')
    
    plt.plot([fpr_max, fpr_max], [0, tpr_max], '--', linewidth=3, color=color_scheme[0], alpha=.35)
    plt.plot([0, fpr_max], [tpr_max, tpr_max], '--', linewidth=3, color=color_scheme[0], alpha=.35)
    
    plt.grid(False)

    plt.tick_params(axis='both',          # Applies to both x and y axis
                    which='both',         # Applies to both major and minor ticks
                    direction='in',       # Sets ticks to the inside
                    bottom=True,          # Enables bottom ticks
                    top=False,             # Enables top ticks
                    left=True,            # Enables left ticks
                    right=False)           # Enables right ticks
    plt.tick_params(axis='both', length=5, width=3)

    for spine in plt.gca().spines.values():
        spine.set_linewidth(3)  # Set the thickness here

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.xlabel('False Positive Rate', fontsize=22, labelpad=8, fontdict=dict(fontstretch = 'condensed'))
    plt.ylabel('True Positive Rate', fontsize=22, labelpad=10)#, fontdict=dict(weight='bold'))
    plt.plot([0.32, 0.42], [0.031, 0.031], color=color_scheme[0], linewidth=3)
    plt.text(0.45, 0.01, rf'Area: {area:.3f}$\pm${(area-area_lower):.3f}', fontsize=20, fontdict=dict(fontstretch = 'condensed'))
    plt.show()
    
    
def plot_prr_curve(prediction_rate, tpr, x_adj=0, y_adj=0):
    fig = plt.figure(figsize=(6, 6))
    plt.rcParams['font.family'] = 'Arial'

    recall, ci_lower_recall, ci_upper_recall = compute_ci_mean(tpr)

    sort_index = np.argsort(prediction_rate)
    prediction_rate_sorted = np.flip(prediction_rate[sort_index])

    f1_scores = 2 * (prediction_rate_sorted * recall) / (prediction_rate_sorted + recall)
    recall = recall[sort_index]
    ci_lower_recall = ci_lower_recall[sort_index]
    ci_upper_recall = ci_upper_recall[sort_index]

    max_f1_index = np.argmax(f1_scores)
    alpha_max = (np.arange(0.01, 1.00, 0.01))[max_f1_index]
    pred_max = prediction_rate_sorted[max_f1_index]
    recall_max = recall[max_f1_index]

    plt.plot(prediction_rate_sorted, recall, color=color_scheme[0], linewidth=4)
    plt.fill_between(prediction_rate_sorted, ci_lower_recall, ci_upper_recall, color=color_scheme[0], alpha=.2, linewidth=0)
    plt.grid(False)

    plt.tick_params(axis='both',          # Applies to both x and y axis
                    which='both',         # Applies to both major and minor ticks
                    direction='in',       # Sets ticks to the inside
                    bottom=True,          # Enables bottom ticks
                    top=False,             # Enables top ticks
                    left=True,            # Enables left ticks
                    right=False)           # Enables right ticks
    plt.tick_params(axis='both', length=5, width=3)

    for spine in plt.gca().spines.values():
        spine.set_linewidth(3)  # Set the thickness here

    plt.plot([pred_max, pred_max], [0, recall_max], '--', linewidth=3, color=color_scheme[0], alpha=.35)
    plt.plot([0, pred_max], [recall_max, recall_max], '--', linewidth=3, color=color_scheme[0], alpha=.35)
    
    plt.text(pred_max+x_adj, recall_max+y_adj, rf'$\alpha=${alpha_max:.2f}', verticalalignment='top', horizontalalignment='right', fontsize=20, color=color_scheme[0])
    plt.scatter(pred_max, recall_max, marker='o', s=150, color=color_scheme[0], linewidths=3, facecolors='none')
    # plt.scatter(data_point_x, data_point_y, marker='o', s=150, color=color_scheme[0], linewidths=3, facecolors='none')
    # for i, (x, y, label) in enumerate(zip(data_point_x, data_point_y, point_labels)):
    #plt.text(x+0.0215, y-0.0115, f' {label}', verticalalignment='top', horizontalalignment='right', fontsize=20, color=color_scheme[0])

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20) 

    plt.xlabel('Prediction Rate', fontsize=22, labelpad=8, fontdict=dict(fontstretch = 'condensed'))
    plt.ylabel('Recall', fontsize=22, labelpad=10, fontdict=dict(fontstretch = 'condensed'))#, fontdict=dict(weight='bold'))
    # plt.xlim(0.65,1.01)
    # plt.ylim(0,1.03)
    plt.show()
    
def maximize_f1(x, y): 
    all_alphas = np.arange(0.01, 0.99, 0.01)

    plot_idx = np.logical_and(1-all_alphas <= 0.91, 1-all_alphas >= .5)
    plot_alphas = all_alphas[plot_idx]

    f1_scores = 2 * (x[plot_idx] * y[plot_idx]) / (x[plot_idx] + y[plot_idx])
    max_idx = np.argmax(f1_scores)

    return max_idx, plot_alphas[max_idx], x[plot_idx][max_idx], y[plot_idx][max_idx]

def normalized_f1_score(x, y): 
    x_norm = (x-np.min(x))/(np.max(x)-np.min(x)) 
    y_norm = (y-np.min(y))/(np.max(y)-np.min(y)) 
    return np.argmax((2 * x_norm * y_norm) / (x_norm + y_norm))