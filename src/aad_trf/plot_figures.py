import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from trf import TRF
from aad_dataset import AAD_Dataset
from aad_plotter import *
from utils import *

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
plt.rc('font', family='Helvetica')

def get_data_by_subject(config_id, dataset_name, path_dict):
    """Get TRF coefficients for individual subjects and return as a dictionary"""
    files = glob.glob(f"{path_dict['models']}/dataset-{dataset_name}_models-trf_config-{config_id}_sub-*.pkl")
    files.sort()
    
    data_dict = {}
    for file in files:
        filename = os.path.basename(file).split(".")[0]
        test_sub_id = get_field_from_filename(filename, "sub")
        file_name = f"dataset-{dataset_name}_data-aad-trfscores_config-{config_id}_sub-{test_sub_id}.pkl"
        dataset = AAD_Dataset.load_from_file(os.path.join(path_dict['features'], file_name))
        
        trf = TRF.load(file)
        coef = trf.get_model_coef()
        prediction = trf.predict(dataset[dataset.sub_ids == test_sub_id])
        features = dataset.get_features()
        features = features.reshape(features.shape[0], 2, -1)
        scores = np.zeros((features.shape[0], features.shape[2]))
        for i, label in enumerate(dataset.labels):
            scores[i,:] = features[i,label,:]
        scores = scores.mean(axis=0)
        data_dict[test_sub_id] = {
            'coef': coef,
            'prediction': prediction,
            'scores': scores,
        }
    
    return data_dict, trf, dataset

def plot_clf_accuracy(clf_config_ids):
    all_clf_results = []
    for clf_config_id in clf_config_ids:
        config_id = f"classifier-{clf_config_id:03d}"
        config = load_config(config_id)
        config_id_trf = config['config_id_trf']
        config_trf = load_config(config_id_trf)
        dataset_name = config_trf['dataset']
        model_name = config['model_name']

        base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        path_dict = set_paths_from_config(base_path, config_trf)
        
        clf_results_filename = f"dataset-{dataset_name}_reports-clf-performance_models-{model_name}_config-{config_id}"
        clf_results = pd.read_csv(os.path.join(path_dict['reports'], f"{clf_results_filename}.csv"))

        print(f"Model: {model_name}, Dataset: {dataset_name}")
        print(f"Accuracy: {clf_results['accuracy_test'].mean():.3f} +/- {clf_results['accuracy_test'].std():.3f}")
        print(f"min: {clf_results['accuracy_test'].min():.3f}, max: {clf_results['accuracy_test'].max():.3f}")

        clf_results['model_name'] = model_name
        clf_results['dataset'] = dataset_name
        all_clf_results.append(clf_results)

    all_clf_results_df = pd.concat(all_clf_results, ignore_index=True)
    all_clf_results_df['dataset_model'] = all_clf_results_df['dataset'] + "_" + all_clf_results_df['model_name']

    fig, ax = plt.subplots(figsize=(6,7))
    ax = sns.swarmplot(data=all_clf_results_df, 
                        x="dataset_model", 
                        y="accuracy_test", 
                        hue="test_sub_id",
                        # jitter=0.04, 
                        marker='o',
                        palette='icefire',
                        # alpha=0.5,
                        legend=False,
                        ax=ax
                        )
    ax = sns.boxplot(data=all_clf_results_df, 
                        x="dataset_model", 
                        y="accuracy_test", 
                        width=0.4, 
                        color='k',
                        fill=False,
                        # label=["NH-logistic", "NH-SVC", "CI-logistic", "CI-SVC"],
                        ax=ax
                        )
    
    unique_subjects = all_clf_results_df['test_sub_id'].unique()
    for sub in unique_subjects:
        sub_data = all_clf_results_df[all_clf_results_df['test_sub_id'] == sub]
        ax.plot(range(len(sub_data)), sub_data['accuracy_test'], 
                color='gray', alpha=0.5  , linewidth=0.5)
        
    ax.axhline(0.5, color='gray', linestyle='--')
    ax.grid(axis='y')
    ax.set_ylim(0.4, 0.9)
    ax.set_title("Classifier performance")
    ax.set_ylabel("Accuracy")
    ax.tick_params('y', labelsize=15)
    
    return ax, all_clf_results_df

def plot_clf_correlations(all_clf_results_df, model_names=None):
    """
    Plot correlations between classifier accuracies with regression lines using Spearman's rank correlation.
    Includes correction for multiple comparisons.
    
    Parameters:
    -----------
    all_clf_results_df : pandas.DataFrame
        DataFrame containing classifier results
    model_names : list, optional
        List of custom names for models to display on the plot
    
    Returns:
    --------
    axes : matplotlib.pyplot.Axes
        The axes objects containing the plots
    corr_results_df : pandas.DataFrame
        DataFrame containing correlation coefficients and p-values (raw and corrected)
    """
    # Get unique dataset_model combinations
    models = all_clf_results_df['dataset_model'].unique()
    
    # Create figure with subplots
    fig, axes = plt.subplots(len(models)-1, len(models)-1, figsize=(3*(len(models)-1), 3*(len(models)-1)))
    fig.suptitle('Correlations between Classifier Accuracies')
    
    # Store correlation results
    corr_results = []
    
    # Import stats for correlation calculation
    from scipy import stats
    
    # Get all p-values first for multiple comparison correction
    p_values = []
    for i in range(len(models)-1):
        for j in range(i+1, len(models)):
            model1 = models[i]
            model2 = models[j]
            
            # Get data for each model
            data1 = all_clf_results_df[all_clf_results_df['dataset_model'] == model1]['accuracy_test']
            data2 = all_clf_results_df[all_clf_results_df['dataset_model'] == model2]['accuracy_test']
            
            # Calculate Spearman correlation and p-value
            rho, p_value = stats.spearmanr(data2, data1)
            p_values.append(p_value)
    
    # Perform multiple comparison correction
    import statsmodels.stats.multitest as smm
    _, p_corrected, _, _ = smm.multipletests(p_values, method="bonferroni")
    
    # Plot all combinations
    idx = 0
    for i in range(len(models)-1):
        for j in range(i+1, len(models)):
            model1 = models[i]
            model2 = models[j]
            
            # Get data for each model
            data1 = all_clf_results_df[all_clf_results_df['dataset_model'] == model1]['accuracy_test']
            data2 = all_clf_results_df[all_clf_results_df['dataset_model'] == model2]['accuracy_test']
            
            # Calculate Spearman correlation and p-value
            rho, p_value = stats.spearmanr(data1, data2)
            p_corrected_val = p_corrected[idx]
            idx += 1
            
            # Store results
            model1_name = model_names[i] if model_names is not None else model1
            model2_name = model_names[j] if model_names is not None else model2
            corr_results.append({
                'model1': model1_name,
                'model2': model2_name,
                'spearman_rho': rho,
                'p_value': p_value,
                'p_corrected': p_corrected_val
            })
            
            # Create scatter plot with regression line
            sns.regplot(x=data2, y=data1, ax=axes[i, j-1], scatter_kws={'alpha':0.5}, line_kws={'color': 'red'})
            
            # Customize plot
            if model_names is not None:
                axes[i, j-1].set_xlabel(model_names[j])
                axes[i, j-1].set_ylabel(model_names[i])
            else: 
                axes[i, j-1].set_xlabel(model2)
                axes[i, j-1].set_ylabel(model1)
            axes[i, j-1].set_xlim(0.45, 0.9)
            axes[i, j-1].set_ylim(0.45, 0.9)
            axes[i, j-1].plot([0.45, 0.9], [0.45, 0.9], 'k--', alpha=0.3)  # Add identity line
            
            # Add correlation and p-value annotation with significance stars based on corrected p-values
            sig_stars = ''
            if p_corrected_val < 0.001:
                sig_stars = '***'
            elif p_corrected_val < 0.01:
                sig_stars = '**'
            elif p_corrected_val < 0.05:
                sig_stars = '*'
            
            axes[i, j-1].text(0.05, 0.95, f'rho = {rho:.2f}{sig_stars}', 
                              transform=axes[i, j-1].transAxes,
                              verticalalignment='top')
    
    plt.tight_layout()
    
    # Add a note about correction
    fig.text(0.5, 0.01, '† p-values corrected using Bonferroni', ha='center', fontsize=9)
    
    # Convert results to DataFrame
    corr_results_df = pd.DataFrame(corr_results)
    
    return axes, corr_results_df

def main():
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    config_ids_trf = ["trf-004"]
    plot_channels = ['FCz', 'Cz', 'C3']
    plot_channnels_str = "-".join(plot_channels)
    # for config_id_trf in config_ids_trf:
    #     config_trf = load_config(config_id_trf)
    #     dataset_name = config_trf['dataset']        
    #     path_dict = set_paths_from_config(base_path, config_trf)

    #     data_dict, trf, dataset = get_data_by_subject(config_id_trf, dataset_name, path_dict)
    #     coef_array = np.stack([data['coef'] for data in data_dict.values()])
    #     coef_mean = coef_array.mean(axis=0)
    #     coef_se = coef_array.std(axis=0) / np.sqrt(coef_array.shape[0])

    #     fig_filename = f"dataset-{dataset_name}_reports-trf-coef-wave_config-{config_id_trf}_channel-{plot_channnels_str}_sub-average"
    #     ax = plot_trf_waveform(coef_mean, trf.get_delays_in_sec(), trf.delays, dataset, plot_channels=plot_channels,
    #                            se=coef_se)
    #     plt.savefig(os.path.join(path_dict['reports'], f'{fig_filename}.pdf'), transparent=True)
    #     plt.savefig(os.path.join(path_dict['reports'], f'{fig_filename}.svg'), transparent=True)
    #     plot_delays = (0.125, 0.175) if trf.direction == 'forward' else (-0.125, -0.075)
    #     ax = plot_trf_topo(coef_mean, times=trf.get_delays_in_sec(), delays=plot_delays, eeg_info=dataset.eeg_info)
    #     topo_filename = f"dataset-{dataset_name}_reports-trf-coef-topo-0.15_config-{config_id_trf}_sub-average"
    #     plt.savefig(os.path.join(path_dict['reports'], f"{topo_filename}.pdf"), transparent=True)
    #     plt.savefig(os.path.join(path_dict['reports'], f"{topo_filename}.svg"), transparent=True)

    #     scores_array = np.stack([data['scores'] for data in data_dict.values()])
    #     scores_mean = scores_array.mean(axis=0)
    #     if trf.direction == 'forward':
    #         vlim = (-np.abs(scores_mean).max(), np.abs(scores_mean).max())
    #         ax = plot_scores_topo(scores_mean, dataset.eeg_info, vlim=vlim)
    #         save_filename = f"dataset-{dataset_name}_reports-scores-topo_config-{config_id_trf}_sub-average"
    #         plt.savefig(os.path.join(path_dict['reports'], f"{save_filename}.pdf"), transparent=True)
    #         plt.savefig(os.path.join(path_dict['reports'], f"{save_filename}.svg"), transparent=True)
    #     else:
    #         pass

    #     prediction_list = []
    #     for subj in data_dict.keys():
    #         prediction = data_dict[subj]['prediction']
    #         dataset_subj = dataset[dataset.sub_ids == subj]
    #         prediction_up = prediction[:,dataset_subj.labels == 0,:].mean(axis=1)
    #         prediction_down = prediction[:,dataset_subj.labels == 1,:].mean(axis=1)
    #         if trf.direction == 'forward':
    #             prediction_up = select_channels(prediction_up.T, plot_channels, dataset)
    #             prediction_down = select_channels(prediction_down.T, plot_channels, dataset)
    #             prediction_up = prediction_up.T.squeeze()
    #             prediction_down = prediction_down.T.squeeze()
    #             prediction_up *= 1.5
    #             prediction_down *= 1.5
    #         else:
    #             prediction_up *= 30
    #             prediction_down *= 30
    #         prediction_list.append(np.stack([prediction_up, prediction_down]))
    #     prediction_array = np.stack(prediction_list)
    #     prediction_mean = prediction_array.mean(axis=0)
    #     prediction_se = prediction_array.std(axis=0) / np.sqrt(prediction_array.shape[0])

    #     times = dataset.get_times()
    #     if trf.direction == 'forward':
    #         fig, axs = plot_eeg_waveform(dataset, plot_channels=plot_channels)
    #         axs[0].plot(times, prediction_mean[0], color='r', label="Recon.", alpha=0.6)
    #         axs[0].fill_between(times, prediction_mean[0]-prediction_se[0], prediction_mean[0]+prediction_se[0], color='r', alpha=0.2)
    #         axs[0].legend(loc='upper right')
    #         axs[1].plot(times, prediction_mean[1], color='b', label="Recon.", alpha=0.6)
    #         axs[1].fill_between(times, prediction_mean[1]-prediction_se[1], prediction_mean[1]+prediction_se[1], color='b', alpha=0.2)
    #         axs[1].legend(loc='upper right')
    #         plt.setp(axs, ylim=(0, 0.3))
    #         plt.setp(axs, ylabel="Amplitude (A.U.)")
    #         fig.suptitle(f"Averaged True and Recon. EEG waveform {plot_channels}")
    #     else:
    #         fig, axs = plot_audio_waveform(dataset)
    #         axs[0].plot(times, prediction_mean[0], color='r', label="Recon.", alpha=0.6)
    #         axs[0].fill_between(times, np.squeeze(prediction_mean[0]-prediction_se[0]), np.squeeze(prediction_mean[0]+prediction_se[0]), 
    #                             color='r', alpha=0.2)
    #         axs[0].legend(loc='upper right')
    #         axs[1].plot(times, prediction_mean[1], color='b', label="Recon.", alpha=0.6)
    #         axs[1].fill_between(times, np.squeeze(prediction_mean[1]-prediction_se[1]), np.squeeze(prediction_mean[1]+prediction_se[1]), 
    #                             color='b', alpha=0.2)
    #         axs[1].legend(loc='upper right')
    #         plt.setp(axs, ylabel="Amplitude (A.U.)")
    #         fig.suptitle("Averaged True and Recon. Audio waveform")
    #     save_filename = f"dataset-{dataset_name}_reports-trf-prediction_config-{config_id_trf}_channels-{plot_channnels_str}_sub-grand-average"
    #     plt.savefig(os.path.join(path_dict['reports'], f"{save_filename}.pdf"), transparent=True)
    #     plt.savefig(os.path.join(path_dict['reports'], f"{save_filename}.svg"), transparent=True)

    clf_config_ids = [10,11,14,15]
    path_dict = set_paths_from_config(base_path, load_config(config_ids_trf[0]))
    ax, all_clf_results_df = plot_clf_accuracy(clf_config_ids) # Plot classifier accuracy
    
    from scipy.stats import friedmanchisquare
    df_pivot = all_clf_results_df.pivot(index='test_sub_id', columns='model_name', values='accuracy_test')
    stat, p_friedman = friedmanchisquare(*df_pivot.T.values)
    print(f"Friedman test statistic: {stat}, p-value: {p_friedman}")

    from scipy.stats import wilcoxon
    from itertools import combinations

    results = [{"Test": "Friedman", "Model 1": "All", "Model 2": "All", "Statistic": stat, "p-value": p_friedman}]
    model_names = df_pivot.columns
    p_values = []
    comparisons = []
    for m1, m2 in combinations(model_names, 2):
        stat, p = wilcoxon(df_pivot[m1], df_pivot[m2])
        print(f"Wilcoxon test between {m1} and {m2}: p-value = {p}")
        p_values.append(p)
        comparisons.append((m1, m2))
        results.append({"Test": "Wilcoxon", "Model 1": m1, "Model 2": m2, "Statistic": stat, "p-value": p})

    import statsmodels.stats.multitest as smm
    # Apply Bonferroni correction
    _, p_corrected, _, _ = smm.multipletests(p_values, method="bonferroni")

    # Update results with corrected p-values
    for i, (m1, m2) in enumerate(comparisons):
        results[i + 1]["p-value (Bonferroni)"] = p_corrected[i]  # Skip the Friedman test entry

    # Convert results to a DataFrame and save as CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(path_dict['reports'], "model_comparison_stats.csv"), index=False)

    ax.set_xticklabels(["Logistic", "SVC", "Backward", "CNN"])
    plt.savefig(os.path.join(path_dict['reports'], "dataset-updown-nh_reports-clf-accuracy-combined.pdf"),
        transparent=True)
    plt.savefig(os.path.join(path_dict['reports'], "dataset-updown-nh_reports-clf-accuracy-combined.svg"),
        transparent=True)
    
    axs, corr_results_df = plot_clf_correlations(all_clf_results_df, ["Logistic", "SVC", "Backward", "CNN"])
    corr_results_df.to_csv(os.path.join(path_dict['reports'], "model_correlation_stats.csv"), index=False)
    plt.savefig(os.path.join(path_dict['reports'], 
                            "dataset-updown-nh_reports-clf-accuracy-correlations.pdf"),
                transparent=True,
                bbox_inches='tight')
    plt.savefig(os.path.join(path_dict['reports'], 
                            "dataset-updown-nh_reports-clf-accuracy-correlations.svg"),
                transparent=True,
                bbox_inches='tight')

if __name__ == '__main__':
    main()
