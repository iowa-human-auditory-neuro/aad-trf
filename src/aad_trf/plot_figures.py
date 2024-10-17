import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from trf import TRF
from aad_dataset import AAD_Dataset
from aad_plotter import *
from utils import *

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

config_ids = ["trf-004"]
plot_channels = ['FCz']
time_step = 0.1

coefs_nh = np.array([])
coefs_ci = np.array([])
for config_id in config_ids:
    config = load_config(config_id)
    dataset_name = config['dataset']
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    path_dict = set_paths_from_config(base_path, config)

    files = glob.glob(f"{path_dict['models']}/dataset-{dataset_name}_models-trf_config-{config_id}_sub-*.pkl")
    files.sort()

    scores_mean = None
    prediction = []
    for file in files:
        filename = os.path.basename(file).split(".")[0]
        test_sub_id = get_field_from_filename(filename, "sub")
        print(f"Test subject: {test_sub_id}")

        dataset = AAD_Dataset.load_from_file(f"{path_dict['features']}/dataset-{dataset_name}_data-aad-trfscores_config-{config_id}_sub-{test_sub_id}.pkl")
        test_dataset = dataset[dataset.sub_ids == test_sub_id]

        trf = TRF.load(file)
        coef = trf.get_model_coef()
        coef = select_channels(coef, plot_channels, dataset)
        coef = coef.squeeze()

        if config_id == "trf-001":
            coefs_nh = np.append(coefs_nh, coef)
            coefs_nh = coefs_nh.reshape(-1, len(coef))
        else:
            coefs_ci = np.append(coefs_ci, coef)
            coefs_ci = coefs_ci.reshape(-1, len(coef))

        features = dataset.get_features()
        features = features.reshape(features.shape[0],2,-1)
        scores = np.zeros((features.shape[0],features.shape[2]))
        labels = dataset.labels
        for i, label in enumerate(labels):
            scores[i,:] = features[i,label,:]
        scores = scores.mean(axis=0)
        if scores_mean is None:
            scores_mean = scores
        else:
            scores_mean += scores

        prediction.append(trf.predict(test_dataset))
        
        # vlim = (-np.abs(scores).max(), np.abs(scores).max())
        # ax = plot_scores_topo(scores,
        #                  dataset.eeg_info,
        #                  vlim=vlim
        #                  )
    
    scores_mean /= len(files)
    vlim = (-np.abs(scores_mean).max(), np.abs(scores_mean).max())
    ax = plot_scores_topo(scores_mean,
                        dataset.eeg_info,
                        vlim=vlim
                        )
    plt.savefig(f"{path_dict['reports']}/dataset-{dataset_name}_reports-scores-topo_config-{config_id}_sub-average.svg",
                transparent=True)
    
    prediction = np.concatenate(prediction, axis=1)
    plot_channels = ['FCz']
    plot_channels_str = "".join(plot_channels)
    fig_filename = f"dataset-{dataset_name}_reports-trf-prediction_config-{config_id}_sub-grand-average_ch-{plot_channels_str}"
    save_path = os.path.join(path_dict['reports'], f"{fig_filename}.svg")
    if trf.direction == 'forward':
        axs = plot_eeg_prediction(dataset,
                                prediction,
                                plot_channels=plot_channels,
                                scaling_factor=1.5
                                )
    elif trf.direction == 'backward':
        axs = plot_audio_prediction(dataset, prediction)
    plt.savefig(save_path)

times = trf.get_delays_in_sec()
delays = trf.delays

coefs_nh_mean = coefs_nh.mean(axis=0)
coefs_ci_mean = coefs_ci.mean(axis=0)
coefs_nh_std = coefs_nh.std(axis=0)
coefs_ci_std = coefs_ci.std(axis=0)

fig, ax = plt.subplots(figsize=(7,6))
ax.plot(times, coefs_nh_mean, color='k', marker='.', linestyle='-', label='NH')
ax.fill_between(times, coefs_nh_mean-coefs_nh_std, coefs_nh_mean+coefs_nh_std, color='k', alpha=0.2)
ax.plot(times, coefs_ci_mean, color='r', marker='.', linestyle='--', label='CI')
ax.fill_between(times, coefs_ci_mean-coefs_ci_std, coefs_ci_mean+coefs_ci_std, color='r', alpha=0.2)
ax.axvline(0, color='k', linestyle=':')
ax.axhline(0, color='k', linestyle='-')
ax.set_xlim(delays)
ax.set_xticks(np.arange(delays[0], delays[1]+time_step, time_step))
ax.legend()
ax.set_title(f"TRF coefficients {plot_channels}")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Coefficient (A.U.)")
plt.savefig(f"{path_dict['reports']}/dataset-updown-nh-ci_reports-trf-coef-wave_config_sub-average.svg",
            transparent=True)


config_ids = ["classifier-004", "classifier-005", "classifier-006", "classifier-009"]
all_clf_results = []

for config_id in config_ids:
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

fig, ax = plt.subplots(figsize=(10,8))
ax = sns.stripplot(data=all_clf_results_df, 
                    x="dataset_model", 
                    y="accuracy_test", 
                    jitter=0.04, 
                    color='k',
                    marker='o',
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
ax.axhline(0.5, color='gray', linestyle='--')
ax.grid(axis='y')
ax.set_ylim(0.2, 0.8)
ax.set_title("Classifier performance")
ax.set_ylabel("Accuracy")
ax.tick_params('y', labelsize=15)

plt.savefig(os.path.join(path_dict['reports'], "all_clf_results.svg"),
            transparent=True)