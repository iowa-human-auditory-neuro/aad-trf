import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import mne
import seaborn as sns
from tqdm import tqdm
import pickle
import matplotlib
from utils import *
from aad_dataset import AAD_Dataset

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
plt.rc('font', family='Helvetica')

base_path = "/Users/jusungham/HANG/aad-trf/"
predictions = []
corrects = []
model_names = []
clf_config_id = 10
config_id = f"classifier-{clf_config_id:03d}"
config = load_config(config_id)
model_name = config['model_name']
config_id_trf = config['config_id_trf']
config_trf = load_config(config_id_trf)
dataset_name = config_trf['dataset']

# search fif files in data/updown-nh/eeg/
eeg_dir = base_path + f"data/{dataset_name}/eeg/"
eeg_files = [f for f in os.listdir(eeg_dir) if f.endswith('.fif')]
sub_ids = [f.split('_')[0].split('-')[1] for f in eeg_files]

for i, sub_id in enumerate(sub_ids):
    dataset = AAD_Dataset()
    dataset_fname = f'dataset-{dataset_name}_data-aad-trfscores_config-{config_id_trf}_sub-{sub_id}.pkl'
    dataset_fpath = base_path + "/data/features/" + dataset_fname
    dataset = dataset.load_from_file(dataset_fpath)
    features = dataset.get_features()
    model_fname = f'dataset-updown-nh_models-{config["model_name"]}_config-{config_id}_sub-{sub_id}.pkl'
    model_fpath = base_path + "models/" + model_fname
    with open(model_fpath, "rb") as f:
        model = pickle.load(f)
    clf = model.model
    pred_logits = clf.decision_function(features)
    patterns = np.array([np.corrcoef(features[:, i], pred_logits)[0, 1] for i in range(features.shape[1])])
    if i == 0:
        patterns_df = pd.DataFrame({
            'sub_id': sub_id,
            'feature_idx': np.arange(len(patterns)),
            'pattern': patterns
        })
    else:
        patterns_df = pd.concat([patterns_df, pd.DataFrame({
            'sub_id': sub_id,
            'feature_idx': np.arange(len(patterns)),
            'pattern': patterns
        })])
patterns_df.to_csv(f'{base_path}/reports/dataset-{dataset_name}_model-{model_name}_config-{config_id}_sub-all_patterns.csv', index=False)

epoch_path = 'data/updown-nh/eeg/sub-wj0681_task-updown-epo.fif'
epoch = mne.read_epochs(epoch_path)

# plot grand averaged importance
grand_avg_importance = np.abs(patterns_df.groupby('feature_idx')['pattern'].mean().values)
patterns_up = grand_avg_importance[:grand_avg_importance.shape[0]//2]
patterns_down = grand_avg_importance[grand_avg_importance.shape[0]//2:]
fig, axs = plt.subplots(1,2, figsize=(3*2, 3))
vlim = (0, np.max(grand_avg_importance))
for importance, ax, label in zip([patterns_up, patterns_down], axs, ['Up', 'Down']):
    topo, _ = mne.viz.plot_topomap(importance,
                        pos=epoch.info,
                        axes=ax,
                        cmap='rocket_r',
                        vlim=vlim,
                        sphere='eeglab',
                        show=False)
    plt.colorbar(topo,
                orientation='vertical', 
                shrink=0.8,
                ax=ax)
    ax.set_title(label)
plt.tight_layout()
plt.savefig(f'{base_path}/reports/dataset-{dataset_name}_model-{model_name}_config-{config_id}_sub-all_importance.pdf')

for sub_id in sub_ids:
    sub_patterns = patterns_df[patterns_df['sub_id'] == sub_id]['pattern'].values
    patterns_up = sub_patterns[:sub_patterns.shape[0]//2]
    patterns_down = sub_patterns[sub_patterns.shape[0]//2:]

    fig, axs = plt.subplots(1,2, figsize=(3*2, 3))
    vlim = (-np.max(np.abs(sub_patterns)), np.max(np.abs(sub_patterns)))
    for salmap, ax, label in zip([patterns_up, patterns_down], axs, ['Up', 'Down']):
        topo, _ = mne.viz.plot_topomap(salmap,
                            pos=epoch.info,
                            axes=ax,
                            cmap='vlag',
                            vlim=vlim,
                            sphere='eeglab',
                            show=False)
        plt.colorbar(topo,
                    orientation='vertical', 
                    shrink=0.8,
                    ax=ax)
        ax.set_title(label)
    plt.tight_layout()
    plt.savefig(f'{base_path}/reports/dataset-{dataset_name}_model-{model_name}_config-{config_id}_sub-{sub_id}_patterns.pdf')