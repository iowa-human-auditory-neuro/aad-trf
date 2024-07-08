import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import mne
from sklearn.model_selection import LeaveOneGroupOut

from aad_dataset import AAD_Dataset
from trf import TRF
from utils import load_config, set_paths_from_config
from aad_plotter import plot_eeg_prediction, plot_audio_prediction

parser = argparse.ArgumentParser()
parser.add_argument("--config_id", type=str, default="exp-002", help="Configuration ID")
args = parser.parse_args()
config_id = args.config_id

config = load_config(config_id)
config_id_trf = config['trf']['config_id']
config_trf = load_config(config_id_trf)
config_id_audio = config['preprocess-audio']['config_id']
config_id_eeg = config['preprocess-eeg']['config_id']

dataset_name = config['dataset']
base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
path_dict = set_paths_from_config(base_path, config)

audio_filename = f"dataset-{dataset_name}_data-audio_config-{config_id_audio}"
audio = np.load(os.path.join(path_dict['features'], f"{audio_filename}.npy"))
eeg_filename = f"dataset-{dataset_name}_data-eeg_config-{config_id_eeg}_-epo"
epochs = mne.read_epochs(os.path.join(path_dict['features'], f"{eeg_filename}.fif"))
dataset = AAD_Dataset()
dataset.create(epochs, audio)
dataset.normalize() if config_trf['normalize'] else None
del audio, epochs

cv = LeaveOneGroupOut()
groups = dataset.sub_ids
search_space = np.logspace(config_trf['search_space'][0], 
                        config_trf['search_space'][1], 
                        config_trf['search_space'][2]
                        )
prediction = []
for train_index, test_index in cv.split(dataset.labels, groups=groups):
    train_dataset = dataset[train_index]
    test_dataset = dataset[test_index]
    test_sub_id = test_dataset.sub_ids[0]
    print(f"Test subject: {test_sub_id}")

    trf = TRF(direction=config_trf['direction'], 
            delays=tuple(config_trf['delays']), 
            scoring=config_trf['scoring']
            )
    trf.optimize_hyperparmeters(train_dataset, 
                                search_space, 
                                n_folds=config_trf['n_folds']
                                )
    trf.train(train_dataset)
    trf_filename = f"dataset-{dataset_name}_models-trf_config-{config_id_trf}_sub-{test_sub_id}"
    trf.save(os.path.join(path_dict['models'], f"{trf_filename}.pkl"))
    # trf = TRF.load(os.path.join(path_dict['models'], f"{trf_filename}.pkl"))
    prediction.append(trf.predict(test_dataset))

    dataset = trf.parse_scores_as_features(dataset)
    dataset_filename = f"dataset-{dataset_name}_data-aad-trfscores_config-{config_id_trf}_sub-{test_sub_id}"
    # dataset = AAD_Dataset.load_from_file(os.path.join(path_dict['features'], f"{dataset_filename}.pkl"))
    dataset.save(os.path.join(path_dict['features'], f"{dataset_filename}.pkl"))
    
    if trf.direction == 'forward':
        plot_list = ['coef', 'scores', 'hp-tuning', 'prediction']
    elif trf.direction == 'backward':
        plot_list = ['coef', 'hp-tuning', 'prediction']
    
    for type in plot_list:
        print(f"Plotting {type}...")
        plot_channels = ['FCz']
        plot_channels_str = "".join(plot_channels)
        fig_filename = f"dataset-{dataset_name}_reports-trf-{type}_config-{config_id_trf}_sub-{test_sub_id}_ch-{plot_channels_str}"
        ax = trf.plot(type=type, 
                        dataset=test_dataset,
                        plot_channels=plot_channels,
                        save=True,
                        save_path=os.path.join(path_dict['reports'], f"{fig_filename}.png")
                        )

# Grand Average EEG - TRF prediction
prediction = np.concatenate(prediction, axis=1)
fig_filename = f"dataset-{dataset_name}_reports-trf-prediction_config-{config_id_trf}_sub-grand-average_ch-{plot_channels_str}"
save_path = os.path.join(path_dict['reports'], f"{fig_filename}.png")
if trf.direction == 'forward':
    axs = plot_eeg_prediction(dataset,
                            prediction,
                            plot_channels=plot_channels,
                            scaling_factor=1.5
                            )
elif trf.direction == 'backward':
    axs = plot_audio_prediction(dataset, prediction)
plt.savefig(save_path)