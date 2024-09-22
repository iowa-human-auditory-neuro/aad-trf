import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import mne
from sklearn.model_selection import LeaveOneGroupOut

from aad_dataset import AAD_Dataset
from trf import TRF
from utils import load_config, set_paths_from_config
from aad_plotter import plot_eeg_prediction

parser = argparse.ArgumentParser()
parser.add_argument("--config_id", plot=str, default="trf-001", help="Configuration ID for TRF")
args = parser.parse_args()
config_id = args.config_id

config = load_config(config_id)
config_id_audio = config['config_id_audio']
config_id_eeg = config['config_id_eeg']

dataset_name = config['dataset']
base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
path_dict = set_paths_from_config(base_path, config)

audio_filename = f"dataset-{dataset_name}_data-audio_config-{config_id_audio}"
audio = np.load(os.path.join(path_dict['features'], f"{audio_filename}.npy"))
eeg_filename = f"dataset-{dataset_name}_data-eeg_config-{config_id_eeg}_-epo"
epochs = mne.read_epochs(os.path.join(path_dict['features'], f"{eeg_filename}.fif"))
dataset = AAD_Dataset()
dataset.create(epochs, audio)
dataset.normalize() if config['normalize'] else None
del audio, epochs

cv = LeaveOneGroupOut()
groups = dataset.sub_ids
search_space = np.logspace(config['search_space'][0], 
                        config['search_space'][1], 
                        config['search_space'][2]
                        )
eeg_prediction = []
for train_index, test_index in cv.split(dataset.labels, groups=groups):
    train_dataset = dataset[train_index]
    test_dataset = dataset[test_index]
    test_sub_id = test_dataset.sub_ids[0]
    print(f"Test subject: {test_sub_id}")

    trf = TRF(direction=config['direction'], 
            delays=tuple(config['delays']), 
            scoring=config['scoring']
            )
    trf.optimize_hyperparmeters(train_dataset, 
                                search_space, 
                                n_folds=config['n_folds']
                                )
    trf.train(train_dataset)
    trf_filename = f"dataset-{dataset_name}_models-trf_config-{config_id}_sub-{test_sub_id}"
    trf.save(os.path.join(path_dict['models'], f"{trf_filename}.pkl"))
    eeg_prediction.append(trf.predict(test_dataset))

    dataset = trf.parse_scores_as_features(dataset)
    dataset_filename = f"dataset-{dataset_name}_data-aad-trfscores_config-{config_id}_sub-{test_sub_id}"
    dataset.save(os.path.join(path_dict['features'], f"{dataset_filename}.pkl"))

    if trf.direction == 'forward':
        plots = ['coef-waveform', 'coef-topo', 'scores', 'hp-tuning', 'prediction']
    else:
        plots = ['coef-waveform', 'coef-topo', 'hp-tuning', 'prediction']

    for plot in plots:
        print(f"Plotting {plot}...")
        plot_channels = ['FCz']
        plot_channels_str = "".join(plot_channels)
        plot_delays = (0.125,0.175) if trf.direction=='forward' else (-0.175,-0.125)
        fig_filename = f"dataset-{dataset_name}_reports-trf-{plot}_config-{config_id}_sub-{test_sub_id}_ch-{plot_channels_str}"
        ax = trf.plot(type=plot, 
                        dataset=test_dataset,
                        plot_channels=plot_channels,
                        delays=plot_delays,
                        save=True,
                        save_path=os.path.join(path_dict['reports'], f"{fig_filename}.png")
                        )

# Grand Average EEG - TRF prediction
eeg_prediction = np.concatenate(eeg_prediction, axis=1)
fig_filename = f"dataset-{dataset_name}_reports-trf-prediction_config-{config_id}_sub-grand-average_ch-{plot_channels_str}"
save_path = os.path.join(path_dict['reports'], f"{fig_filename}.png")
axs = plot_eeg_prediction(dataset,
                          eeg_prediction,
                          plot_channels=plot_channels,
                          scaling_factor=1.5
                          )
plt.savefig(save_path)