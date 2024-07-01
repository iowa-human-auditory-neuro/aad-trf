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
parser.add_argument("--config_id", type=str, default="dataset-updown-nh_exp-1", help="Configuration ID")
args = parser.parse_args()
config_id = args.config_id

base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
config = load_config(config_id)
config_trf = config['trf']
path_dict = set_paths_from_config(base_path, config)

audio = np.load(os.path.join(path_dict['features'], f"{config_id}_data-audio.npy"))
epochs = mne.read_epochs(os.path.join(path_dict['features'], f"{config_id}_data-eeg-epo.fif"))
dataset = AAD_Dataset()
dataset.create(epochs, audio)
dataset.normalize() if config['normalize'] else None
del audio, epochs

cv = LeaveOneGroupOut()
groups = dataset.sub_ids
search_space = np.logspace(config_trf['search_space'][0], 
                        config_trf['search_space'][1], 
                        config_trf['search_space'][2]
                        )
eeg_prediction = []
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
    trf.save(os.path.join(path_dict['models'], f"{config_id}_models-trf_sub-{test_sub_id}.pkl"))
    eeg_prediction.append(trf.predict(test_dataset))

    dataset = trf.parse_scores_as_features(dataset)
    dataset.save(os.path.join(path_dict['features'], f"{config_id}_data-aad-trfscores_sub-{test_sub_id}.pkl"))

    for type in ['coef', 'scores', 'hp-tuning', 'prediction']:
        print(f"Plotting {type}...")
        plot_channels = ['FCz']
        plot_channels_str = "".join(plot_channels)
        save_filename = f"{config_id}_reports-trf-{type}_sub-{test_sub_id}_ch-{plot_channels_str}"
        ax = trf.plot(type=type, 
                        dataset=test_dataset,
                        plot_channels=plot_channels,
                        save=True,
                        save_path=os.path.join(path_dict['reports'], f"{save_filename}.png")
                        )

# Grand Average EEG - TRF prediction
eeg_prediction = np.concatenate(eeg_prediction, axis=1)
save_filename = f"{config_id}_reports-trf-prediction_sub-grand-average_ch-{plot_channels_str}"
save_path = os.path.join(path_dict['reports'], f"{save_filename}.png")
axs = plot_eeg_prediction(dataset,
                          eeg_prediction,
                          plot_channels=plot_channels,
                          scaling_factor=1.5
                          )
plt.savefig(save_path)