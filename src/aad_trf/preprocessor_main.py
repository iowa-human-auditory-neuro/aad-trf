import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from audio_preprocessor import AudioPreprocessor
from eeg_preprocessor import EEG_Preprocessor
from utils import load_config, set_paths_from_config

parser = argparse.ArgumentParser(description="Preprocess audio and eeg files")
parser.add_argument("--dataset", type=str, default="updown-ci", help="dataset name")
parser.add_argument("--config_id_audio", type=str, default="audio-002", help="Configuration ID of audio")
parser.add_argument("--config_id_eeg", type=str, default="eeg-003", help="Configuration ID of eeg")

args = parser.parse_args()
config_id_audio = args.config_id_audio
config_id_eeg = args.config_id_eeg
dataset_name = args.dataset
config = {'dataset': dataset_name}

base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
path_dict = set_paths_from_config(base_path, config)

config_audio = load_config(config_id_audio)
audio_preprocessor = AudioPreprocessor(config_audio)
audio_array, sfreq_audio = audio_preprocessor.load_audio_data(path_dict['audio'])
audio = audio_preprocessor.run(audio_array)
print(audio.shape)
axs = audio_preprocessor.plot(audio)
audio_filename = f"dataset-{dataset_name}_data-audio_config-{config_id_audio}"
fig_filename_audio = f"dataset-{dataset_name}_reports-prep-audio-waveform_config-{config_id_audio}"
np.save(os.path.join(path_dict['features'], f"{audio_filename}.npy"), audio)
plt.savefig(os.path.join(path_dict['reports'], f"{fig_filename_audio}.png"))
plt.close()

config_eeg = load_config(config_id_eeg)
eeg_preprocessor = EEG_Preprocessor(config_eeg)
epochs = eeg_preprocessor.load_epochs(path_dict['epochs'], interpolate_bads=True)
epochs = eeg_preprocessor.run(epochs)
epochs = epochs.shift_time(-0.5, relative=True)
epochs_filename = f"dataset-{dataset_name}_data-eeg_config-{config_id_eeg}_-epo"
epochs.save(os.path.join(path_dict['features'], f"{epochs_filename}.fif"), overwrite=True)
print(epochs)
fig = epochs['up'].average().plot(show=False)
fig_filename_up = f"dataset-{dataset_name}_reports-grand-avg-up_config-{config_id_eeg}"
plt.savefig(os.path.join(path_dict['reports'], f"{fig_filename_up}.png"))
plt.close(fig)
fig = epochs['down'].average().plot(show=False)
fig_filename_down = f"dataset-{dataset_name}_reports-grand-avg-down_config-{config_id_eeg}"
plt.savefig(os.path.join(path_dict['reports'], f"{fig_filename_down}.png"))
plt.close(fig)