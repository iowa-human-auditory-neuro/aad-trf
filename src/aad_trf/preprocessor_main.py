import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from audio_preprocessor import AudioPreprocessor
from eeg_preprocessor import EEG_Preprocessor
from utils import load_config, set_paths_from_config

parser = argparse.ArgumentParser(description="Preprocess audio and eeg files")
parser.add_argument("--config_id", type=str, default="dataset-updown-nh_exp-1", help="Configuration ID")
args = parser.parse_args()
config_id = args.config_id

base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
config = load_config(config_id)
path_dict = set_paths_from_config(base_path, config)

audio_preprocessor = AudioPreprocessor(config)
audio_array, sfreq_audio = audio_preprocessor.load_audio_data(path_dict['audio'])
audio = audio_preprocessor.run(audio_array)
print(audio.shape)
axs = audio_preprocessor.plot(audio)
np.save(os.path.join(path_dict['features'], f"{config_id}_data-audio.npy"), audio)
plt.savefig(os.path.join(path_dict['reports'], f"{config_id}_reports-prep-audio-waveform.png"))
plt.close()

eeg_preprocessor = EEG_Preprocessor(config)
epochs = eeg_preprocessor.load_epochs(path_dict['epochs'])
epochs = eeg_preprocessor.run(epochs)
epochs.save(os.path.join(path_dict['features'], f"{config_id}_data-eeg-epo.fif"), overwrite=True)
print(epochs)
fig = epochs['up'].average().plot(show=False)
plt.savefig(os.path.join(path_dict['reports'], f"{config_id}_reports-grand-avg-up.png"))
plt.close(fig)
fig = epochs['down'].average().plot(show=False)
plt.savefig(os.path.join(path_dict['reports'], f"{config_id}_reports-grand-avg-down.png"))
plt.close(fig)