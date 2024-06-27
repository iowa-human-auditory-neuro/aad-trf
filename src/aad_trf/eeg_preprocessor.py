
import glob
import pandas as pd
from mne import Epochs, read_epochs, concatenate_epochs
from math import ceil

from utils import get_field_from_filename

class EEG_Preprocessor:
    def __init__(self, config):
        self.config = config
        self.config_eeg = config['eeg_preprocessing']
        self.preprocessed = False

    def load_epochs(self, epochs_path) -> Epochs:
        files = sorted(glob.glob(f"{epochs_path}/*epo.fif"))
        epochs_list = []
        remove_bads = "n"
        for file in files:
            filename = file.split("/")[-1]
            sub_id = get_field_from_filename(filename, "sub")
            assert len(sub_id) == 6, f"Invalid subID: {sub_id}"
            epochs = read_epochs(file)
            if epochs.info["bads"]:
                if remove_bads == "n":
                    remove_bads = input("Bad channels info are present. Do you want to remove this info? (y/n): ")
                if remove_bads == "y":
                    epochs.info["bads"] = []
                elif remove_bads != "n":
                    raise ValueError("Invalid input")

            if epochs.metadata is None:
                epochs.metadata = pd.DataFrame({"sub_id": [sub_id]*len(epochs)})
            elif "sub_id" not in epochs.metadata.columns:
                epochs.metadata["sub_id"] = [sub_id]*len(epochs)
            epochs_list.append(epochs)

        return concatenate_epochs(epochs_list)

    def rereference(self, epochs: Epochs, method) -> Epochs:
        if method == "mastoids":
            if epochs.info['nchan'] == 32:
                ref = ['P7', 'P8']
            elif epochs.info['nchan'] == 64:
                ref = ['P9', 'P10']
            ref = [ch for ch in ref if ch in epochs.ch_names]
        elif method == "average":
            ref = 'average'
        elif method in epochs.ch_names:
            ref = method
        else:
            ref = []

        return epochs.set_eeg_reference(ref)
    
    def run(self, epochs: Epochs) -> Epochs:
        self.rereferencing = self.config_eeg['rereferencing']
        self.baseline = tuple(self.config_eeg['baseline'])
        self.cutoff_freq = tuple(self.config_eeg['cutoff_freq'])
        self.crop_time = tuple(self.config_eeg['crop_time'])
        self.downsfreq = self.config_eeg['downsfreq']

        if self.rereferencing:
            epochs = self.rereference(epochs, self.rereferencing)
        epochs = epochs.apply_baseline(baseline=self.baseline)
        if any(self.cutoff_freq):
            epochs = epochs.filter(l_freq=self.cutoff_freq[0], h_freq=self.cutoff_freq[1])
        if any(self.crop_time):
            epochs = epochs.crop(tmin=self.crop_time[0], tmax=self.crop_time[1], include_tmax=False)
        if self.downsfreq is not None:
            epochs = epochs.decimate(ceil(epochs.info['sfreq']/self.downsfreq))

        self.preprocessed = True

        return epochs

if __name__ == "__main__":
    import os
    import argparse
    from utils import load_config, set_paths_from_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_id", type=str, default="dataset-updown-nh_exp-1", help="Configuration ID")
    args = parser.parse_args()
    config_id = args.config_id

    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    config = load_config(config_id)
    path_dict = set_paths_from_config(base_path, config)

    eeg_preprocessor = EEG_Preprocessor(config)
    epochs = eeg_preprocessor.load_epochs(path_dict['epochs'])
    epochs = eeg_preprocessor.run(epochs)
    print(epochs)
    epochs['up'].average().plot()
    epochs['down'].average().plot()
    