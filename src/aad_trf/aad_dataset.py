import os
import numpy as np
import mne
import pickle as pkl
import matplotlib.pyplot as plt

class AAD_Dataset:

    event2audio_codebook = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]

    def __init__(self, 
                 eeg:np.ndarray=None, 
                 audio:np.ndarray=None, 
                 labels:np.ndarray=None, 
                 sub_ids:np.ndarray=None,
                 features:np.ndarray=None,
                 sfreq:float=None, 
                 eeg_channels:list=None,
                 eeg_info:mne.Info=None
                 ):
        self.eeg = eeg
        self.audio = audio
        self.labels = labels
        self.sub_ids = sub_ids
        self.sfreq = sfreq
        self.eeg_channels = eeg_channels
        self.eeg_info = eeg_info
        self.features = features
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return AAD_Dataset(self.eeg[idx], 
                           self.audio[idx], 
                           self.labels[idx], 
                           self.sub_ids[idx],
                           self.features[idx],
                           self.sfreq,
                           self.eeg_channels,
                           self.eeg_info
                           )
    
    def create(self,
               eeg_epochs:mne.Epochs,
               audio:np.ndarray
               ):
        self.eeg = eeg_epochs.get_data()
        self._event_ids = eeg_epochs.events[:,2]
        self.audio = self.make_audio_array(audio)
        self.sfreq = eeg_epochs.info['sfreq']
        self.eeg_channels = eeg_epochs.ch_names
        self.eeg_info = eeg_epochs.info
        if self.audio.ndim != 3:
            self.audio = self.audio[:, np.newaxis, :] # add channel dimension
        self.labels = np.array([1 - event % 2 for event in self._event_ids]) # 0 for up, 1 for down
        self.sub_ids = eeg_epochs.metadata['sub_id'].values
        self.features = self.eeg

    def normalize(self):
        self.eeg = self.z_score_normalize(self.eeg)
        self.audio = self.z_score_normalize(self.audio)
        self.features = self.z_score_normalize(self.features)

    def make_audio_array(self, audio):
        return np.array([audio[self.event2audio_codebook[event-1]] for event in self._event_ids])

    def get_attended_audio(self,
                           moveaxis=False,
                           normalize=False
                           ):
        attended_audio = []
        for trial in range(len(self)):
            attended_audio.append(self.audio[trial, [self.labels[trial]],:])
        attended_audio = np.array(attended_audio)
        if moveaxis:
            attended_audio = np.moveaxis(audio, -1, 0)
        if normalize:
            attended_audio = self.z_score_normalize(audio)

        return attended_audio
    
    def get_times(self, start_time:float=0.5):
        return (np.arange(self.audio.shape[-1]) / self.sfreq) + start_time
    
    def get_audio(self, 
                  moveaxis=False, 
                  normalize=False
                  ):
        audio = self.audio
        if moveaxis:
            audio = np.moveaxis(audio, -1, 0)
        if normalize:
            audio = self.z_score_normalize(audio)
        
        return audio
    
    def get_eeg(self, 
                moveaxis=False,
                normalize=False
                ):
        eeg = self.eeg
        if moveaxis:
            eeg = np.moveaxis(eeg, -1, 0)
        if normalize:
            eeg = self.z_score_normalize(eeg)

        return eeg
    
    def get_features(self,
                     moveaxis=False,
                     normalize=False
                     ):
        features = self.features
        if moveaxis:
            features = np.moveaxis(features, -1, 0)
        if normalize:
            features = self.z_score_normalize(features)
        
        return features
    
    def save(self, path:str):
        print(f"Saving dataset to {path}")
        with open(path, "wb") as f:
            pkl.dump(self, f)
    
    @classmethod
    def load_from_file(cls, path:str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not found.")
        else:
            with open(path, "rb") as f:
                return pkl.load(f)

    @staticmethod
    def get_channel_positions(eeg_epochs:mne.Epochs):
        poistion_dict = eeg_epochs.get_montage().get_positions()
        xyz_pos = [poistion_dict['ch_pos'][ch] for ch in eeg_epochs.ch_names]
        return np.stack(xyz_pos, axis=0)
    
    @staticmethod
    def z_score_normalize(data, axis=None):
        if axis is None:
            return (data - np.mean(data, axis=-1, keepdims=True)) / np.std(data, axis=-1, keepdims=True)
        else:
            return (data - np.mean(data, axis=axis, keepdims=True)) / np.std(data, axis=axis, keepdims=True)

if __name__ == '__main__':
    import argparse

    from utils import load_config, set_paths_from_config
    from aad_plotter import plot_audio_waveform, plot_eeg_waveform

    # Test the AAD_Dataset class
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_id", type=str, default="dataset-updown-nh_exp-1", help="Configuration ID")
    args = parser.parse_args()
    config_id = args.config_id

    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    config = load_config(config_id)
    path_dict = set_paths_from_config(base_path, config)

    audio = np.load(os.path.join(path_dict['features'], f"{config_id}_data-audio.npy"))
    epochs = mne.read_epochs(os.path.join(path_dict['features'], f"{config_id}_data-eeg-epo.fif"))
    dataset = AAD_Dataset()
    dataset.create(epochs, audio)

    print(len(dataset))
    # generate random index
    random_idx = np.random.randint(0, len(dataset), size=(10,))
    dataset_sub = dataset[random_idx]
    print(len(dataset_sub))
    print(dataset_sub.eeg.shape, 
          dataset_sub.audio.shape, 
          dataset_sub.labels, 
          dataset_sub.sub_ids, 
          dataset_sub.sfreq, 
          dataset_sub.eeg_info,
          dataset_sub.eeg_channels,
          dataset_sub.features.shape
          )
    attended_audio = dataset.get_attended_audio()
    print(attended_audio.shape)
    ax_audio = plot_audio_waveform(dataset)
    ax_eeg = plot_eeg_waveform(dataset)
    pass