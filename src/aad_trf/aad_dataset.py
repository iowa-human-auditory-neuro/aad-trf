import os
import numpy as np
import pandas as pd
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
                 eeg_info:mne.Info=None,
                 times:np.ndarray=None
                 ):
        self.eeg = eeg
        self.audio = audio
        self.labels = labels
        self.sub_ids = sub_ids
        self.features = features
        self.sfreq = sfreq
        self.eeg_channels = eeg_channels
        self.eeg_info = eeg_info
        self.times = times
    
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
                           self.eeg_info,
                           self.times
                           )
    
    def create(self,
               eeg_epochs:mne.Epochs,
               audio:np.ndarray
               ):
        self.eeg = eeg_epochs.get_data()
        self._event_ids = eeg_epochs.events[:,2]
        self.audio = self.make_audio_array(audio)
        if self.audio.ndim != 3:
            self.audio = self.audio[:, np.newaxis, :] # add channel dimension
        self.labels = np.array([1 - event % 2 for event in self._event_ids]) # 0 for up, 1 for down
        self.sub_ids = eeg_epochs.metadata['sub_id'].values
        self.features = self.eeg
        self.sfreq = eeg_epochs.info['sfreq']
        self.eeg_channels = eeg_epochs.ch_names
        self.eeg_info = eeg_epochs.info
        self.times = eeg_epochs.times

    def normalize(self):
        self.eeg = self.z_score_normalize(self.eeg)
        self.audio = self.z_score_normalize(self.audio)
        self.features = self.z_score_normalize(self.features)

    def make_audio_array(self, audio):
        return np.array([audio[self.event2audio_codebook[event-1]] for event in self._event_ids])

    def _get_attended_audio(self):
        attended_audio = []
        for trial in range(len(self)):
            attended_audio.append(self.audio[trial, [self.labels[trial]],:])
        attended_audio = np.array(attended_audio)

        return attended_audio
    
    def _get_ignored_audio(self):
        ignored_audio = []
        for trial in range(len(self)):
            ignored_audio.append(self.audio[trial, [1-self.labels[trial]],:])
        ignored_audio = np.array(ignored_audio)

        return ignored_audio

    def get_times(self, start_time:float=0.5):
        # return (np.arange(self.audio.shape[-1]) / self.sfreq) + start_time
        return self.times
    
    def get_audio(self, 
                  attended=None,
                  moveaxis=False, 
                  normalize=False
                  ):
        if attended is None:
            audio = self.audio
        elif attended == 'mix':
            audio = np.mean(self.audio, axis=1, keepdims=True)
        elif attended:
            audio = self._get_attended_audio()
        else:
            audio = self._get_ignored_audio()
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

    def to_epochs(self) -> mne.Epochs:
        """Convert AAD_Dataset to MNE Epochs object

        Returns:
            mne.Epochs: MNE Epochs object containing the EEG data
        """
        # Create events array
        n_events = len(self)
        events = np.zeros((n_events, 3), dtype=int)
        events[:, 0] = np.arange(n_events)  # Sample indices
        events[:, 2] = self.labels # Event IDs (0 for up, 1 for down)
        
        # Create Epochs object
        epochs = mne.EpochsArray(
            data=self.eeg,
            info=self.eeg_info,
            events=events,
            tmin=self.times[0],
            event_id={'up': 0, 'down': 1}
        )
        
        # Add metadata
        epochs.metadata = pd.DataFrame({
            'sub_id': self.sub_ids
        })
        
        return epochs

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
    parser.add_argument("--dataset", type=str, default="updown-nh", help="dataset name")
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
    audio_filename = f"dataset-{dataset_name}_data-audio_config-{config_id_audio}"
    audio = np.load(os.path.join(path_dict['features'], f"{audio_filename}.npy"))
    config_eeg = load_config(config_id_eeg)
    epochs_filename = f"dataset-{dataset_name}_data-eeg_config-{config_id_eeg}_-epo"
    epochs = mne.read_epochs(os.path.join(path_dict['features'], f"{epochs_filename}.fif"))
    dataset = AAD_Dataset()
    dataset.create(epochs, audio)
    del epochs, audio

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
    attended_audio = dataset.get_audio(attended=True)
    print(attended_audio.shape)
    # ax_audio = plot_audio_waveform(dataset)
    ax_eeg = plot_eeg_waveform(dataset)

    dataset_filename = f"dataset-{dataset_name}_data-aad_config-test_sub-all"
    dataset.save(os.path.join(path_dict['features'], f"{dataset_filename}.pkl"))
    pass