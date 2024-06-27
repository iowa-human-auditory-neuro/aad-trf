import glob
import numpy as np
from scipy.io import wavfile
from scipy.signal import hilbert, resample
from math import ceil
import matplotlib.pyplot as plt

class AudioPreprocessor:
    def __init__(self, config:dict):
        self.config = config
        self.config_audio = config['audio_preprocessing']

    def load_audio_data(self, audio_path:str) -> tuple[np.ndarray, float]:
        audio_files = sorted(glob.glob(f"{audio_path}/*.wav"))
        audio_list = []
        for audio_file in audio_files:
            sfreq_audio, audio = wavfile.read(audio_file)
            audio = audio.T.astype(np.float64)
            if audio.ndim == 1:
                audio = audio[np.newaxis, :]
            audio_list.append(audio)
        self.sfreq_audio = sfreq_audio
        audio_array = np.stack(audio_list, axis=0)  # Shape: (n_files, n_channels, n_samples)
        
        return audio_array, sfreq_audio
    
    def crop(self, audio:np.ndarray, crop_time:tuple) -> np.ndarray:
        crop_from_idx = ceil(self.sfreq_audio * crop_time[0]) if crop_time[0] is not None else 0
        crop_until_idx = ceil(self.sfreq_audio * crop_time[1]) if crop_time[1] is not None else None
        
        return audio[:,:,crop_from_idx:crop_until_idx]
    
    def downsample(self, audio:np.ndarray, downsfreq:float) -> np.ndarray:
        audio_duration = audio.shape[-1] / self.sfreq_audio
        self.sfreq_audio = downsfreq
        
        return resample(audio, ceil(downsfreq * audio_duration), axis=-1)

    def extract_envelope(self, audio:np.ndarray) -> np.ndarray:
        return np.abs(hilbert(audio))
    
    def run(self, audio:np.ndarray) -> np.ndarray:
        downsfreq = self.config_audio['downsfreq']
        crop_time = tuple(self.config_audio['crop_time'])

        audio = self.extract_envelope(audio)

        if downsfreq:
            audio = self.downsample(audio, downsfreq) 
        if any(crop_time):
            audio = self.crop(audio, crop_time)

        return audio
    
    def plot(self, audio:np.ndarray):
        fig, axs = plt.subplots(audio.shape[0], audio.shape[1])
        times = np.arange(audio.shape[-1]) / self.sfreq_audio
        for i in range(audio.shape[0]):
            for j in range(audio.shape[1]):
                axs[i,j].plot(times, audio[i,j,:])
        
        return axs

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

    audio_preprocessor = AudioPreprocessor(config)
    audio_array, sfreq_audio = audio_preprocessor.load_audio_data(path_dict['audio'])
    print(audio_array.shape)
    audio = audio_preprocessor.run(audio_array)
    print(audio.shape)
    axs = audio_preprocessor.plot(audio)