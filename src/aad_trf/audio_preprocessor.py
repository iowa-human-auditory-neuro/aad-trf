import glob
import numpy as np
from scipy.io import wavfile
from scipy.signal import hilbert, resample
from math import ceil
import matplotlib.pyplot as plt

class AudioPreprocessor:
    def __init__(self, config:dict):
        self.config = config

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
        downsfreq = self.config['downsfreq']
        crop_time = tuple(self.config['crop_time'])

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
    pass