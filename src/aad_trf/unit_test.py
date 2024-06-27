
def test_utils():
    from utils import load_config, set_paths_from_config

    config = load_config("dataset-updown-nh_exp-1")
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    path_dict = set_paths_from_config(base_path, config)
    print(path_dict)

def test_audio_preprocessor():
    from audio_preprocessor import AudioPreprocessor

    audio_preprocessor = AudioPreprocessor(config)
    audio_array, sfreq_audio = audio_preprocessor.load_audio_data(path_dict['audio'])
    print(audio_array.shape)
    audio = audio_preprocessor.run(audio_array)
    print(audio.shape)
    axs = audio_preprocessor.plot(audio)

def test_eeg_preprocessor():
    from eeg_preprocessor import EEG_Preprocessor

    eeg_preprocessor = EEG_Preprocessor(config)
    epochs = eeg_preprocessor.load_epochs(path_dict['epochs'])
    epochs = eeg_preprocessor.run(epochs)
    print(epochs)
    epochs['up'].average().plot()
    epochs['down'].average().plot()

def test_aad_dataset():
    import numpy as np
    from mne import read_epochs
    from aad_dataset import AAD_Dataset

    audio = np.load(os.path.join(path_dict['features'], f"{config_id}_audio.npy"))
    epochs = read_epochs(os.path.join(path_dict['features'], f"{config_id}-epo.fif"))
    dataset = AAD_Dataset()
    dataset.create(epochs, audio)

def test_aad_trf():
    from aad_trf import TRF
    pass

def test_aad_classifier():
    from aad_classifier import AAD_Classifier
    pass

if __name__ == "__main__":
    import os
    import argparse
    from utils import load_config, set_paths_from_config

    parser = argparse.ArgumentParser(description="Preprocess EEG epochs file. Author: Jusung Ham")
    parser.add_argument("--config_id", type=str, default="dataset-updown-nh_exp-1", help="Configuration ID")
    args = parser.parse_args()
    config_id = args.config_id

    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    config = load_config(config_id)
    path_dict = set_paths_from_config(base_path, config)