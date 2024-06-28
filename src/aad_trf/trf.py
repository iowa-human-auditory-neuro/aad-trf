import numpy as np
import os
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.linalg import norm
from tqdm import tqdm
# from mne import Epochs, read_epochs
from mne.decoding import ReceptiveField
from sklearn.model_selection import KFold, LeaveOneGroupOut

from aad_dataset import AAD_Dataset
from aad_plotter import *

class TRF:
    def __init__(self, 
                 model:ReceptiveField=None, 
                 direction='forward', 
                 delays=(-0.2,0.4), 
                 scoring='corrcoef',
                 lambda_=None
                 ):
        self.model = model
        self.direction = direction
        self.delays = delays
        self.scoring = scoring
        self.lambda_ = lambda_
        
    def _get_input_output(self, dataset:AAD_Dataset):
        if self.direction == 'forward':
            input_data, output_data = dataset.get_attended_audio(moveaxis=True), dataset.get_eeg(moveaxis=True)
        elif self.direction == 'backward':
            input_data, output_data = dataset.get_eeg(moveaxis=True), dataset.get_attended_audio(moveaxis=True)
        else: 
            raise ValueError(f"Invalid direction {self.direction}")
        
        return input_data, output_data
    
    def optimize_hyperparmeters(self,
                                dataset:AAD_Dataset,
                                search_space:list, 
                                n_folds:int=10
                                ):
        print(f"Optimizing hyperparameter among {search_space} using {n_folds}-fold cv")
        print(f"Time delays for TRF: {self.delays}")
        cv = KFold(n_splits=n_folds, shuffle=True) # inner cv for lambda optimization
        optimization_result = pd.DataFrame()

        for lambda_ in search_space:
            scores = []
            print(f"Lambda: {lambda_:.2e}")
            for train_index, valid_index in cv.split(dataset.labels):
                train_dataset = dataset[train_index]
                valid_dataset = dataset[valid_index]
                # eeg_train, audio_train = train_dataset.eeg, train_dataset.audio
                # eeg_valid, audio_valid = valid_dataset.eeg, valid_dataset.audio
                # print(eeg_train.shape, audio_train.shape, eeg_valid.shape, audio_valid.shape)
                # move time axis to the first dimension
                # eeg_train, eeg_valid, audio_train, audio_valid = [
                #     np.moveaxis(arr, -1, 0) for arr in (eeg_train, eeg_valid, audio_train, audio_valid)
                # ]
                # print(eeg_train.shape, audio_train.shape, eeg_valid.shape, audio_valid.shape)
                model = ReceptiveField(tmin=self.delays[0], 
                                    tmax=self.delays[1], 
                                    sfreq=dataset.sfreq, 
                                    estimator=lambda_, 
                                    scoring=self.scoring
                                    )
                input_train, output_train = self._get_input_output(train_dataset)
                input_valid, output_valid = self._get_input_output(valid_dataset)
                model.fit(input_train, output_train)
                scores.append(model.score(input_valid, output_valid).mean()) # average score across channels
            optimization_result[lambda_] = scores
        # print(optimization_result)
        self._optimization_result = optimization_result
        self.lambda_ = optimization_result.mean().idxmax()
        print(f"Best lambda: {self.lambda_}")

    def train(self, dataset:AAD_Dataset):
        print(f"Training TRF with lambda={self.lambda_}")
        model = ReceptiveField(tmin=self.delays[0], 
                            tmax=self.delays[1], 
                            sfreq=dataset.sfreq, 
                            estimator=self.lambda_, 
                            scoring=self.scoring
                            )
        input_train, output_train = self._get_input_output(dataset)
        model.fit(input_train, output_train)

        self.model = model
    
    def eval(self, dataset:AAD_Dataset):
        print(f"Evaluating TRF on {len(dataset)} trials...")
        input_data, output_data = self._get_input_output(dataset)
        
        scores = []
        for trial in tqdm(range(output_data.shape[1])):
            score = self.model.score(input_data[:,trial,:], output_data[:,trial,:])
            scores.append(score)
        return np.stack(scores, axis=0) # (trials, channels)
    
    def eval_fast(self, dataset:AAD_Dataset):
        """
        WARNING: This method is not fully tested. Do not use this method under this version.
        Faster version of eval using vectorized computation.
        """
        input_data, output_data = self._get_input_output(dataset)
        predicted_output = self.model.predict(input_data)
        return self.pearson_corr_3d(predicted_output, output_data, axis=0)

    def eval_external_audio(self, dataset:AAD_Dataset, audio):
        print(f"Evaluating TRF on {len(dataset)} trials...")
        _, output_data = self._get_input_output(dataset)
        
        scores = []
        for trial in tqdm(range(output_data.shape[1])):
            up_score = self.model.score(audio[0, :, np.newaxis], output_data[:,trial,:])
            down_score = self.model.score(audio[1, :, np.newaxis], output_data[:,trial,:])
            scores.append([up_score, down_score])
        final_score = np.stack(scores, axis=0) # (trials, up/down, channels)
        return final_score.reshape(final_score.shape[0], -1)
    
    def predict(self, dataset:AAD_Dataset):
        input_data, output_data = self._get_input_output(dataset)
        return self.model.predict(input_data)
    
    def get_model_coef(self):
        return self.model.coef_
    
    def get_delays_in_sec(self):
        return self.model.delays_ / float(self.model.sfreq)
    
    def parse_scores_as_features(self, dataset:AAD_Dataset, audio):
        # dataset.features = self.eval(dataset)
        dataset.features = self.eval_external_audio(dataset, audio)        
        return dataset

    def save(self, path:str):
        print(f"Saving TRF object to {path}")
        with open(path, "wb") as f:
            pkl.dump(self, f)

    def plot(self, 
             dataset:AAD_Dataset,
             type:str='coef', 
             plot_channels:list=None,
             save:bool=False,
             save_path:str=None
             ):
        if type == 'coef':
            coef = self.get_model_coef()
            times = self.get_delays_in_sec()
            ax = plot_trf_coef(coef, 
                               times, 
                               self.delays, 
                               dataset,
                               plot_channels=plot_channels
                               )
        elif type == 'scores':
            ax = plot_scores_topo(self.eval(dataset).mean(axis=0), 
                                  dataset.eeg_info
                                  )
        elif type == 'hp-tuning':
            ax = plot_lambda_optimization(self._optimization_result)
        elif type == 'prediction':
            ax = plot_eeg_prediction(dataset, 
                                     self.predict(dataset),
                                     plot_channels=plot_channels
                                     )
        else:
            raise ValueError(f"Invalid plot type {type}")
        
        plt.savefig(save_path) if save else plt.show()
        
        return ax
    
    @classmethod
    def load(cls, path:str):
        print(f"Loading TRF object from {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not found.")
        else:
            with open(path, "rb") as f:
                return pkl.load(f)

    @staticmethod
    def pearson_corr_3d(x, y, axis=0):
        """
        WARNING: This method is under development. Do not use this method under this version.
        Compute Pearson correlation coefficient between 3D arrays x and y.
        """
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        xm = x - np.mean(x, axis=axis)
        ym = y - np.mean(y, axis=axis)
        normxm = norm(xm, axis=axis)
        normym = norm(ym, axis=axis)

        return np.sum(xm * ym, axis=axis) / (normxm * normym)

if __name__ == '__main__':
    import os
    import argparse

    from utils import load_config, set_paths_from_config

    # Test the AAD_Dataset class
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
    for train_index, test_index in cv.split(dataset.labels, groups=groups):
        train_dataset = dataset[train_index]
        test_dataset = dataset[test_index]
        test_sub_id = test_dataset.sub_ids[0]
        print(f"Test subject: {test_sub_id}")

        # trf = TRF(direction=config_trf['direction'], 
        #           delays=tuple(config_trf['delays']), 
        #           scoring=config_trf['scoring'], 
        #           lambda_=1000
        #           )
        # # trf.optimize_hyperparmeters(train_dataset, 
        # #                             search_space, 
        # #                             n_folds=config_trf['n_folds']
        # #                             )
        # trf.train(train_dataset)
        # trf.save(os.path.join(path_dict['models'], f"{config_id}_models-trf_sub-{test_sub_id}.pkl"))
        trf = TRF.load(os.path.join(path_dict['models'], f"{config_id}_models-trf_sub-{test_sub_id}.pkl"))
        # scores = trf.eval(test_dataset)
        # dataset = trf.parse_scores_as_features(dataset)
        # dataset.save(os.path.join(path_dict['features'], f"{config_id}_data-aad-trfscores_sub-{test_sub_id}.pkl"))
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
        break
        