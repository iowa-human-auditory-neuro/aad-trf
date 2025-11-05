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
        
        self.patterns = True if self.direction == 'backward' else False
        
    def _get_input_output(self, dataset:AAD_Dataset, attended=None):
        if self.direction == 'forward':
            input_data = dataset.get_audio(moveaxis=True, attended=attended)
            output_data = dataset.get_eeg(moveaxis=True)
        elif self.direction == 'backward':
            input_data = dataset.get_eeg(moveaxis=True)
            output_data = dataset.get_audio(moveaxis=True, attended=attended)
        else: 
            raise ValueError(f"Invalid direction {self.direction}")
        
        return input_data, output_data
    
    def optimize_hyperparmeters(self,
                                dataset:AAD_Dataset,
                                search_space:list, 
                                n_folds:int=10,
                                attended=None,
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
                                    scoring=self.scoring,
                                    patterns=self.patterns
                                    )
                input_train, output_train = self._get_input_output(train_dataset, attended=attended)
                input_valid, output_valid = self._get_input_output(valid_dataset, attended=attended)
                model.fit(input_train, output_train)
                scores.append(model.score(input_valid, output_valid).mean()) # average score across channels
            optimization_result[lambda_] = scores
        # print(optimization_result)
        self._optimization_result = optimization_result
        self.lambda_ = optimization_result.mean().idxmax()
        print(f"Best lambda: {self.lambda_}")

    def train(self, dataset:AAD_Dataset, attended=None):
        print(f"Training TRF with lambda={self.lambda_}")
        model = ReceptiveField(tmin=self.delays[0], 
                            tmax=self.delays[1], 
                            sfreq=dataset.sfreq, 
                            estimator=self.lambda_, 
                            scoring=self.scoring,
                            patterns=self.patterns
                            )
        input_train, output_train = self._get_input_output(dataset, attended=attended)
        model.fit(input_train, output_train)

        self.model = model

    def eval(self, dataset:AAD_Dataset, attended=None):
        print(f"Evaluating TRF on {len(dataset)} trials...")
        input_data, output_data = self._get_input_output(dataset, attended=attended)

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
    
    def predict(self, dataset:AAD_Dataset, attended=None):
        input_data, output_data = self._get_input_output(dataset, attended=attended)
        return self.model.predict(input_data)
    
    def parse_scores_as_features(self, dataset:AAD_Dataset):
        input_data, output_data = self._get_input_output(dataset, attended=None)
        scores = np.zeros((len(dataset), 2, self.model.coef_.shape[0])) # (trials, up/down, channels)
        if self.direction == 'forward':
            for trial in tqdm(range(len(dataset))):
                scores[trial,0,:] = self.model.score(input_data[:,trial,[0]].repeat(2, axis=1), output_data[:,trial,:])
                scores[trial,1,:] = self.model.score(input_data[:,trial,[1]].repeat(2, axis=1), output_data[:,trial,:])
        elif self.direction == 'backward':
            for trial in tqdm(range(len(dataset))):
                scores[trial,0,:] = self.model.score(input_data[:,trial,:], output_data[:,trial,[0]].repeat(2, axis=1))
                scores[trial,1,:] = self.model.score(input_data[:,trial,:], output_data[:,trial,[1]].repeat(2, axis=1))
        dataset.features = scores.reshape(len(dataset), -1)
        return dataset
    
    def get_model_coef(self):
        if self.direction == 'forward':
            return self.model.coef_[:,0,:]
        elif self.direction == 'backward':
            return self.model.patterns_[0,:,:]
        else:
            raise ValueError(f"Invalid direction {self.direction}. Direction must be either 'forward' or 'backward'.")
    
    def get_delays_in_sec(self):
        return self.model.delays_ / float(self.model.sfreq)
    
    def save(self, path:str):
        print(f"Saving TRF object to {path}")
        with open(path, "wb") as f:
            pkl.dump(self, f)

    def plot(self, 
             dataset:AAD_Dataset,
             plot_type:str='coef', 
             delays:tuple=None,
             plot_channels:list=None,
             save:bool=False,
             save_path:str=None
             ):
        delays = self.delays if delays is None else delays
        if plot_type == 'coef-waveform':
            coef = self.get_model_coef()
            times = self.get_delays_in_sec()
            ax = plot_trf_waveform(coef, 
                                       times, 
                                       self.delays, 
                                       dataset,
                                       plot_channels=plot_channels
                                      )
        elif plot_type == 'coef-topo':
            coef = self.get_model_coef()
            times = self.get_delays_in_sec()
            ax = plot_trf_topo(coef, 
                                times, 
                                delays, 
                                dataset.eeg_info
                                )
        elif plot_type == 'scores':
            ax = plot_scores_topo(self.eval(dataset).mean(axis=0), 
                                  dataset.eeg_info
                                  )
        elif plot_type == 'hp-tuning':
            ax = plot_lambda_optimization(self._optimization_result)
        elif plot_type == 'prediction':
            prediction = self.predict(dataset)
            if self.direction == 'forward':
                ax = plot_eeg_prediction(dataset,
                                    prediction,
                                    plot_channels=plot_channels,
                                    scaling_factor=1.5
                                    )
            elif self.direction == 'backward':
                ax = plot_audio_prediction(dataset, prediction)
        else:
            raise ValueError(f"Invalid plot type {plot_type}")
        
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
    
    def main(args=None):
        import mne

        from utils import load_config, set_paths_from_config

        config_id = args.config_id

        config = load_config(config_id)
        config_id_audio = config['config_id_audio']
        config_id_eeg = config['config_id_eeg']

        dataset_name = config['dataset']
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        path_dict = set_paths_from_config(base_path, config)

        audio_filename = f"dataset-{dataset_name}_data-audio_config-{config_id_audio}"
        audio = np.load(os.path.join(path_dict['features'], f"{audio_filename}.npy"))
        eeg_filename = f"dataset-{dataset_name}_data-eeg_config-{config_id_eeg}_-epo"
        epochs = mne.read_epochs(os.path.join(path_dict['features'], f"{eeg_filename}.fif"))
        dataset = AAD_Dataset()
        dataset.create(epochs, audio)
        dataset.normalize() if config['normalize'] else None
        del audio, epochs

        cv = LeaveOneGroupOut()
        groups = dataset.sub_ids
        search_space = np.logspace(config['search_space'][0], 
                                config['search_space'][1], 
                                config['search_space'][2]
                                )
        prediction = []
        for train_index, test_index in cv.split(dataset.labels, groups=groups):
            train_dataset = dataset[train_index]
            test_dataset = dataset[test_index]
            test_sub_id = test_dataset.sub_ids[0]
            print(f"Test subject: {test_sub_id}")

            trf = TRF.load(os.path.join(path_dict['models'], f"dataset-{dataset_name}_models-trf_config-{config_id}_sub-{test_sub_id}.pkl"))
            # trf = TRF(direction=config['direction'], 
            #         delays=tuple(config['delays']), 
            #         scoring=config['scoring']
            #         )
            # trf.optimize_hyperparmeters(train_dataset, 
            #                             search_space, 
            #                             n_folds=config['n_folds']
            #                             )
            # trf.train(train_dataset)
            # # trf_filename = f"dataset-{dataset_name}_models-trf_config-{config_id}_sub-{test_sub_id}"
            # # trf.save(os.path.join(path_dict['models'], f"{trf_filename}.pkl"))
            # eeg_prediction.append(trf.predict(test_dataset))

            # dataset = trf.parse_scores_as_features(dataset)
            # dataset_filename = f"dataset-{dataset_name}_data-aad-trfscores_config-{config_id}_sub-{test_sub_id}"
            # dataset.save(os.path.join(path_dict['features'], f"{dataset_filename}.pkl"))

            if trf.direction == 'forward':
                plots = ['coef-waveform', 'coef-topo', 'scores', 'hp-tuning', 'prediction']
            else:
                plots = ['coef-waveform', 'coef-topo', 'hp-tuning', 'prediction']

            for plot in plots:
                print(f"Plotting {plot}...")
                plot_channels = ['FCz']
                plot_channels_str = "".join(plot_channels)
                fig_filename = f"dataset-{dataset_name}_reports-trf-{plot}_config-{config_id}_sub-{test_sub_id}_ch-{plot_channels_str}"
                plot_delays = (0.125,0.175) if trf.direction=='forward' else (-0.175,-0.125)
                ax = trf.plot(plot_type=plot, 
                                dataset=test_dataset,
                                delays=plot_delays,
                                plot_channels=plot_channels,
                                save=False,
                                save_path=os.path.join(path_dict['reports'], f"{fig_filename}.png")
                                )

        # Grand Average EEG - TRF prediction
        prediction = np.concatenate(prediction, axis=1)
        # fig_filename = f"dataset-{dataset_name}_reports-trf-prediction_config-{config_id}_sub-grand-average_ch-{plot_channels_str}"
        # save_path = os.path.join(path_dict['reports'], f"{fig_filename}.png")
        if trf.direction == 'forward':
            axs = plot_eeg_prediction(dataset,
                                prediction,
                                plot_channels=plot_channels,
                                scaling_factor=1.5
                                )
        elif trf.direction == 'backward':
            axs = plot_audio_prediction(dataset, prediction)
        # plt.savefig(save_path)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_id", type=str, default="trf-001", help="Configuration ID for TRF")
    args = parser.parse_args()

    TRF.main(args)