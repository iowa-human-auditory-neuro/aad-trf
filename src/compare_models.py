import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import mne

from aad_trf.utils import *

def parse_clf_results(base_path, config_id:str) -> pd.DataFrame:
    config = load_config(config_id)
    config_id_trf = config['config_id_trf']
    config_trf = load_config(config_id_trf)
    dataset_name = config_trf['dataset']
    path_dict = set_paths_from_config(base_path, config_trf)
    model_name = config['model_name']
    file_name = f"dataset-{dataset_name}_reports-clf-results-by-trial_models-{model_name}_config-{config_id}"
    return pd.read_csv(os.path.join(path_dict['reports'], file_name + ".csv"))

def plot_butterfly(epochs_correct, epochs_incorrect, 
                   ylim:dict=dict(eeg=[-3.5, 3.5]), save=False):
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))
    epochs_correct['up'].average().plot(gfp=True, show=False, ylim=ylim, axes=axs[0, 0])
    epochs_correct['down'].average().plot(gfp=True, show=False, ylim=ylim, axes=axs[0, 1])
    epochs_incorrect['up'].average().plot(gfp=True, show=False, ylim=ylim, axes=axs[1, 0])
    epochs_incorrect['down'].average().plot(gfp=True, show=False, ylim=ylim, axes=axs[1, 1])

    up_onsets = np.linspace(0,4,6)
    up_onsets = up_onsets[1:-1]
    down_onsets = np.linspace(0,4,5)
    down_onsets = down_onsets[1:-1]
    for ax in axs.flat:
        for up_onset in up_onsets:
            ax.axvline(up_onset, color='r', linestyle='--')
        for down_onset in down_onsets:
            ax.axvline(down_onset, color='b', linestyle='--')
        ax.axvline(0, color='k', linestyle='--')
    
    axs[0, 0].set_title('Correct Up')
    axs[0, 1].set_title('Correct Down')
    axs[1, 0].set_title('Incorrect Up')
    axs[1, 1].set_title('Incorrect Down')
    plt.tight_layout()
    
    if save:
        plt.savefig(os.path.join(base_path, 'reports', 'updown-ci_correct_incorrect_channel-all.svg'))
    
    return fig, axs

def plot_correct_incorrect(epochs_correct, epochs_incorrect, 
                           channel=['FCz'], ylim=[-1.6e-6, 1.6e-6], save=False):
    if type(channel) == str:
        channel = [channel]
    up_onsets = np.linspace(0,4,6)
    up_onsets = up_onsets[1:-1]
    down_onsets = np.linspace(0,4,5)
    down_onsets = down_onsets[1:-1]
    fig, axs = plt.subplots(2, 1, figsize=(12, 6))
    for ax, cond in zip(axs, ['up', 'down']):
        evoked_correct_fcz = epochs_correct[cond].average().pick(channel)
        se_correct_fcz = epochs_correct[cond].standard_error().pick(channel)
        ax.plot(evoked_correct_fcz.times, evoked_correct_fcz.data.squeeze(), color='green', label='Correct')
        ax.fill_between(evoked_correct_fcz.times, evoked_correct_fcz.data.squeeze() - se_correct_fcz.data.squeeze(),
                        evoked_correct_fcz.data.squeeze() + se_correct_fcz.data.squeeze(), color='green', alpha=0.2)
        evoked_incorrect_fcz = epochs_incorrect[cond].average().pick(channel)
        se_incorrect_fcz = epochs_incorrect[cond].standard_error().pick(channel)
        ax.plot(evoked_incorrect_fcz.times, evoked_incorrect_fcz.data.squeeze(), color='orange', label='Incorrect')
        ax.fill_between(evoked_incorrect_fcz.times, evoked_incorrect_fcz.data.squeeze() - se_incorrect_fcz.data.squeeze(),
                        evoked_incorrect_fcz.data.squeeze() + se_incorrect_fcz.data.squeeze(), color='orange', alpha=0.2)

        # Plot vertical lines with condition-dependent styles
        if cond == 'up':
            ax.vlines(up_onsets, ymin=ylim[0], ymax=ylim[1], 
                     colors='r', linestyles='--')  # dashed for matching condition
            ax.vlines(down_onsets, ymin=ylim[0], ymax=ylim[1], 
                     colors='b', linestyles=':')   # dotted for other condition
        else:  # cond == 'down'
            ax.vlines(up_onsets, ymin=ylim[0], ymax=ylim[1], 
                     colors='r', linestyles=':')   # dotted for other condition
            ax.vlines(down_onsets, ymin=ylim[0], ymax=ylim[1], 
                     colors='b', linestyles='--')  # dashed for matching condition
        
        ax.axvline(0, color='k', linestyle='--')
        ax.set_title(f'{cond.capitalize()}')
        ax.set_xlim(epochs_correct.tmin, epochs_correct.tmax)
        ax.set_ylim(ylim)
        ax.set_ylabel('Amplitude (uV)')
        ax.set_xlabel('Time (s)')
        ax.legend()
    plt.tight_layout()
    if save:
        channel_names = '-'.join(channel)
        plt.savefig(os.path.join(base_path, 'reports', f'updown-ci_correct_incorrect_channel-{channel_names}.svg'))
    
    return fig, axs

def plot_gfp(epochs_correct, epochs_incorrect, ylim=None, save=False):
    """Plot Global Field Power for correct and incorrect trials
    
    Args:
        epochs_correct (mne.Epochs): Epochs for correct trials
        epochs_incorrect (mne.Epochs): Epochs for incorrect trials
        ylim (tuple, optional): Y-axis limits. Defaults to None.
        save (bool, optional): Save plot to file. Defaults to False.
    
    Returns:
        tuple: (figure, axes)
    """
    up_onsets = np.linspace(0,4,6)
    up_onsets = up_onsets[1:-1]
    down_onsets = np.linspace(0,4,5)
    down_onsets = down_onsets[1:-1]

    fig, axs = plt.subplots(2, 1, figsize=(12, 6))
    
    for ax, cond in zip(axs, ['up', 'down']):
        # Calculate GFP for correct trials
        evoked_correct = epochs_correct[cond].average()
        data_correct = np.std(evoked_correct.data, axis=0)
        
        # Calculate GFP for incorrect trials
        evoked_incorrect = epochs_incorrect[cond].average()
        data_incorrect = np.std(evoked_incorrect.data, axis=0)
        
        # Plot GFP
        ax.plot(evoked_correct.times, data_correct, color='green', label='Correct')
        ax.plot(evoked_incorrect.times, data_incorrect, color='orange', label='Incorrect')

        # Plot vertical lines with condition-dependent styles
        if cond == 'up':
            ax.vlines(up_onsets, ymin=ylim[0], ymax=ylim[1], 
                     colors='r', linestyles='--')  # dashed for matching condition
            ax.vlines(down_onsets, ymin=ylim[0], ymax=ylim[1], 
                     colors='b', linestyles=':')   # dotted for other condition
        else:  # cond == 'down'
            ax.vlines(up_onsets, ymin=ylim[0], ymax=ylim[1], 
                     colors='r', linestyles=':')   # dotted for other condition
            ax.vlines(down_onsets, ymin=ylim[0], ymax=ylim[1], 
                     colors='b', linestyles='--')  # dashed for matching condition
        ax.axvline(0, color='k', linestyle='--')

        ax.set_title(f'{cond.capitalize()}')
        ax.set_xlim(epochs_correct.tmin, epochs_correct.tmax)
        ax.set_ylim(ylim)
        ax.set_ylabel('GFP (µV)')
        ax.set_xlabel('Time (s)')
        ax.legend()

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(base_path, 'reports', 'updown-ci_correct_incorrect_gfp.svg'))
    
    return fig, axs

if __name__ == '__main__':
    base_path = "/Users/jusungham/HANG/aad-trf/"
    results = []
    for clf_config_ids in [6, 9]:
        config_id = f"classifier-{clf_config_ids:03d}"
        results.append(parse_clf_results(base_path, config_id))
    results_df_lr = results[0]
    results_df_svc = results[1]
    prediction_lr = results_df_lr['prediction'].astype(bool)
    prediction_svc = results_df_svc['prediction'].astype(bool)
    agreement = prediction_lr == prediction_svc
    print(f"agreement rate: {agreement.mean():.2f}")
    from sklearn.metrics import cohen_kappa_score
    kappa = cohen_kappa_score(prediction_lr, prediction_svc)
    print(f"Cohen's Kappa: {kappa:.4f}")

    correct_lr = results_df_lr['correct']
    correct_svc = results_df_svc['correct']
    agreement_correct = correct_lr == correct_svc
    print(f"agreement rate (correct): {agreement_correct.mean():.2f}")

    correct_all = correct_lr & correct_svc
    incorrect_all = ~correct_lr & ~correct_svc
    correct_all_idx = correct_all[correct_all].index
    incorrect_all_idx = incorrect_all[incorrect_all].index
    correct_all_idx_sample = np.random.choice(correct_all_idx, len(incorrect_all_idx), replace=False)

    # get index of correct trials
    correct_true_lr = correct_lr[correct_lr] # subset of boolean array
    correct_idx_lr = correct_true_lr.sample(n=len(results_df_lr) - sum(correct_lr), random_state=42).index
    correct_true_svc = correct_svc[correct_svc]
    correct_idx_svc = correct_true_svc.sample(n=len(results_df_svc) - sum(correct_svc), random_state=42).index

    # load AAD_dataset
    from aad_trf.aad_dataset import AAD_Dataset
    config_id_audio = 'audio-002'
    config_id_eeg = 'eeg-003'
    dataset_name = 'updown-ci'
    audio_filename = f"dataset-{dataset_name}_data-audio_config-{config_id_audio}"
    audio = np.load(os.path.join(base_path, 'data', 'features', f"{audio_filename}.npy"))
    eeg_filename = f"dataset-{dataset_name}_data-eeg_config-{config_id_eeg}_-epo"
    epochs = mne.read_epochs(os.path.join(base_path, 'data', 'features', f"{eeg_filename}.fif"))
    dataset = AAD_Dataset()
    dataset.create(epochs, audio)
    del audio, epochs

    epochs_correct = dataset[correct_all_idx_sample].to_epochs()
    epochs_incorrect = dataset[incorrect_all_idx].to_epochs()
    epochs_correct.drop_bad(reject=dict(eeg=100e-6))
    epochs_incorrect.drop_bad(reject=dict(eeg=100e-6))

    fig1, axs1 = plot_butterfly(epochs_correct, epochs_incorrect, ylim=None, save=True)
    fig2, axs2 = plot_correct_incorrect(epochs_correct, epochs_incorrect, channel=['FCz'], save=True)
    fig3, axs3 = plot_gfp(epochs_correct, epochs_incorrect, ylim=[0,1e-6], save=True)
    # evoked = epochs_correct['up'].average()
    # evoked.plot_joint(times=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], title='Correct Up', show=False)
