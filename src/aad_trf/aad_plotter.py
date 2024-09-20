import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from mne.viz import plot_topomap

from aad_dataset import AAD_Dataset

def select_channels(data:np.ndarray,
                    plot_channels:list,
                    dataset:AAD_Dataset
                    ):
    if plot_channels is str:
        plot_channels = [plot_channels]
    
    if plot_channels is None:
        pass
    elif all(channel in dataset.eeg_channels for channel in plot_channels):
        channel_idx = [dataset.eeg_channels.index(channel) for channel in plot_channels]
        data = data[channel_idx]
    elif plot_channels == 'gfp':
        data = np.std(data, axis=0)
    else:
        raise ValueError(f"Invalid channel name {plot_channels}")
    
    return data

def plot_audio_waveform(dataset:AAD_Dataset):
    audio = dataset.get_audio(attended=True).squeeze()
    audio_up = audio[dataset.labels == 0].mean(axis=0).T
    audio_down = audio[dataset.labels == 1].mean(axis=0).T
    times = dataset.get_times()

    fig, axs = plt.subplots(2,1,figsize=(6,6))
    axs[0].plot(times, audio_up, color='b')
    axs[1].plot(times, audio_down, color='r')
    plt.setp(axs, xlim=(times[0], times[-1]))
    plt.setp(axs, ylabel="Amplitude (A.U.)")
    axs[0].set_title("Up")
    axs[1].set_title("Down")
    fig.supxlabel("Time (s)")
    fig.suptitle("Audio waveform")

    return axs

def plot_eeg_waveform(dataset:AAD_Dataset,
                      plot_channels:list=None
                      ):
    eeg = dataset.get_eeg()
    eeg_up = eeg[dataset.labels == 0].mean(axis=0)
    eeg_down = eeg[dataset.labels == 1].mean(axis=0)
    times = dataset.get_times()

    eeg_up = select_channels(eeg_up, plot_channels, dataset)
    eeg_down = select_channels(eeg_down, plot_channels, dataset)

    eeg_up = eeg_up.T
    eeg_down = eeg_down.T

    fig, axs = plt.subplots(2,1,figsize=(6,6))
    for i, eeg_waveform, title in zip(range(2), (eeg_up, eeg_down), ("Up", "Down")):
        axs[i].plot(times, eeg_waveform, color='k', label="True")
        axs[i].axhline(0, color='k', linestyle='-')
        for up_onset in np.linspace(0,4,6):
            axs[i].axvline(up_onset, color='r', linestyle='--')
        for down_onset in np.linspace(0,4,5):
            axs[i].axvline(down_onset, color='b', linestyle='--')
        axs[i].set_xlim(times[0], times[-1])
        axs[i].set_ylabel("Amplitude (uV)")
        axs[i].set_title(f"{title}")
    fig.supxlabel("Time (s)")
    fig.suptitle(f"EEG waveform {plot_channels}")

    return fig, axs

def plot_trf_coef(coef:np.ndarray,
                  times:np.ndarray,
                  delays:tuple,
                  dataset:AAD_Dataset,
                  plot_channels:list=None,
                  time_step=0.1
                  ):
    coef = select_channels(coef, plot_channels, dataset)
    coef = coef.squeeze()
    coef = coef.T

    fig, ax = plt.subplots(figsize=(7,6))
    ax.plot(times, coef, color='k', marker='.')
    ax.axvline(0, color='k', linestyle=':')
    ax.axhline(0, color='k', linestyle='-')
    ax.set_xlim(delays)
    # ax.set_ylim(-0.06, 0.06)
    ax.set_xticks(np.arange(delays[0], delays[1]+time_step, time_step))
    ax.legend(plot_channels) if len(plot_channels) > 1 else None
    ax.set_title(f"TRF coefficients ({plot_channels})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Coefficient (A.U.)")
    
    return ax

def plot_lambda_optimization(optimization_result:pd.DataFrame):
    optimization_plot = optimization_result.transpose().plot(marker='.',
                                                        figsize=(6, 6),
                                                        color='gray',
                                                        linestyle=':',
                                                        title="Model performance for different regularization parameters",
                                                        xlabel="Regularization parameter",
                                                        ylabel="Score",
                                                        logx=True,
                                                        legend=False
                                                        )
    # plot mean score
    mean_score = optimization_result.mean()
    mean_score.plot(ax=optimization_plot, marker='o', color='black', linestyle='-')

    return optimization_plot

def plot_scores_topo(scores, 
                     eeg_info:mne.Info, 
                     scoring:str='corrcoef'
                     ):
    if scoring == 'corrcoef':
        vlim = (-0.1,0.1)
    elif scoring == 'r2':
        vlim = (0,1)
    
    if set(['FPz', 'Oz', 'T7', 'T8']) <= set(eeg_info['ch_names']):
        sphere = 'eeglab'
    else:
        sphere = 'auto'

    fig, ax = plt.subplots(figsize=(6,6))
    score_topography, _ = plot_topomap(scores, 
                                    pos=eeg_info,
                                    show=False,
                                    vlim=vlim,
                                    sphere=sphere,
                                    axes=ax
                                    )
    plt.colorbar(score_topography, 
                 label=scoring, 
                 orientation='vertical', 
                 shrink=0.5,
                 ticks=[vlim[0], 0, vlim[1]],
                 ax=ax
                 )
    ax.set_title("Model performance across channels")

    return ax

def plot_eeg_prediction(dataset:AAD_Dataset,
                        eeg_prediction:np.ndarray,
                        plot_channels:list=None,
                        scaling_factor=2
                        ):
    eeg_prediction *= scaling_factor
    eeg_up = eeg_prediction[:,dataset.labels == 0,:].mean(axis=1)
    eeg_down = eeg_prediction[:,dataset.labels == 1,:].mean(axis=1)

    eeg_up = select_channels(eeg_up.T, plot_channels, dataset)
    eeg_down = select_channels(eeg_down.T, plot_channels, dataset)

    eeg_up = eeg_up.T
    eeg_down = eeg_down.T
    
    times = dataset.get_times()
    fig, axs = plot_eeg_waveform(dataset, plot_channels=plot_channels)
    axs[0].plot(times, eeg_up, color='r', label="Predicted")
    axs[0].legend(loc='upper right')
    axs[1].plot(times, eeg_down, color='b', label="Predicted")
    axs[1].legend(loc='upper right')
    plt.setp(axs, ylabel="Amplitude (A.U.)")
    fig.suptitle(f"Averaged True and Predicted EEG waveform {plot_channels}")

    return axs

def plot_clf_results(clf_results:pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6,8))
    ax = sns.stripplot(data=clf_results, 
                       x="classifier", 
                       y="accuracy_test", 
                       jitter=False, 
                       color='k',
                       marker='o',
                       ax=ax
                       )
    ax = sns.boxplot(data=clf_results, 
                     x="classifier", 
                     y="accuracy_test", 
                     width=0.1, 
                     color='k',
                     fill=False,
                     ax=ax
                     )
    ax.axhline(0.5, color='gray', linestyle='--')
    ax.grid(axis='y')
    ax.set_ylim(0.2, 0.8)
    ax.set_title("Classifier performance")
    ax.set_ylabel("Accuracy")

    return ax

def plot_importance(importance:np.ndarray,
                    eeg_info:mne.Info, 
                    ):
    importance_half_idx = len(importance) // 2
    importance_up = importance[:importance_half_idx]
    importance_down = importance[importance_half_idx:]
    vlim = (-np.abs(importance).max(), np.abs(importance).max())

    if set(['FPz', 'Oz', 'T7', 'T8']) <= set(eeg_info['ch_names']):
        sphere = 'eeglab'
    else:
        sphere = 'auto'

    fig, axs = plt.subplots(1,2,figsize=(6,4))
    score_topography, _ = plot_topomap(importance_up, 
                                    pos=eeg_info,
                                    show=False,
                                    vlim=vlim,
                                    sphere=sphere,
                                    axes=axs[0]
                                    )
    axs[0].set_title("Up")
    score_topography, _ = plot_topomap(importance_down, 
                                    pos=eeg_info,
                                    show=False,
                                    vlim=vlim,
                                    sphere=sphere,
                                    axes=axs[1]
                                    )
    axs[1].set_title("Down")
    fig.colorbar(score_topography, 
                 orientation='vertical', 
                 shrink=0.5,
                 ticks=[vlim[0], 0, vlim[1]],
                 ax=axs
                 )
    fig.suptitle("Feature importance across channels")

    return axs

if __name__ == '__main__':
    pass