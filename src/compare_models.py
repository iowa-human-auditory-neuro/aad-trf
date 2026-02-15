import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import mne
import seaborn as sns
from tqdm import tqdm
from aad_trf.utils import *
import pickle

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
plt.rc('font', family='Helvetica')

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
                   ylim:dict=dict(eeg=[-3.5, 3.5])):
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
        ax.axvline(0, color='purple', linestyle='--')
        ax.axvline(-0.5, color='k', linestyle=':')
    
    axs[0, 0].set_title('Correct Up')
    axs[0, 1].set_title('Correct Down')
    axs[1, 0].set_title('Incorrect Up')
    axs[1, 1].set_title('Incorrect Down')
    plt.tight_layout()
    
    return fig, axs

def plot_correct_incorrect(epochs_correct, epochs_incorrect, 
                           channel=['FCz'], ylim=[-3.0e-6, 2.5e-6]):
    if type(channel) == str:
        channel = [channel]
    up_onsets = np.linspace(0,4,6)
    up_onsets = up_onsets[1:-1]
    down_onsets = np.linspace(0,4,5)
    down_onsets = down_onsets[1:-1]
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    for ax, cond in zip(axs, ['up', 'down']):
        evoked_correct = epochs_correct[cond].average()
        evoked_correct_gfp = np.std(evoked_correct.data, axis=0)
        evoked_correct_fcz = evoked_correct.pick(channel)
        se_correct_fcz = epochs_correct[cond].standard_error().pick(channel)
        ax.plot(evoked_correct_fcz.times, evoked_correct_fcz.data.squeeze(), 
                color='green', label='Correct', linestyle=(0,(5,1)))
        ax.fill_between(evoked_correct_fcz.times, evoked_correct_fcz.data.squeeze() - se_correct_fcz.data.squeeze(),
                        evoked_correct_fcz.data.squeeze() + se_correct_fcz.data.squeeze(), color='green', alpha=0.2)
        ax.plot(evoked_correct.times, evoked_correct_gfp*0.7 + ylim[0], color='green', linestyle=(0,(5,1)))
        ax.fill_between(evoked_correct.times, [ylim[0]]*len(evoked_correct.times), evoked_correct_gfp*0.7 + ylim[0], 
                        color='green', alpha=0.25)

        evoked_incorrect = epochs_incorrect[cond].average()
        evoked_incorrect_gfp = np.std(evoked_incorrect.data, axis=0)
        evoked_incorrect_fcz = evoked_incorrect.pick(channel)
        se_incorrect_fcz = epochs_incorrect[cond].standard_error().pick(channel)
        ax.plot(evoked_incorrect_fcz.times, evoked_incorrect_fcz.data.squeeze(), 
                color='orange', label='Incorrect', linestyle=(0, (3,1,1,1)))
        ax.fill_between(evoked_incorrect_fcz.times, evoked_incorrect_fcz.data.squeeze() - se_incorrect_fcz.data.squeeze(),
                        evoked_incorrect_fcz.data.squeeze() + se_incorrect_fcz.data.squeeze(), color='orange', alpha=0.2)
        ax.plot(evoked_incorrect.times, evoked_incorrect_gfp*0.7 + ylim[0], color='orange', linestyle=(0, (3,1,1,1)))
        ax.fill_between(evoked_incorrect.times, [ylim[0]]*len(evoked_incorrect.times), evoked_incorrect_gfp*0.7 + ylim[0], 
                        color='orange', alpha=0.25)

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
        
        ax.axvline(0, color='purple', linestyle='--')
        ax.axvline(-0.5, color='k', linestyle=':')
        ax.set_title(f'Task: Attend "{cond.capitalize()}"')
        ax.set_xlim(epochs_correct.tmin, epochs_correct.tmax)
        ax.set_ylim(ylim)
        ax.set_ylabel('Amplitude (µV)')
        ax.set_xlabel('Time (s)')
        ax.legend()
    plt.tight_layout()

    return fig, axs

def plot_correct_incorrect2(epochs_correct, epochs_incorrect, 
                           channel=['FCz'], ylim=[-3.0e-6, 2.5e-6]):
    if type(channel) == str:
        channel = [channel]
    up_onsets = np.linspace(0,4,6)
    up_onsets = up_onsets[1:-1]
    down_onsets = np.linspace(0,4,5)
    down_onsets = down_onsets[1:-1]
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    for ax, epochs in zip(axs, [epochs_correct, epochs_incorrect]):
        evoked_up = epochs['up'].average()
        evoked_up_gfp = np.std(evoked_up.data, axis=0)
        evoked_up_pick = evoked_up.pick(channel)
        se_up_pick = epochs['up'].standard_error().pick(channel)
        ax.plot(evoked_up_pick.times, evoked_up_pick.data.squeeze(), 
                color='tab:red', label='Attend Up', linestyle=(0,(5,1)))
        ax.fill_between(evoked_up_pick.times, evoked_up_pick.data.squeeze() - se_up_pick.data.squeeze(),
                        evoked_up_pick.data.squeeze() + se_up_pick.data.squeeze(), color='tab:red', alpha=0.2)
        ax.plot(evoked_up.times, evoked_up_gfp*0.7 + ylim[0], color='tab:red', linestyle=(0,(5,1)))
        ax.fill_between(evoked_up.times, [ylim[0]]*len(evoked_up.times), evoked_up_gfp*0.7 + ylim[0],
                        color='tab:red', alpha=0.25)

        evoked_down = epochs['down'].average()
        evoked_down_gfp = np.std(evoked_down.data, axis=0)
        evoked_down_pick = evoked_down.pick(channel)
        se_down_pick = epochs['down'].standard_error().pick(channel)
        ax.plot(evoked_down_pick.times, evoked_down_pick.data.squeeze(), 
                color='tab:blue', label='Attend Down', linestyle=(0, (3,1,1,1)))
        ax.fill_between(evoked_down_pick.times, evoked_down_pick.data.squeeze() - se_down_pick.data.squeeze(),
                        evoked_down_pick.data.squeeze() + se_down_pick.data.squeeze(), color='tab:blue', alpha=0.2)
        ax.plot(evoked_down_pick.times, evoked_down_gfp*0.7 + ylim[0], color='tab:blue', linestyle=(0, (3,1,1,1)))
        ax.fill_between(evoked_down_pick.times, [ylim[0]]*len(evoked_down_pick.times), evoked_down_gfp*0.7 + ylim[0], 
                        color='tab:blue', alpha=0.25)

        # Plot vertical lines with condition-dependent styles
        ax.vlines(up_onsets, ymin=ylim[0], ymax=ylim[1], 
                    colors='r', linestyles='--')  # dashed for matching condition
        ax.vlines(down_onsets, ymin=ylim[0], ymax=ylim[1], 
                    colors='b', linestyles='--')   # dashed for other condition
        
        ax.axvline(0, color='purple', linestyle='--')
        ax.axvline(-0.5, color='k', linestyle=':')
        ax.set_xlim(epochs_correct.tmin, epochs_correct.tmax)
        ax.set_ylim(ylim)
    axs[0].set_ylabel('Amplitude (µV)')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_xticks(np.arange(epochs_correct.tmin, epochs_correct.tmax, 0.5))
    axs[0].legend()
    plt.tight_layout()

    return fig, axs

def plot_erp_gfp(epochs, channel=['FCz'], ylim=[-3.0e-6, 2.5e-6]):
    if type(channel) == str:
        channel = [channel]
    up_onsets = np.linspace(0,4,6)
    up_onsets = up_onsets[1:-1]
    down_onsets = np.linspace(0,4,5)
    down_onsets = down_onsets[1:-1]
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    for cond, color in zip(['up', 'down'], ['tab:red', 'tab:blue']):
        evoked = epochs[cond].average()
        evoked_gfp = np.std(evoked.data, axis=0)
        evoked_pick = evoked.pick(channel)
        se_pick = epochs[cond].standard_error().pick(channel)
        ax.plot(evoked_pick.times, evoked_pick.data.squeeze(), 
                color=color, label=f'Attend {cond}')
        ax.fill_between(evoked_pick.times, evoked_pick.data.squeeze() - se_pick.data.squeeze(),
                        evoked_pick.data.squeeze() + se_pick.data.squeeze(), color=color, alpha=0.2)
        ax.plot(evoked.times, evoked_gfp*0.7 + ylim[0], color=color)
        ax.fill_between(evoked.times, [ylim[0]]*len(evoked.times), evoked_gfp*0.7 + ylim[0], 
                        color=color, alpha=0.25)

        # Plot vertical lines with condition-dependent styles
        ax.vlines(up_onsets, ymin=ylim[0], ymax=ylim[1], 
                    colors='r', linestyles='--')  # dashed for matching condition
        ax.vlines(down_onsets, ymin=ylim[0], ymax=ylim[1], 
                    colors='b', linestyles='--')   # dashed for other condition
        
        ax.axvline(0, color='purple', linestyle='--')
        ax.axvline(-0.5, color='k', linestyle=':')
    ax.set_xlim(epochs.tmin, epochs.tmax)
    ax.set_ylim(ylim)
    ax.set_ylabel('Amplitude (µV)')
    ax.set_xlabel('Time (s)')
    ax.legend(loc='upper right')
    # ticks every 0.5s
    ax.set_xticks(np.arange(epochs.tmin, epochs.tmax, 0.5))
    plt.tight_layout()

    return fig, ax


def plot_gfp(epochs_correct, epochs_incorrect, ylim=None):
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
        ax.axvline(-0.5, color='k', linestyle=':')
        ax.axvline(0, color='purple', linestyle='--')

        ax.set_title(f'Task: Attend "{cond.capitalize()}"')
        ax.set_xlim(epochs_correct.tmin, epochs_correct.tmax)
        ax.set_ylim(ylim)
        ax.set_ylabel('GFP (µV)')
        ax.set_xlabel('Time (s)')
        ax.legend()

    plt.tight_layout()
    
    return fig, axs

def plot_gfp_overlay(epochs, ylim=None):
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

    fig, ax = plt.subplots(figsize=(12, 4))

    for cond, color in zip(['up', 'down'], ['green', 'orange']):
        # Calculate GFP for correct trials
        evoked = epochs[cond].average()
        data = np.std(evoked.data, axis=0)
        
        # Plot GFP
        ax.plot(evoked.times, data, color=color, label=f'Attend {cond.capitalize()}')

        # Plot vertical lines with condition-dependent styles
        ax.vlines(up_onsets, ymin=ylim[0], ymax=ylim[1], 
                    colors='r', linestyles='--')  # dashed for matching condition
        ax.vlines(down_onsets, ymin=ylim[0], ymax=ylim[1], 
                    colors='b', linestyles='--')   # dotted for other condition

        ax.axvline(-0.5, color='k', linestyle=':')
        ax.axvline(0, color='purple', linestyle='--')

        ax.set_title(f'Task: Attend "{cond.capitalize()}"')
        ax.set_xlim(epochs.tmin, epochs.tmax)
        ax.set_ylim(ylim)
        ax.set_ylabel('GFP (µV)')
        ax.set_xlabel('Time (s)')
        ax.legend()

    plt.tight_layout()
    
    return fig, ax

def _calculate_ami_single(epochs, up_onsets, down_onsets, window_size):
    """
    Helper function to calculate AMI for a single condition (correct or incorrect).
    
    Returns detailed peak information in addition to the standard AMI values.
    """
    # Calculate ERP values for each condition
    up_evoked = epochs['up'].average()
    down_evoked = epochs['down'].average()
    
    # Get GFP for each evoked response
    up_gfp = np.std(up_evoked.get_data(), axis=0)
    down_gfp = np.std(down_evoked.get_data(), axis=0)
    
    # Get times array
    times = up_evoked.times
    
    # Dictionary to store detailed peak information
    peak_info = {
        'up': {
            'peaks': [],
            'times': [],
            'indices': []
        },
        'down': {
            'peaks': [],
            'times': [],
            'indices': []
        },
        'up_evoked': up_evoked,
        'down_evoked': down_evoked,
        'up_gfp': up_gfp,
        'down_gfp': down_gfp,
        'times': times
    }
    
    # Lists to store peak values
    up_peaks_attend_up = []
    down_peaks_attend_up = []
    up_peaks_attend_down = []
    down_peaks_attend_down = []
    
    # Find peaks for "Up" onsets in both conditions
    for onset in up_onsets:
        # Find indices within the window after onset
        start_time = onset + window_size[0]
        end_time = onset + window_size[1]
        
        # Make sure the times are within the epoch range
        if start_time >= times[0] and end_time <= times[-1]:
            start_idx = np.where(times >= start_time)[0][0]
            end_idx = np.where(times <= end_time)[0][-1]
            
            # Find maximum GFP in the window
            up_max_idx = start_idx + np.argmax(up_gfp[start_idx:end_idx])
            down_max_idx = start_idx + np.argmax(down_gfp[start_idx:end_idx])
            
            up_max = up_gfp[up_max_idx]
            down_max = down_gfp[down_max_idx]
            
            up_peaks_attend_up.append(up_max)
            up_peaks_attend_down.append(down_max)
            
            # Store detailed information
            peak_info['up']['peaks'].append({
                'up_peak': up_max,
                'down_peak': down_max,
                'up_peak_idx': up_max_idx,
                'down_peak_idx': down_max_idx,
                'up_peak_time': times[up_max_idx],
                'down_peak_time': times[down_max_idx],
                'window': (start_time, end_time),
                'window_idx': (start_idx, end_idx)
            })
    
    # Find peaks for "Down" onsets in both conditions
    for onset in down_onsets:
        # Find indices within the window after onset
        start_time = onset + window_size[0]
        end_time = onset + window_size[1]
        
        # Make sure the times are within the epoch range
        if start_time >= times[0] and end_time <= times[-1]:
            start_idx = np.where(times >= start_time)[0][0]
            end_idx = np.where(times <= end_time)[0][-1]
            
            # Find maximum GFP in the window
            up_max_idx = start_idx + np.argmax(up_gfp[start_idx:end_idx])
            down_max_idx = start_idx + np.argmax(down_gfp[start_idx:end_idx])
            
            up_max = up_gfp[up_max_idx]
            down_max = down_gfp[down_max_idx]
            
            down_peaks_attend_up.append(up_max)
            down_peaks_attend_down.append(down_max)
            
            # Store detailed information
            peak_info['down']['peaks'].append({
                'up_peak': up_max,
                'down_peak': down_max,
                'up_peak_idx': up_max_idx,
                'down_peak_idx': down_max_idx,
                'up_peak_time': times[up_max_idx],
                'down_peak_time': times[down_max_idx],
                'window': (start_time, end_time),
                'window_idx': (start_idx, end_idx)
            })
    
    # Calculate Aa - attended condition peaks
    # "Up" peaks for attend "Up" and "Down" peaks for attend "Down"
    aa_up = np.mean(up_peaks_attend_up) if up_peaks_attend_up else 0
    aa_down = np.mean(down_peaks_attend_down) if down_peaks_attend_down else 0
    aa = (aa_up + aa_down) / 2
    
    # Calculate Au - unattended condition peaks
    # "Down" peaks for attend "Up" and "Up" peaks for attend "Down"
    au_up = np.mean(down_peaks_attend_up) if down_peaks_attend_up else 0
    au_down = np.mean(up_peaks_attend_down) if up_peaks_attend_down else 0
    au = (au_up + au_down) / 2
    
    # Calculate AMI
    if aa + au > 0:  # Avoid division by zero
        ami = (aa - au) / (aa + au)
    else:
        ami = 0
    
    # Store the aggregate values
    peak_info['aa_up'] = aa_up
    peak_info['aa_down'] = aa_down
    peak_info['au_up'] = au_up
    peak_info['au_down'] = au_down
    peak_info['aa'] = aa
    peak_info['au'] = au
    peak_info['ami'] = ami
    
    return aa, au, ami, peak_info

def bootstrap_balanced_ami(epochs_A, epochs_B, up_onsets, down_onsets,
                           window_size=(0.05, 0.25), n_bootstraps=1000):
    """
    Compute bootstrapped AMI arrays for two epoch sets by trial-count balancing.
    Returns two arrays of shape (n_bootstraps,).
    """

    idx_up_A = np.where(epochs_A.events[:, 2] == 0)[0]
    idx_down_A = np.where(epochs_A.events[:, 2] == 1)[0]
    idx_up_B = np.where(epochs_B.events[:, 2] == 0)[0]
    idx_down_B = np.where(epochs_B.events[:, 2] == 1)[0]
    n_sub_up = min(len(idx_up_A), len(idx_up_B))
    n_sub_down = min(len(idx_down_A), len(idx_down_B))

    results = []
    for epochs , idx_up, idx_down in [(epochs_A, idx_up_A, idx_down_A), (epochs_B, idx_up_B, idx_down_B)]:
        aa_bs = []
        au_bs = []
        ami_bs = []
        for _ in tqdm(range(n_bootstraps)):
            idx_up_sub = np.random.choice(idx_up, n_sub_up, replace=True)
            idx_down_sub = np.random.choice(idx_down, n_sub_down, replace=True)
            idx = np.concatenate((idx_up_sub, idx_down_sub))
            # subset epochs
            epochs_sub = epochs[idx]
            # compute AMI on subsampled data (aggregated across trials)
            aa, au, ami, peak_info = _calculate_ami_single(epochs_sub, up_onsets, down_onsets, window_size)
            aa_bs.append(aa)
            au_bs.append(au)
            ami_bs.append(ami)
    
        results.append([np.mean(aa_bs), np.mean(au_bs), np.mean(ami_bs), peak_info])
    
    return results[0], results[1]

def calculate_ami(epochs_correct, epochs_incorrect, up_onsets, down_onsets, window_size=(0.16, 0.28), bootstrap:bool=False):
    """
    Calculate Attentional Modulation Index (AMI) for correct and incorrect epochs.
    """
    # Get unique subject IDs if available
    has_subject_ids = hasattr(epochs_correct, 'metadata') and 'sub_id' in epochs_correct.metadata.columns
    
    if has_subject_ids:
        subjects_correct = epochs_correct.metadata['sub_id'].unique()
        subjects_incorrect = epochs_incorrect.metadata['sub_id'].unique()
        subjects = np.unique(np.concatenate((subjects_correct, subjects_incorrect)))
    else:
        # If no subject info, we'll still process the whole dataset as "all"
        subjects = ['all']
    
    # Initialize results dictionary
    results = {
        'subject': [],
        'ami_correct': [],
        'ami_incorrect': [],
        'aa_correct': [],
        'au_correct': [],
        'aa_incorrect': [],
        'au_incorrect': []
    }
    
    # To store detailed peak info
    peak_info = {
        'correct': {},
        'incorrect': {},
        'aggregated': {} # Always include aggregated results
    }
    
    # First calculate aggregated results using all data
    print("Calculating aggregated AMI across all subjects/epochs...")
    if bootstrap:
        print("Using bootstrap balanced AMI calculation for aggregated data...")
        ami_results_correct_agg, ami_results_incorrect_agg = bootstrap_balanced_ami(
            epochs_correct, epochs_incorrect, up_onsets, down_onsets, window_size, n_bootstraps=1000)
        aa_correct_agg, au_correct_agg, ami_correct_agg, peak_info_correct_agg = ami_results_correct_agg
        aa_incorrect_agg, au_incorrect_agg, ami_incorrect_agg, peak_info_incorrect_agg = ami_results_incorrect_agg
    else:
        print("Using standard AMI calculation for aggregated data...")
        aa_correct_agg, au_correct_agg, ami_correct_agg, peak_info_correct_agg = _calculate_ami_single(
            epochs_correct, up_onsets, down_onsets, window_size)
        aa_incorrect_agg, au_incorrect_agg, ami_incorrect_agg, peak_info_incorrect_agg = _calculate_ami_single(
            epochs_incorrect, up_onsets, down_onsets, window_size)
    
    # Store aggregated results
    peak_info['aggregated']['correct'] = peak_info_correct_agg
    peak_info['aggregated']['incorrect'] = peak_info_incorrect_agg
    
    # Store aggregated metrics in results
    results['subject'].append('aggregated')
    results['ami_correct'].append(ami_correct_agg)
    results['ami_incorrect'].append(ami_incorrect_agg)
    results['aa_correct'].append(aa_correct_agg)
    results['au_correct'].append(au_correct_agg)
    results['aa_incorrect'].append(aa_incorrect_agg)
    results['au_incorrect'].append(au_incorrect_agg)
    
    # Then process individual subjects if we have subject IDs
    if has_subject_ids:
        for subject in subjects:
            # Select epochs for the current subject
            query = f'sub_id == "{subject}"'
            try:
                epochs_correct_subj = epochs_correct[query]
                epochs_incorrect_subj = epochs_incorrect[query]
            except (KeyError, ValueError, NameError) as e:
                print(f"Error selecting epochs for subject {subject}: {e}")
                print("Available metadata columns:", epochs_correct.metadata.columns.tolist())
                print("Skipping subject:", subject)
                continue
            
            print(f"Number of epochs for {subject}: {[len(epochs_correct_subj), len(epochs_incorrect_subj)]}")
            
            # Calculate AMI for correct trials
            if bootstrap:
                print(f"Using bootstrap balanced AMI calculation for subject {subject}...")
                ami_results_correct, ami_results_incorrect = bootstrap_balanced_ami(
                    epochs_correct_subj, epochs_incorrect_subj, up_onsets, down_onsets, window_size, n_bootstraps=1000)
                aa_correct, au_correct, ami_correct, peak_info_correct = ami_results_correct
                aa_incorrect, au_incorrect, ami_incorrect, peak_info_incorrect = ami_results_incorrect
            else:
                print(f"Using standard AMI calculation for subject {subject}...")
                aa_correct, au_correct, ami_correct, peak_info_correct = _calculate_ami_single(
                    epochs_correct_subj, up_onsets, down_onsets, window_size)
                aa_incorrect, au_incorrect, ami_incorrect, peak_info_incorrect = _calculate_ami_single(
                    epochs_incorrect_subj, up_onsets, down_onsets, window_size)
            
            # Store results
            results['subject'].append(subject)
            results['ami_correct'].append(ami_correct)
            results['ami_incorrect'].append(ami_incorrect)
            results['aa_correct'].append(aa_correct)
            results['au_correct'].append(au_correct)
            results['aa_incorrect'].append(aa_incorrect)
            results['au_incorrect'].append(au_incorrect)
            
            # Store peak info
            peak_info['correct'][subject] = peak_info_correct
            peak_info['incorrect'][subject] = peak_info_incorrect
    elif subjects[0] == 'all':
        # Handle case where no subject IDs but we still want 'all' entry
        # (Maintaining backward compatibility)
        epochs_correct_subj = epochs_correct
        epochs_incorrect_subj = epochs_incorrect
        
        # Reuse the aggregated results we already calculated
        results['subject'].append('all')
        results['ami_correct'].append(ami_correct_agg)
        results['ami_incorrect'].append(ami_incorrect_agg)
        results['aa_correct'].append(aa_correct_agg)
        results['au_correct'].append(au_correct_agg)
        results['aa_incorrect'].append(aa_incorrect_agg)
        results['au_incorrect'].append(au_incorrect_agg)
        
        # Store the same peak info
        peak_info['correct']['all'] = peak_info_correct_agg
        peak_info['incorrect']['all'] = peak_info_incorrect_agg
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Perform statistical comparison if there are enough subjects
    if len(subjects) > 1 and has_subject_ids:
        # Exclude 'aggregated' row for statistical tests
        indiv_results = results_df[results_df['subject'] != 'aggregated']
        
        from scipy import stats
        # Use Wilcoxon signed-rank test instead of t-test
        w_stat, p_value = stats.wilcoxon(indiv_results['ami_correct'], indiv_results['ami_incorrect'])
        print(f"Statistical comparison of AMI between correct and incorrect trials:")
        print(f"Wilcoxon signed-rank test: W={w_stat:.4f}, p-value: {p_value:.4f}")
        
        # Add columns for difference and significance
        results_df['ami_diff'] = results_df['ami_correct'] - results_df['ami_incorrect']
        results_df['p_value'] = p_value
    
    return results_df, peak_info

def plot_ami_peaks(ami_results, peak_info, epochs_correct, epochs_incorrect, up_onsets, down_onsets, window_size=(0.08, 0.12), ylim=None):
    """
    Plot GFP with highlighted regions showing which peaks are used in AMI calculation.
    Uses pre-computed peak information to avoid redundant calculations.
    """
    # First create the GFP plot
    fig, axs = plot_gfp(epochs_correct, epochs_incorrect, ylim=ylim)
    
    # Always use aggregated results if available, otherwise fallback to first subject
    # or directly use the 'all' subject
    if 'aggregated' in peak_info:
        correct_info = peak_info['aggregated']['correct'] 
        incorrect_info = peak_info['aggregated']['incorrect']
    elif 'all' in peak_info['correct']:
        # Legacy case - use 'all' if it exists
        correct_info = peak_info['correct']['all']
        incorrect_info = peak_info['incorrect']['all']
    else:
        # Last resort - use first available subject
        correct_info = next(iter(peak_info['correct'].values()))
        incorrect_info = next(iter(peak_info['incorrect'].values()))
    
    # For each condition, highlight the regions used to find peaks for AMI
    for ax_idx, (ax, attend_cond) in enumerate(zip(axs, ['up', 'down'])):
        # Use pre-computed peak info
        for correctness, info in [('Correct', correct_info), ('Incorrect', incorrect_info)]:
            # Skip visualization for incorrect trials to avoid clutter if desired
            if correctness != 'Correct':
                continue
                
            # Get the peaks for this attend condition
            attend_peaks = info[attend_cond]['peaks']
            
            # Draw the peak windows and add annotations
            for i, peak_data in enumerate(attend_peaks):
                # Extract window information
                start_time, end_time = peak_data['window']
                
                # Choose color based on stimulus type (up or down)
                color = 'r' if attend_cond == 'up' else 'b'
                
                # Highlight the window
                ax.axvspan(start_time, end_time, color=color, alpha=0.2,
                          label=f"{attend_cond.capitalize()} peak window" if i == 0 else "")
                
                # Get the peak value
                peak_value = peak_data[f'{attend_cond}_peak']
                
                # Mark the peak point
                peak_time = peak_data[f'{attend_cond}_peak_time']
                ax.plot(peak_time, peak_value, f'{color}o', markersize=4)
                
                # Add text annotation with peak value
                peak_y = peak_value * 1.1  # Position above the peak
                if peak_y > ylim[1] * 0.9:
                    peak_y = ylim[1] * 0.85  # Adjust if too high
                
                ax.text(start_time + (end_time-start_time)/2, peak_y,
                       f"{peak_value:.2e}", color=color, fontsize=7, ha='center')
        
        # Update legend
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper right')
        
        # Add textboxes with AMI calculations
        for i, (correctness, info) in enumerate([('Correct', correct_info), ('Incorrect', incorrect_info)]):
            # Calculate AMI components for attended condition
            if attend_cond == 'up':
                aa = info['aa_up']
                au = info['au_up']
            else:  # attend_cond == 'down'
                aa = info['aa_down']
                au = info['au_down']
            
            # Calculate condition-specific AMI
            ami = (aa - au) / (aa + au) if (aa + au) > 0 else 0
            
            # Add textbox
            y_pos = ylim[1] * (0.6 - 0.2 * i)  # Position based on correct/incorrect
            bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            ax.text(0.02, y_pos,
                   f"{correctness} trials:\nAa = {aa:.2e}\nAu = {au:.2e}\nAMI = {ami:.3f}",
                   transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', bbox=bbox_props)
    
    plt.tight_layout()
    
    return fig, axs

def plot_ami_comparison(ami_results, figsize=(4, 7), swarm_kwargs=None, box_kwargs=None, ylim=None):
    """
    Create a plot comparing AMI values between correct and incorrect conditions.
    
    Parameters:
    -----------
    ami_results : pd.DataFrame
        DataFrame containing AMI results with 'ami_correct', 'ami_incorrect', and 'subject' columns
    figsize : tuple, optional
        Figure size (width, height) in inches
    swarm_kwargs : dict, optional
        Additional keyword arguments to pass to sns.swarmplot
    box_kwargs : dict, optional
        Additional keyword arguments to pass to sns.boxplot
        
    Returns:
    --------
    matplotlib.axes.Axes
        The axes object containing the plot for further customization
    """
    # Default kwargs
    if swarm_kwargs is None:
        swarm_kwargs = {}
    if box_kwargs is None:
        box_kwargs = {}
    
    # Separate individual subjects from aggregated results
    individual_results = ami_results[~ami_results['subject'].isin(['aggregated', 'all'])]
    aggregated_results = ami_results[ami_results['subject'].isin(['aggregated', 'all'])]
    
    # Create a long-form dataframe for seaborn plotting (individual subjects only)
    ami_long = pd.DataFrame({
        'AMI': list(individual_results['ami_correct']) + list(individual_results['ami_incorrect']),
        'Condition': ['Correct'] * len(individual_results) + ['Incorrect'] * len(individual_results),
        'Subject': list(individual_results['subject']) * 2
    })
    
    # Create figure and plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.set_style("whitegrid")
    
    # Default swarm plot settings
    default_swarm = {
        'x': 'Condition',
        'y': 'AMI',
        'data': ami_long,
        # 'hue': 'Subject',
        'marker': 'o',
        'size': 8,
        'color': 'black',
        # 'palette': 'deep',
        'alpha': 0.7,
        'legend': False
    }
    # Override defaults with any provided kwargs
    default_swarm.update(swarm_kwargs)
    
    # Draw swarmplot only for individual subjects
    if len(individual_results) > 0:
        swarm = sns.swarmplot(ax=ax, **default_swarm)
    
    # Default box plot settings
    default_box = {
        'x': 'Condition',
        'y': 'AMI',
        'data': ami_long,
        'width': 0.4,
        'color': 'black',
        'fill': False
    }
    # Override defaults with any provided kwargs
    default_box.update(box_kwargs)
    
    # Draw boxplot on top (only for individual subjects)
    if len(individual_results) > 0:
        box = sns.boxplot(ax=ax, **default_box)
    
    # Draw zero line
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    
    # Draw lines connecting points from the same subject across conditions (individuals only)
    if len(individual_results) > 1:  # Only if we have multiple subjects
        # Create a dictionary mapping condition names to x-coordinates
        condition_to_pos = {cond: i for i, cond in enumerate(['Correct', 'Incorrect'])}
        
        # Connect points for each subject
        for subject in individual_results['subject'].unique():
            # Extract this subject's data for each condition
            subject_data = ami_long[ami_long['Subject'] == subject]
            
            if len(subject_data) >= 2:  # Need at least 2 points to draw a line
                # Get the x and y coordinates for the line
                x_coords = [condition_to_pos[cond] for cond in subject_data['Condition']]
                y_coords = subject_data['AMI'].values
                
                # Draw the line connecting points
                ax.plot(x_coords, y_coords, 'k-', alpha=0.3, linewidth=0.7, zorder=1)
    
    # # Add aggregated results as a special marker if they exist
    # if len(aggregated_results) > 0:
    #     # Get positions for correct and incorrect conditions
    #     x_positions = [0, 1]  # 0 for Correct, 1 for Incorrect
        
    #     # Get the aggregated AMI values
    #     agg_row = aggregated_results.iloc[0]
    #     y_values = [agg_row['ami_correct'], agg_row['ami_incorrect']]
        
    #     # Plot as a star marker with a label
    #     ax.plot(x_positions, y_values, 'D', color='red', markersize=10, 
    #             alpha=0.8, label='Aggregated', zorder=10)
        
    #     # Connect with a thicker line
    #     ax.plot(x_positions, y_values, '-', color='red', linewidth=2, 
    #             alpha=0.6, zorder=9)
    
    # Add p-value annotation to the plot
    if 'p_value' in ami_results.columns:
        p_value = ami_results['p_value'].iloc[0]
        # Determine significance level for annotation
        if p_value < 0.001:
            sig_text = '***'
        elif p_value < 0.01:
            sig_text = '**'
        elif p_value < 0.05:
            sig_text = '*'
        else:
            sig_text = f'p={p_value:.3f}'
        
        # # Get y positions for the annotation - adjust to fit within the plot area
        # if len(ami_long) > 0:
        #     y_max = ami_long['AMI'].max()
        #     y_min = ami_long['AMI'].min()
        # else:
        #     # If no individual subjects, use aggregated values
        #     y_values = [agg_row['ami_correct'], agg_row['ami_incorrect']]
        #     y_max = max(y_values)
        #     y_min = min(y_values)
        
        # y_range = y_max - y_min
        # y_pos = y_max + 0.03 * y_range
        
        # # Apply tight_layout first to set proper axes dimensions
        # plt.tight_layout()
        
        # # Get current axis limits
        # y_top = ax.get_ylim()[1]
        
        # # If our calculated y_pos would be outside the visible plot area, adjust it
        # if y_pos >= y_top:
        #     y_pos = y_top * 0.95
        
        # Add the annotation line and text
        y_pos = ylim[1] * 0.90 if ylim is not None else ax.get_ylim()[1] * 0.90
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        ax.plot([0, 0, 1, 1], [y_pos-0.01*y_range, y_pos, y_pos, y_pos-0.01*y_range], lw=1.5, c='black')
        ax.text(0.5, y_pos + 0.01*y_range, sig_text, ha='center', va='bottom', fontsize=14)
        
        # Add some extra space at the top for the annotation
        ax.set_ylim(bottom=None, top=y_pos + 0.05*y_range)
    
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # ax.set_title('AMI Distribution by Condition')
    plt.tight_layout()
    
    return ax

def perm_cluster_test(epochs1, epochs2, picks=None, n_permutations=1000, tail=0, threshold=None):
    from mne.stats import permutation_cluster_test
    data_1_pick = epochs1.get_data(picks=picks)  # shape (n_epochs, n_channels, n_times)
    data_2_pick = epochs2.get_data(picks=picks)  # shape (n_epochs, n_channels, n_times

    X = [data_1_pick, data_2_pick]
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
        X, n_permutations=n_permutations, tail=tail, threshold=threshold, out_type='mask')
    # get significant clusters timepoints
    time = epochs1.times
    sig_times = []
    for cl_idx, p_val in enumerate(cluster_p_values):
        if p_val < 0.05:
            sig_times.append(time[clusters[cl_idx][0]])
    print(f"Significant clusters found at time points: {sig_times}")

    return sig_times

if __name__ == '__main__':
    base_path = "/Users/jusungham/HANG/aad-trf/"
    predictions = []
    corrects = []
    model_names = []
    clf_config_ids = [16,16]
    config_ids_str = '-'.join([str(clf_config_id) for clf_config_id in clf_config_ids])
    for clf_config_id in clf_config_ids:
        config_id = f"classifier-{clf_config_id:03d}"
        result_df = parse_clf_results(base_path, config_id)
        config = load_config(config_id)
        model_names.append(config['model_name'])
        prediction = result_df['prediction']
        correct = result_df['correct']
        predictions.append(prediction)
        corrects.append(correct)
    
    predictions = np.array(predictions)
    corrects = np.array(corrects).astype(bool)
    agreement = np.all(predictions == predictions[0], axis=0)
    agreement_rate = agreement.mean()
    print(f"agreement rate: {agreement_rate:.4f}")

    # if len(clf_config_ids) == 2:
    #     from sklearn.metrics import cohen_kappa_score
    #     kappa = cohen_kappa_score(predictions[0], predictions[1])
    #     print(f"Cohen's Kappa: {kappa:.4f}")
    # else:
    #     from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa
    #     ratings = predictions.astype(int)
    #     ratings = ratings.T
    #     rating_counts = np.zeros((ratings.shape[0], 2), dtype=int)
    #     for i in range(ratings.shape[0]):
    #         unique, counts = np.unique(ratings[i], return_counts=True)
    #         rating_counts[i, unique] = counts

    #     # Calculate Fleiss' Kappa
    #     fleiss_kappa_value = fleiss_kappa(rating_counts, method='fleiss')
    #     print(f"Fleiss' Kappa: {fleiss_kappa_value:.4f}")

    # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # if len(clf_config_ids) == 2:
    #     from matplotlib_venn import venn2
    #     subsets = (set(np.where(corrects[0])[0]), set(np.where(corrects[1])[0]))
    #     venn2(subsets=subsets, set_labels=(model_names[0], model_names[1]), ax=ax)
    #     plt.text(0.5, -0.5, f"{np.all(~corrects, axis=0).sum()}")
    #     plt.savefig(os.path.join(base_path, 'reports', f'updown-nh_venn-diagram_config-{config_ids_str}.svg'), transparent=True)
    #     plt.savefig(os.path.join(base_path, 'reports', f'updown-nh_venn-diagram_config-{config_ids_str}.pdf'), transparent=True)
    # elif len(clf_config_ids) > 2:
    #     if len(clf_config_ids) == 3:
    #         from matplotlib_venn import venn3
    #         subsets = set(np.where(corrects[0])[0]), set(np.where(corrects[1])[0]), set(np.where(corrects[2])[0])
    #         venn3(subsets=subsets, set_labels=model_names, ax=ax)
    #         plt.text(0.5, -0.5, f"{np.all(~corrects, axis=0).sum()}")
    #         plt.savefig(os.path.join(base_path, 'reports', f'updown-nh_venn-diagram_config-{config_ids_str}.svg'), transparent=True)
    #         plt.savefig(os.path.join(base_path, 'reports', f'updown-nh_venn-diagram_config-{config_ids_str}.pdf'), transparent=True)
        
    #     from upsetplot import from_indicators, UpSet
    #     correct_dict = {model_name: correct for model_name, correct in zip(model_names, corrects)}
    #     correct_df = pd.DataFrame(correct_dict)
    #     correct_by_model = from_indicators(correct_df)
    #     upset = UpSet(correct_by_model, subset_size='count', orientation='horizontal', show_percentages=True)
    #     upset.plot()
    #     plt.savefig(os.path.join(base_path, 'reports', f'updown-nh_upset-plot_config-{config_ids_str}.svg'), transparent=True)
    #     plt.savefig(os.path.join(base_path, 'reports', f'updown-nh_upset-plot_config-{config_ids_str}.pdf'), transparent=True)

    correct_all = np.all(corrects, axis=0)
    incorrect_all = np.all(~corrects, axis=0)
    correct_all_idx = np.where(correct_all)[0]
    incorrect_all_idx = np.where(incorrect_all)[0]
    correct_all_idx_sample = np.random.choice(correct_all_idx, len(incorrect_all_idx), replace=False)

    # correct_all = correct_lr & correct_svc
    # incorrect_all = ~correct_lr & ~correct_svc
    # correct_all_idx = correct_all[correct_all].index
    # incorrect_all_idx = incorrect_all[incorrect_all].index
    # correct_all_idx_sample = np.random.choice(correct_all_idx, len(incorrect_all_idx), replace=False)

    # # # get index of correct trials
    # correct_true_lr = correct_lr[correct_lr] # subset of boolean array
    # correct_idx_lr = correct_true_lr.sample(n=len(results_df_lr) - sum(correct_lr), random_state=42).index
    # correct_true_svc = correct_svc[correct_svc]
    # correct_idx_svc = correct_true_svc.sample(n=len(results_df_svc) - sum(correct_svc), random_state=42).index

    # load AAD_dataset
    from aad_trf.aad_dataset import AAD_Dataset
    config_id_audio = 'audio-002'
    config_id_eeg = 'eeg-003'
    dataset_name = 'updown-nh'
    audio_filename = f"dataset-{dataset_name}_data-audio_config-{config_id_audio}"
    audio = np.load(os.path.join(base_path, 'data', 'features', f"{audio_filename}.npy"))
    eeg_filename = f"dataset-{dataset_name}_data-eeg_config-{config_id_eeg}_-epo"
    epochs = mne.read_epochs(os.path.join(base_path, 'data', 'features', f"{eeg_filename}.fif"))
    dataset = AAD_Dataset()
    dataset.create(epochs, audio)
    del audio, epochs

    epochs_correct = dataset[correct_all_idx_sample].to_epochs()
    epochs_incorrect = dataset[incorrect_all_idx].to_epochs()
    epochs_correct.drop_bad(reject=dict(eeg=300e-6))
    epochs_incorrect.drop_bad(reject=dict(eeg=300e-6))
    channel = ['C6']
    # sig_times_correct = perm_cluster_test(epochs_correct['up'], epochs_correct['down'], picks=channel, n_permutations=1000, tail=0, threshold=None)
    # sig_times_incorrect = perm_cluster_test(epochs_incorrect['up'], epochs_incorrect['down'], picks=channel, n_permutations=1000, tail=0, threshold=None)

    # # fig1, axs1 = plot_butterfly(epochs_correct, epochs_incorrect, ylim=None)
    # # plt.savefig(os.path.join(base_path, 'reports', f'updown-nh_correct_incorrect_channel-all_config-{config_ids_str}.pdf'),
    # #         transparent=True)
    # fig2, axs2 = plot_correct_incorrect2(epochs_correct, epochs_incorrect, 
                                        # channel=channel,
                                        # ylim=[-3.0e-6, 2.5e-6])
    # highlight significant time points
    # for sig_time in sig_times_correct:
    #     axs2[0].axvspan(sig_time[0], sig_time[-1], color='yellow', alpha=0.3)
    # for sig_time in sig_times_incorrect:
    #     axs2[1].axvspan(sig_time[0], sig_time[-1], color='yellow', alpha=0.3)
    
    # channel_names = '-'.join(channel)
    # plt.savefig(os.path.join(base_path, 'reports', f'updown-nh_correct_incorrect2_channel-{channel_names}-gfp_config-{config_ids_str}.pdf'),
    #             transparent=True)
    # plt.savefig(os.path.join(base_path, 'reports', f'updown-nh_correct_incorrect2_channel-{channel_names}-gfp_config-{config_ids_str}.svg'),
    #             transparent=True)
    # plt.savefig(os.path.join(base_path, 'reports', f'updown-nh_correct_incorrect2_channel-{channel_names}-gfp_config-{config_ids_str}.png'),
    #             transparent=True, dpi=300)
    # fig3, axs3 = plot_gfp(epochs_correct, epochs_incorrect, ylim=[0,1.8e-6])
    # plt.savefig(os.path.join(base_path, 'reports', f'updown-nh_correct_incorrect_gfp_config-{config_ids_str}.pdf'),
    #                 transparent=True)
    
    ### Calculate AMI
    window_n1 = (0.08, 0.16)
    window_p2 = (0.17, 0.28)
    up_onsets = np.linspace(0,4,6)
    up_onsets = up_onsets[1:-1]
    down_onsets = np.linspace(0,4,5)
    down_onsets = down_onsets[1:-1]

    # print("Calculating Attentional Modulation Index (AMI)...")
    # ami_results, peak_info = calculate_ami(
    #     epochs_correct, epochs_incorrect, up_onsets, down_onsets, 
    #     bootstrap=True, window_size=window_n1)
    
    ### Save AMI results to CSV
    # config_ids_str = 'all'
    ami_csv_path = os.path.join(base_path, 'reports', f'updown-nh_ami-results_config-{config_ids_str}_n1.csv')
    # ami_results.to_csv(ami_csv_path, index=False)
    ami_results = pd.read_csv(ami_csv_path)
    # print(f"AMI results saved to {ami_csv_path}")
    
    # # one sample t-test to check if ami_correct is significantly different from zero
    # from scipy import stats
    # t_stat, p_value = stats.ttest_1samp(ami_results['ami_correct'][1:], 0)
    # print(f"AMI: {ami_results['ami_correct'][1:].mean():.4f} ± {ami_results['ami_correct'][1:].std():.4f} (SD)")
    # print(f"One-sample t-test for AMI : t={t_stat:.4f}, p-value={p_value:.4f}")
    
    # # plot ami distribution across subjects
    # plt.figure(figsize=(2.5,4))
    # ax = sns.boxplot(data=ami_results.iloc[1:], y='ami_correct', color='lightgrey', width=0.4)
    # ax = sns.swarmplot(ax=ax, data=ami_results.iloc[1:], y='ami_correct', color='black', size=6)
    # ax.set_ylabel('AMI (AU)')
    # ax.set_xlabel('')
    # ax.grid(axis='y')
    # plt.tight_layout()
    # plt.savefig(os.path.join(base_path, 'reports', f'updown-nh_ami-correct_boxplot_config-{config_ids_str}_n1.svg'), transparent=True)
    # plt.savefig(os.path.join(base_path, 'reports', f'updown-nh_ami-correct_boxplot_config-{config_ids_str}_n1.pdf'), transparent=True)
    # plt.close()
    
    ### Plot and save the AMI comparison
    plt.figure(figsize=(4,6))
    ax = plot_ami_comparison(ami_results, ylim = [-0.30, 0.45])
    
    ### Save the seaborn plot
    plt.savefig(os.path.join(base_path, 'reports', f'updown-nh_ami-swarmbox_config-{config_ids_str}_n1.svg'), transparent=True)
    plt.savefig(os.path.join(base_path, 'reports', f'updown-nh_ami-swarmbox_config-{config_ids_str}_n1.pdf'), transparent=True)
    # # Visualize which peaks are used in AMI calculation using the pre-computed peak info
    # fig_peaks, axs_peaks = plot_ami_peaks(ami_results, peak_info, epochs_correct, epochs_incorrect, 
    #                                      up_onsets, down_onsets, ylim=[0, 1.8e-6])
    # plt.savefig(os.path.join(base_path, 'reports', f'updown-nh_ami-peaks_config-{config_ids_str}.pdf'), transparent=True)
    # plt.savefig(os.path.join(base_path, 'reports', f'updown-nh_ami-peaks_config-{config_ids_str}.svg'), transparent=True)

    # ami_results, peak_info = calculate_ami(
    #     epochs_correct, epochs_incorrect, up_onsets, down_onsets, 
    #     bootstrap=True, window_size=window_p2)
    # ami_csv_path = os.path.join(base_path, 'reports', f'updown-nh_ami-results_config-{config_ids_str}_p2.csv')
    # ami_results.to_csv(ami_csv_path, index=False)
    # # ami_results = pd.read_csv(ami_csv_path)
    # print(f"AMI results saved to {ami_csv_path}")
    # plt.figure(figsize=(4,6))
    # ax = plot_ami_comparison(ami_results, ylim = [-0.25, 0.35])
    # plt.savefig(os.path.join(base_path, 'reports', f'updown-nh_ami-swarmbox_config-{config_ids_str}_p2.svg'), transparent=True)
    # plt.savefig(os.path.join(base_path, 'reports', f'updown-nh_ami-swarmbox_config-{config_ids_str}_p2.pdf'), transparent=True)

    epochs_all = dataset.to_epochs()
    epochs_all.drop_bad(reject=dict(eeg=300e-6))
    # epochs_up = epochs_all['up']
    # epochs_down = epochs_all['down']
    # data_up_pick = epochs_up.pick(['Cz']).get_data()
    # data_down_pick = epochs_down.pick(['Cz']).get_data()
    # from mne.stats import permutation_cluster_test
    # X = [data_up_pick, data_down_pick]
    # T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
    #     X, n_permutations=1000, tail=0, threshold=None, out_type='indices', n_jobs=1)
    # # get significant clusters timepoints
    # time = epochs_all.times
    # sig_times = []
    # for cl_idx, p_val in enumerate(cluster_p_values):
    #     if p_val < 0.05:
    #         sig_times.append(time[clusters[cl_idx][0]])
    # print(f"Significant clusters found at time points: {sig_times}")

    # fig, ax = plot_erp_gfp(epochs_all, channel=['Cz'], ylim=[-3.0e-6, 3.5e-6])
    # # highlight significant time points
    # for sig_time in sig_times:
    #     ax.axvspan(sig_time[0], sig_time[-1], color='yellow', alpha=0.3)

    # plt.savefig(os.path.join(base_path, 'reports', f'updown-nh_erp_gfp_channel-Cz_config-all.pdf'), transparent=True)
    # plt.savefig(os.path.join(base_path, 'reports', f'updown-nh_erp_gfp_channel-Cz_config-all.svg'), transparent=True)
    # plt.savefig(os.path.join(base_path, 'reports', f'updown-nh_erp_gfp_channel-Cz_config-all.png'), transparent=True, dpi=300)
    
    # fig, ax = plot_gfp_overlay(epochs_all, ylim=[0, 1.8e-6])
    # plt.savefig(os.path.join(base_path, 'reports', f'updown-nh_gfp_overlay_config-all.pdf'), transparent=True)
    # plt.savefig(os.path.join(base_path, 'reports', f'updown-nh_gfp_overlay_config-all.svg'), transparent=True)
    # plt.savefig(os.path.join(base_path, 'reports', f'updown-nh_gfp_overlay_config-all.png'), transparent=True, dpi=300)

    ami_results_all, peak_info_all = calculate_ami(epochs_all, epochs_all, up_onsets, down_onsets, window_size=(0.05, 0.25))
    ami_csv_path_all = os.path.join(base_path, 'reports', f'updown-nh_ami-results_config-all_wide.csv')
    ami_results_all.to_csv(ami_csv_path_all, index=False)

    fig4, axs4 = plot_ami_peaks(ami_results_all, peak_info_all, epochs_all, epochs_all, 
                                up_onsets, down_onsets, ylim=[0, 1.8e-6], window_size=(0.05, 0.25))
    plt.savefig(os.path.join(base_path, 'reports', f'updown-nh_ami-peaks_config-all_wide.pdf'), transparent=True)
    plt.savefig(os.path.join(base_path, 'reports', f'updown-nh_ami-peaks_config-all_wide.svg'), transparent=True)

    # ami_results_all, peak_info_all = calculate_ami(epochs_all, epochs_all, up_onsets, down_onsets, window_size=window_n1)
    # ami_csv_path_all = os.path.join(base_path, 'reports', f'updown-nh_ami-results_config-all_n1.csv')
    # ami_results_all.to_csv(ami_csv_path_all, index=False)

    # fig4, axs4 = plot_ami_peaks(ami_results_all, peak_info_all, epochs_all, epochs_all, 
    #                             up_onsets, down_onsets, ylim=[0, 1.8e-6], window_size=window_n1)
    # plt.savefig(os.path.join(base_path, 'reports', f'updown-nh_ami-peaks_config-all_n1.pdf'), transparent=True)
    # plt.savefig(os.path.join(base_path, 'reports', f'updown-nh_ami-peaks_config-all_n1.svg'), transparent=True)

    # ami_results_all, peak_info_all = calculate_ami(epochs_all, epochs_all, up_onsets, down_onsets, window_size=window_p2)
    # ami_csv_path_all = os.path.join(base_path, 'reports', f'updown-nh_ami-results_config-all_p2.csv')
    # ami_results_all.to_csv(ami_csv_path_all, index=False)
    # fig5, axs5 = plot_ami_peaks(ami_results_all, peak_info_all, epochs_all, epochs_all, 
    #                             up_onsets, down_onsets, ylim=[0, 3.0e-6], window_size=window_p2)
    # plt.savefig(os.path.join(base_path, 'reports', f'updown-nh_ami-peaks_config-all_p2.pdf'), transparent=True)
    # plt.savefig(os.path.join(base_path, 'reports', f'updown-nh_ami-peaks_config-all_p2.svg'), transparent=True)

    # Draw AMI peaks for each subject
    subjects = epochs_all.metadata['sub_id'].unique()
    for subject in subjects:
        epochs_subj = epochs_all[f'sub_id == "{subject}"']
        ami_results_subj, peak_info_subj = calculate_ami(epochs_subj, epochs_subj, up_onsets, down_onsets, window_size=(0.05, 0.25))
        fig_subj, axs_subj = plot_ami_peaks(ami_results_subj, peak_info_subj, epochs_subj, epochs_subj, 
                                            up_onsets, down_onsets, ylim=[0, 3.0e-6], window_size=(0.05, 0.25))
        plt.savefig(os.path.join(base_path, 'reports', f'updown-nh_ami-peaks_subject-{subject}_config-all_wide.pdf'), transparent=True)
        plt.savefig(os.path.join(base_path, 'reports', f'updown-nh_ami-peaks_subject-{subject}_config-all_wide.svg'), transparent=True)
        plt.close(fig_subj)

    for subject in subjects:
        epochs_subj = epochs_all[f'sub_id == "{subject}"']
        ami_results_subj, peak_info_subj = calculate_ami(epochs_subj, epochs_subj, up_onsets, down_onsets, window_size=window_n1)
        fig_subj, axs_subj = plot_ami_peaks(ami_results_subj, peak_info_subj, epochs_subj, epochs_subj, 
                                            up_onsets, down_onsets, ylim=[0, 3.0e-6], window_size=window_n1)
        plt.savefig(os.path.join(base_path, 'reports', f'updown-nh_ami-peaks_subject-{subject}_config-all_n1.pdf'), transparent=True)
        plt.savefig(os.path.join(base_path, 'reports', f'updown-nh_ami-peaks_subject-{subject}_config-all_n1.svg'), transparent=True)
        plt.close(fig_subj)
    
    # for subject in subjects:
    #     epochs_subj = epochs_all[f'sub_id == "{subject}"']
    #     ami_results_subj, peak_info_subj = calculate_ami(epochs_subj, epochs_subj, up_onsets, down_onsets, window_size=window_p2)
    #     fig_subj, axs_subj = plot_ami_peaks(ami_results_subj, peak_info_subj, epochs_subj, epochs_subj, 
    #                                         up_onsets, down_onsets, ylim=[0, 3.0e-6], window_size=window_p2)
    #     plt.savefig(os.path.join(base_path, 'reports', f'updown-nh_ami-peaks_subject-{subject}_config-all_p2.pdf'), transparent=True)
    #     plt.savefig(os.path.join(base_path, 'reports', f'updown-nh_ami-peaks_subject-{subject}_config-all_p2.svg'), transparent=True)
    #     plt.close(fig_subj)

    # trial-count balanced bootstrap comparison
    # ami_B_bs, _, _, _ = _calculate_ami_single(epochs_incorrect, up_onsets, down_onsets, window_size=(0.05, 0.25))
    # ami_A_bs, ami_B_bs = bootstrap_balanced_ami(
    #     epochs_correct, epochs_incorrect, up_onsets, down_onsets,
    #     window_size=(0.05, 0.25), n_bootstraps=1000)
    # diff_bs = ami_A_bs - ami_B_bs
    # p_val = (np.abs(diff_bs) >= np.abs(diff_bs.mean())).mean()  # two-sided
    # print(f"Bootstrap AMI diff mean={diff_bs.mean():.3f}, p≈{p_val:.3f}")
    
    # # Optional: inspect distribution
    # plt.figure()
    # plt.hist(ami_A_bs, bins='auto', alpha=0.5, label='A (balanced)')
    # plt.hist(ami_B_bs, bins='auto', alpha=0.5, label='B (balanced)')
    # plt.axvline(0, color='k', linestyle='--')
    # plt.legend()
    # plt.show()
        
    # --- Calculate and Save RMS Distribution for Each Subject ---
    print("Calculating RMS distribution per subject...")
    rms_records = []
    snr_records = []
    
    import seaborn as sns
    from tqdm import tqdm
    # Ensure we have subject IDs
    if epochs_all.metadata is not None and 'sub_id' in epochs_all.metadata.columns:
        subjects = epochs_all.metadata['sub_id'].unique()
        
        for subject in tqdm(subjects, desc="Subject RMS"):
            # 1. Select epochs for this subject
            #    (Use copy=False to save memory if data is preloaded)
            epochs_subj = epochs_all[f'sub_id == "{subject}"']
            epochs_subj_correct = epochs_correct[f'sub_id == "{subject}"']
            epochs_subj_incorrect = epochs_incorrect[f'sub_id == "{subject}"']
            epochs_subj.crop(tmin=0.5)
            
            # 2. Calculate RMS: sqrt(mean(x^2)) over channels (axis 1) and time (axis 2)
            #    Result is (n_epochs,)
            data = epochs_subj.get_data(copy=True) # shape: (n_epochs, n_ch, n_times)
            evoked = epochs_subj.average().get_data() # shape: (n_ch, n_times)
            evoked_up = epochs_subj['up'].average().get_data()
            gfp_up = np.std(evoked_up, axis=0)  # shape: (n_times,)
            evoked_down = epochs_subj['down'].average().get_data()
            gfp_down = np.std(evoked_down, axis=0)  # shape: (n_times,)
            # get RMS of evoked except for the N1 windows (80-160ms) from onsets using `window_n1` and `up_onsets`, `down_onsets`
            
            times = epochs_subj.times
            mask_times = np.ones(len(times), dtype=bool)
            all_onsets = np.concatenate([up_onsets, down_onsets])
            for onset in all_onsets:
                t_start = onset + window_n1[0]
                t_end = onset + window_n1[1]
                mask_times[(times >= t_start) & (times <= t_end)] = False
            # for onset in all_onsets:
            #     t_start = onset + window_p2[0]
            #     t_end = onset + window_p2[1]
            #     mask_times[(times >= t_start) & (times <= t_end)] = False
            
            # data = data[..., mask_times]
            gfp_up_noise_rms = np.sqrt(np.mean(gfp_up[mask_times] ** 2))
            gfp_down_noise_rms = np.sqrt(np.mean(gfp_down[mask_times] ** 2))
            gfp_up_signal_rms = np.sqrt(np.mean(gfp_up[~mask_times] ** 2))
            gfp_down_signal_rms = np.sqrt(np.mean(gfp_down[~mask_times] ** 2))
            gfp_up_snr = gfp_up_signal_rms / gfp_up_noise_rms
            gfp_down_snr = gfp_down_signal_rms / gfp_down_noise_rms
            # print(f"Subject {subject}: Up SNR={gfp_up_snr:.2f}, Down SNR={gfp_down_snr:.2f}")

            # get peak info for N1 peaks


            snr_records.append({
                'sub_id': subject,
                'snr_up': gfp_up_snr,
                'snr_down': gfp_down_snr
            })
                        
            residuals = data - evoked[np.newaxis, :, :]
            rms_overall = np.sqrt(np.mean(residuals ** 2, axis=(1, 2)))
            residuals_gfp = np.std(residuals, axis=1)  # shape: (n_epochs, n_times)
            rms_gfp = np.sqrt(np.mean(residuals_gfp ** 2, axis=1))  # shape: (n_epochs,)

            for rms in rms_overall:
                rms_records.append({
                    'sub_id': subject,
                    'condition': 'overall',
                    'rms_uv': rms * 1e6  # Convert to µV for readable plots
                })

            for rms in rms_gfp:
                rms_records.append({
                    'sub_id': subject,
                    'condition': 'gfp',
                    'rms_uv': rms * 1e6  # Convert to µV for readable plots
                })
            
            # rms_values = np.sqrt(np.mean(data ** 2, axis=(1, 2)))

            # data_correct = epochs_subj_correct.get_data(copy=True)
            # rms_values_correct = np.sqrt(np.mean(data_correct ** 2, axis=(1, 2)))
            # data_incorrect = epochs_subj_incorrect.get_data(copy=True)
            # rms_values_incorrect = np.sqrt(np.mean(data_incorrect ** 2, axis=(1, 2)))
            
            # # 3. separate by condition (optional, generally useful for plots)
            # events = epochs_subj.events[:, 2]
            # event_id_rev = {v: k for k, v in epochs_subj.event_id.items()}
            # conditions = [event_id_rev[e] for e in events]

            # # 4. Collect into list
            # # for rms, cond in zip(rms_values, conditions):
            # #     rms_records.append({
            # #         'sub_id': subject,
            # #         'condition': cond,
            # #         'rms_uv': rms * 1e6  # Convert to µV for readable plots
            # #     })
            
            # for rms in rms_values_correct:
            #     rms_records.append({
            #         'sub_id': subject,
            #         'condition': 'correct',
            #         'rms_uv': rms * 1e6  # Convert to µV for readable plots
            #     })

            # for rms in rms_values_incorrect:
            #     rms_records.append({
            #         'sub_id': subject,
            #         'condition': 'incorrect',
            #         'rms_uv': rms * 1e6  # Convert to µV for readable plots
            #     })

        # 5. Create DataFrame
        df_rms = pd.DataFrame(rms_records)
        df_snr = pd.DataFrame(snr_records)
        out_snr_csv = os.path.join(base_path, 'reports', f'updown-nh_snr-gfp.csv')
        df_snr.to_csv(out_snr_csv, index=False)
        print(f"Saved SNR data to {out_snr_csv}")
        snr_mean = df_snr[['snr_up', 'snr_down']].mean(axis=1)
        import scipy.stats as stats
        corr, pval = stats.pearsonr(snr_mean, ami_results['ami_correct'][1:])
        print(f"SNR mean correlation with AMI: r={corr:.2f}, p={pval:.3f}")
        plt.figure(figsize=(5,4))
        sns.regplot(x=snr_mean, y=ami_results['ami_correct'][1:])
        plt.xlabel('Average SNR (GFP)')
        plt.ylabel('AMI')
        plt.title(f'SNR vs AMI (r={corr:.2f}, p={pval:.3f})')
        out_snr_corr_plot = os.path.join(base_path, 'reports', f'updown-nh_snr-gfp_ami_corr.pdf')
        plt.savefig(out_snr_corr_plot, bbox_inches='tight')
        print(f"Saved SNR vs AMI correlation plot to {out_snr_corr_plot}")

        # 6. Save to CSV
        out_csv = os.path.join(base_path, 'reports', f'updown-nh_rms-dist-gfp.csv')
        df_rms.to_csv(out_csv, index=False)
        print(f"Saved RMS data to {out_csv}")


        # get average RMS per subject
        rms_summary = df_rms.groupby(['sub_id', 'condition'])['rms_uv'].mean().reset_index()
        # add AMI results to summary
        ami_all = pd.read_csv(os.path.join(base_path, 'reports', f'updown-nh_ami-results_config-all.csv'))
        ami_n1 = pd.read_csv(os.path.join(base_path, 'reports', f'updown-nh_ami-results_config-all_n1.csv'))
        ami_p2 = pd.read_csv(os.path.join(base_path, 'reports', f'updown-nh_ami-results_config-all_p2.csv'))
        rms_summary = rms_summary.merge(ami_all[['subject', 'ami_correct']],
                                        left_on='sub_id', right_on='subject', how='left', suffixes=('', '_all'))
        rms_summary = rms_summary.merge(ami_n1[['subject', 'ami_correct']],
                                        left_on='sub_id', right_on='subject', how='left', suffixes=('', '_n1'))
        rms_summary = rms_summary.merge(ami_p2[['subject', 'ami_correct']],
                                        left_on='sub_id', right_on='subject', how='left', suffixes=('', '_p2'))
        out_summary_csv = os.path.join(base_path, 'reports', f'updown-nh_rms-summary-gfp.csv')
        rms_summary.to_csv(out_summary_csv, index=False)
        print(f"Saved RMS summary to {out_summary_csv}")

        # plot correlation between RMS and AMI with statistics
        import scipy.stats as stats
        for ami_type in ['all', 'n1', 'p2']:
            ami_column = f'ami_correct_{ami_type}' if ami_type != 'all' else 'ami_correct'
            corr_r, pval_pearson = stats.pearsonr(rms_summary['rms_uv'], rms_summary[ami_column])
            rho, pval_spearman = stats.spearmanr(rms_summary['rms_uv'], rms_summary[ami_column])
            plt.figure(figsize=(5,4))
            sns.regplot(x='rms_uv', y=ami_column, data=rms_summary)
            plt.xlabel('Average RMS (µV)')
            plt.ylabel('AMI')
            plt.title(f'RMS vs AMI-{ami_type} (ρ={rho:.2f}, p={pval_spearman:.3f}; r={corr_r:.2f}, p={pval_pearson:.3f})')
            out_corr_plot = os.path.join(base_path, 'reports', f'updown-nh_rms-gfp_ami-{ami_type}_corr.pdf')
            plt.savefig(out_corr_plot, bbox_inches='tight')
            print(f"Saved RMS vs {ami_type} correlation plot to {out_corr_plot}")

        # # 7. Plot using Seaborn FacetGrid (One histogram per subject)
        # #    col_wrap controls how many plots per row
        # #    hue differentiates conditions within each plot
        # sns.set_style("whitegrid")
        # g = sns.FacetGrid(df_rms, col="sub_id", hue='condition',
        #                   col_wrap=5, height=3, aspect=1.2,
        #                   sharex=False, sharey=False, legend_out=True)
        
        # # Map the histogram plot to the grid
        # g.map(sns.histplot, "rms_uv", kde=True, binwidth=0.3, line_kws={'linewidth': 1.5})
        
        # # Labeling
        # g.set_axis_labels("RMS (µV)", "Trial Count")
        # g.fig.suptitle("Single-Trial Noise (RMS) Distribution per Subject", y=1.02, fontsize=16)
        
        # # Save Plot
        # out_plot = os.path.join(base_path, 'reports', f'updown-nh_rms-dist-correct-incorrect.pdf')
        # g.savefig(out_plot, bbox_inches='tight')
        # print(f"Saved RMS distribution plots to {out_plot}")
        
    else:
        print("[WARN] No 'sub_id' in metadata; cannot calculate per-subject RMS.")
