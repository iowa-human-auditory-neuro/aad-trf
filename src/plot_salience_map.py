import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import mne
from sklearn.decomposition import PCA
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, NullLocator  # add NullLocator

def plot_salience_map(salience_map, time, onsets, title=None, cmap='rocket'):
    """
    Plot the salience map as a heatmap.

    Parameters:
    - salience_map: 2D numpy array representing the salience map.
    - title: Title of the plot (optional).
    - cmap: Colormap to use for the heatmap (default is 'rocket').
    - save_path: Path to save the plot (optional).
    """
    plt.figure(figsize=(12, 4))
    sns.heatmap(salience_map, cmap=cmap, cbar=True)
    ylim = plt.ylim()
    # draw vertical lines for onsets
    for onset, line_color, cond in zip(onsets, ['magenta', 'cyan'], ['up', 'down']):
        for o in onset:
            plt.axvline(x=(o-0.5)*64, color=line_color, linestyle='--', linewidth=1)
            # add text for the condition above the plot
            plt.text((o-0.5)*64, ylim[1]*1.01, cond, color='k', fontsize=12, ha='center', va='bottom')
    # Set the x-ticks as time values with step of 0.5 seconds
    plt.xticks(ticks=np.arange(0, len(time), 32), labels=time[::32], rotation=0)
    # Set the y-ticks as channel numbers of 1 to 64 with a step of 5
    plt.yticks(ticks=np.arange(0, salience_map.shape[0], 5), labels=np.arange(1, 65, 5), rotation=0)
    
    if title:
        plt.title(title)
    
    plt.xlabel('Time')
    plt.ylabel('Channels')
    
    return plt

def plot_salience_wave(salience_map, time, onsets, title=None):
    """
    Plot the salience map as a line plot.

    Parameters:
    - salience_map: 2D numpy array representing the salience map.
    - onsets: List of onset times for different conditions.
    - title: Title of the plot (optional).
    """
    plt.figure(figsize=(12, 4))
    # Plot each channel's salience over time
    for i in range(salience_map.shape[0]):
        plt.plot(time, salience_map[i], label=f'Channel {i+1}', alpha=0.5, linewidth=0.8)
    plt.xlim(time[0], time[-1])
    ylim = plt.ylim()
    # Draw vertical lines for onsets
    for onset, line_color, cond in zip(onsets, ['red', 'blue'], ['up', 'down']):
        for o in onset:
            plt.axvline(x=o, color=line_color, linestyle='--', linewidth=1)
            # add text for the condition above the plot
            plt.text(o, ylim[1]*1.01, cond, color='k', fontsize=12, ha='center', va='bottom')

    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(0.5))            # major every 0.5s
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))            # minor every 0.1s
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))    # label major ticks
    ax.tick_params(axis='x', which='minor', labelbottom=False)  # hide minor labels
    plt.minorticks_on()
    ax.yaxis.set_minor_locator(NullLocator())

    if title:
        plt.title(title)
    
    plt.xlabel('Time')
    plt.ylabel('Salience')
    plt.ylim(0, ylim[1])
    
    return plt

def collapse_pca_weights(sal):
    """
    Run PCA treating timepoints as samples and channels as features.
    Returns the first PC loadings (length n_channels).
    """
    # transpose to shape (n_t, n_ch)
    X = sal.T
    pca = PCA(n_components=1)
    pca.fit(X)
    # component_[0] is the direction (length n_ch)
    # here we take absolute value or keep sign depending on interpretation
    return pca.components_[0]

def plot_salience_topomap(salience_map, title=None, cmap='rocket_r'):
    """
    Plot the salience map as a topomap.

    Parameters:
    - salience_map: 2D numpy array representing the salience map.
    - title: Title of the plot (optional).
    - cmap: Colormap to use for the topomap (default is 'viridis').
    - save_path: Path to save the plot (optional).
    """
    salience_map_summary = [np.mean(salience_map, axis=1),
                            np.max(salience_map, axis=1),
                            np.std(salience_map, axis=1),
                            np.sqrt(np.mean(salience_map**2, axis=1)),
                            collapse_pca_weights(salience_map)]
    methods = ['mean', 'max', 'std', 'rms', 'pca']

    # Create a new figure
    fig, axs = plt.subplots(1,len(salience_map_summary), figsize=(3*len(salience_map_summary), 3))
    
    # Create a topomap
    epoch_path = 'data/updown-nh/eeg/sub-wj0681_task-updown-epo.fif'
    epoch = mne.read_epochs(epoch_path)
    for salmap, ax, method in zip(salience_map_summary, axs, methods):
        topo, _ = mne.viz.plot_topomap(salmap,
                            pos=epoch.info,
                            axes=ax,
                            cmap=cmap,
                            sphere='eeglab',
                            show=False)
        # Set the colorbar
        plt.colorbar(topo,
                    orientation='vertical', 
                    shrink=0.8,
                    ax=ax)
        # Set the title
        ax.set_title(method)
    
    if title:
        fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    return plt

if __name__ == "__main__":
    # Create a random salience map for demonstration
    file_dir = 'reports'
    file_name = 'salience-map_total_updown'
    path = os.path.join(file_dir, f'{file_name}.npy')
    salience_map = np.load(path)
    time = np.arange(0.5, 4, 1/64)
    up_onsets = np.linspace(0,4,6)
    up_onsets = up_onsets[1:-1]
    down_onsets = np.linspace(0,4,5)
    down_onsets = down_onsets[1:-1]
    onsets = [up_onsets, down_onsets]

    # Plot the salience map
    save_name = f'dataset-updown-nh_{file_name}'
    save_path = os.path.join(file_dir, save_name)
    plt_smap = plot_salience_map(salience_map, time, onsets)
    plt_smap.savefig(f'{save_path}.svg')
    plt_smap.savefig(f'{save_path}.pdf')

    save_path = os.path.join(file_dir, f'{save_name}_topo')
    plt_stopo = plot_salience_topomap(salience_map, title='Salience Topography', cmap=None)
    plt_stopo.savefig(f'{save_path}.svg')
    plt_stopo.savefig(f'{save_path}.pdf')
    # The above code will create a heatmap of the salience map and save it as 'salience_map.png'.

    plt_swave = plot_salience_wave(salience_map, time, onsets)
    save_path = os.path.join(file_dir, f'{save_name}_wave')
    plt_swave.savefig(f'{save_path}.svg')
    plt_swave.savefig(f'{save_path}.pdf')
