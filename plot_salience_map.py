import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import mne

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
    # draw vertical lines for onsets
    for onset, line_color, cond in zip(onsets, ['magenta', 'cyan'], ['up', 'down']):
        for o in onset:
            plt.axvline(x=(o-0.5)*64, color=line_color, linestyle='--', linewidth=1)
            # add text for the condition above the plot
            plt.text((o-0.5)*64, -2, cond, color='k', fontsize=12, ha='center', va='bottom')
    # Set the x-ticks as time values with step of 0.5 seconds
    plt.xticks(ticks=np.arange(0, len(time), 32), labels=time[::32], rotation=0)
    # Set the y-ticks as channel numbers of 1 to 64 with a step of 5
    plt.yticks(ticks=np.arange(0, salience_map.shape[0], 5), labels=np.arange(1, 65, 5), rotation=0)
    
    if title:
        plt.title(title)
    
    plt.xlabel('Time')
    plt.ylabel('Channels')
    
    return plt

def plot_salience_topomap(salience_map, title=None, cmap='rocket_r'):
    """
    Plot the salience map as a topomap.

    Parameters:
    - salience_map: 2D numpy array representing the salience map.
    - title: Title of the plot (optional).
    - cmap: Colormap to use for the topomap (default is 'viridis').
    - save_path: Path to save the plot (optional).
    """
    salience_map_summary = np.mean(salience_map, axis=1)

    # Create a new figure
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Create a topomap
    epoch_path = 'data/updown-nh/eeg/sub-wj0681_task-updown-epo.fif'
    epoch = mne.read_epochs(epoch_path)
    topo, _ = mne.viz.plot_topomap(salience_map_summary,
                         pos=epoch.info,
                         axes=ax,
                         cmap=cmap,
                         sphere='eeglab',
                         show=False)
    # Set the colorbar
    plt.colorbar(topo,
                orientation='vertical', 
                shrink=0.8,
                ax=ax
                )
    
    if title:
        ax.set_title(title)
    
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

    save_name = f'dataset-updown-nh_{file_name}_topo'
    save_path = os.path.join(file_dir, save_name)
    plt_stopo = plot_salience_topomap(salience_map, title='Salience Topography', cmap=None)
    plt_stopo.savefig(f'{save_path}.svg')
    plt_stopo.savefig(f'{save_path}.pdf')
    # The above code will create a heatmap of the salience map and save it as 'salience_map.png'.
