import os
from glob import glob
from mne import read_epochs
from utils import load_config, set_paths_from_config, get_field_from_filename

base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
config = load_config("dataset-updown-nh_exp-1")
path_dict = set_paths_from_config(base_path, config)

files = sorted(glob(f"{path_dict['epochs']}/*epo.fif"))
epochs_list = []
remove_bads = "n"
for file in files:
    filename = file.split("/")[-1]
    sub_id = get_field_from_filename(filename, "sub")
    assert len(sub_id) == 6, f"Invalid subID: {sub_id}"
    epochs = read_epochs(file)
    print(sub_id)

    if len(epochs) > 120:
        epochs = epochs[-120:]
        print(len(epochs))
        epochs.save(os.path.join(path_dict['epochs'], f"new_sub-{sub_id}_task-updown-epo.fif"), overwrite=False)
    else:
        print(len(epochs))

