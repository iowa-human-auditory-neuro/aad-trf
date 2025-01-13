# aad_trf Module

Version: 1.1

Author: Jusung Ham

Contact: jusung-ham@uiowa.edu

Date: 2025-01-13

---

This module is made for auditory attention decoding (AAD) based on the temporal response function (TRF). It is mainly composed of 3 processes: 1) Audio & EEG preprocessing, 2) TRF modeling 3) Classification, and each process has its own module and corresponding main function (`preprocessor_main.py`, `trf_main.py`, `classifier_main.py`). 


## Python Library Dependency

- mne
- seaborn

Please follow the instruction from [mne documentation page](https://mne.tools/stable/install/manual_install.html#manual-install) to install the MNE-Python package.

## Folder Structure
```
|-- data
|   |-- [raw_dataset]
|   |-- features
|   |-- montages
|   
|-- models
|-- reports
|-- src
|   |-- aad_trf
|   |-- config
```

## Dataset

To start the process, epoched EEG data and audio files are required. EEG data should be saved in the `data/[raw_datset]/eeg` folder as `.fif` file format; Audio data should be saved in the `data/[raw_datset]/audio` folder as `.wav` file format. 

## Configuration Parsing
Configuration should be saved in the `src/config` folder in `.json` format. Below are examples of configuration files. Please refer to the description of individual classes and methods for the possible options for each field.

### Audio preprocessing

```json
{
  "crop_time": [0.5, null],
  "downsfreq": 64
}
```

### EEG preprocessing

```json
{
  "rereferencing": "mastoids",
  "baseline": [-0.4, -0.1],
  "cutoff_freq": [1.0, 15.0],
  "crop_time": [1, 4.5],
  "downsfreq": 64
}
```
### TRF

```json
{
  "dataset": "updown-nh",
  "config_id_audio": "audio-001",
  "config_id_eeg": "eeg-001",
  "normalize": "True",
  "direction": "forward",
  "delays": [0,0.4],
  "search_space": [-2, 9, 12],
  "n_folds": 10,
  "scoring": "corrcoef"
}
```

### Classification

```json
{
  "config_id_trf": "trf-001",
  "model_name": "LogisticRegression"
}
```

## File naming conventions

> [ ]: required field
>
> ( ): optional field

### Dataset name
- [task-name]-[subject-population]
### Configuration file name
- `config-[configuration-id].json`
- configuration id: [processing_step]-[3-digit-number]
### Data file
- `dataset-[dataset-name]_data-[data-type]_config-[configuration-id](_sub-[subject_id]).[file-extension]`
- Examples
  - Preprocessed audio data: `dataset-[dataset-name]_data-audio_config-[configuration-id](_sub-[subject_id]).npy`
  - Preprocessed eeg data: `dataset-[dataset-name]_data-eeg_config-[configuration-id](_sub-[subject-id])_-epo.fif`
  - AAD dataset: `dataset-[dataset-name]_data-aad_config-[configuration-id](_sub-[subject-id]).pkl`

### Models
- `dataset-[dataset-name]_models-[model_type]_config-[configuration-id](_sub-[subject-id]).pkl`

### Figures
-  `dataset-[dataset-name]_reports-[figure_type]_config-[configuration-id](_sub-[subject-id]).png`

### Results
- `dataset-[dataset-name]_reports-[result_type]_config-[configuration-id](_sub-[subject-id]).csv`
