# aad_trf Module

Version: 1.0

Author: Jusung Ham

Contact: jusung-ham@uiowa.edu

Date: 2024-06-25

---

This module is made for auditory attention decoding (AAD) based on the temporal response function (TRF). It is mainly composed of 3 processes: 1) Audio & EEG preprocessing, 2) TRF modeling 3) Classification, and each process has its own moudle and corresponding main function (`preprocessor_main.py`, `trf_main.py`, `classifier_main.py`). 


## Python Library Dependency

- mne
- seaborn

## Folder Sturcture
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

To start the process, epoched EEG data and audio file is required. EEG data should be saved in the `data/[raw_datset]/epochs` folder as `.fif` file format; Audio data should be save in the `data/[raw_datset]/audio` folder as `.wav` file format. 

## Configuration Parsing
Configuration should be saved in the `src/config` folder as `.json` format. There are two levels of configuration. The upper-level configuration include which configuration file to use for each process. The lower-level configuration includes specific parameters for the process. Below are the example of configuration files. Please refer to the description of individual class and methods for the possible options for each field.

### Upper-level cofiguration for controlling all the processes
```json
{
  "dataset": "updown",
  "preprocess-audio": {
    "config_id": "audio-001",
    "overwirte": "False"
  },
  "preprocess-eeg": {
    "config_id": "eeg-001",
    "overwrite": "False"
  },
  "trf": {
    "config_id": "trf-001",
    "overwrite": "False"
  },
  "classifier": {
    "config_id": "classifier-001",
    "overwrite": "False"
  }
}
```
### Lower-level configurations for each process
#### Audio preprocessing

```json
{
  "crop_time": [0.5, null],
  "downsfreq": 64
}
```

#### EEG preprocessing

```json
{
  "rereferencing": "mastoids",
  "baseline": [-0.4, -0.1],
  "cutoff_freq": [1.0, 15.0],
  "crop_time": [1, 4.5],
  "downsfreq": 64
}
```
#### TRF

```json
{
  "normalize": "True",
  "direction": "forward",
  "delays": [0,0.4],
  "search_space": [-2, 9, 12],
  "n_folds": 10,
  "scoring": "corrcoef"
}
```

#### Classification

```json
{
  "model_name": "LogisticRegression"
}
```

## File naming conventions

> [ ]: required field
>
> ( ): optional field

- Dataset name
  - [task-name]-[subject-population]
- Configuration file name
  - `config-[configuration-id].json`
  - configuration id: [processing_step]-[3-digit-number]
- Data file
  - `dataset-[dataset-name]_data-[data-type]_config-[configuration-id](_sub-[subject_id]).[file-extension]`
  - Examples
    - Preprocessed audio data: `dataset-[dataset-name]_data-audio_config-[configuration-id](_sub-[subject_id]).npy`
    - Preprocessed eeg data: `dataset-[dataset-name]_data-eeg_config-[configuration-id](_sub-[subject-id])_-epo.fif`
    - AAD dataset: `dataset-[dataset-name]_data-aad_config-[configuration-id](_sub-[subject-id]).pkl`

- Models
  - `dataset-[dataset-name]_models-[model_type]_config-[configuration-id](_sub-[subject-id]).pkl`

- Figures
  -  `dataset-[dataset-name]_reports-[figure_type]_config-[configuration-id](_sub-[subject-id]).png`

- Results
  - `dataset-[dataset-name]_reports-[result_type]_config-[configuration-id](_sub-[subject-id]).csv`



