import os
import json

def load_config(config_id:str) -> dict:
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
    config_filename = f"{config_id}.json"
    config_file_path = os.path.join(config_path, config_filename)

    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Configuration file {config_file_path} not found.")
    
    return json.load(open(config_file_path, "r"))

def set_paths_from_config(base_path:str, config:dict) -> dict:
    data_path = os.path.join(base_path, "data")
    audio_path = os.path.join(data_path, config['dataset'], "audio")
    epochs_path = os.path.join(data_path, config['dataset'], "eeg")
    features_path = os.path.join(data_path, "features")
    reports_path = os.path.join(base_path, "reports")
    models_path = os.path.join(base_path, "models")

    for path in [features_path, reports_path, models_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    return {
        "data": data_path,
        "audio": audio_path,
        "epochs": epochs_path,
        "features": features_path,
        "reports": reports_path,
        "models": models_path
    }

def get_field_from_filename(filename: str, target_field: str) -> str:
    fields = filename.split("_")
    for field in fields:
        field_parts = field.split("-")
        field_name = field_parts.pop(0)
        if field_name == target_field:
            return "-".join(field_parts)

    raise ValueError(f"Field {target_field} not found in filename {filename}")