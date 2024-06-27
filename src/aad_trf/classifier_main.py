
import numpy as np
import pandas as pd
import os
import glob
import argparse
import matplotlib.pyplot as plt

from aad_dataset import AAD_Dataset
from aad_classifier import AAD_Classifier
from aad_plotter import plot_clf_results
from utils import load_config, set_paths_from_config, get_field_from_filename

parser = argparse.ArgumentParser()
parser.add_argument("--config_id", type=str, default="dataset-updown-nh_exp-1", help="Configuration ID")
args = parser.parse_args()
config_id = args.config_id

base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
config = load_config(config_id)
config_trf = config['trf']
path_dict = set_paths_from_config(base_path, config)
config_clf = config['classifiers']

files = glob.glob(f"{path_dict['features']}/{config_id}_data-aad-trfscores*.pkl")
files.sort()

clf_results = pd.DataFrame(columns=["test_sub_id", 
                                    "classifier", 
                                    "accuracy_train", 
                                    "accuracy_test", 
                                    "precision", 
                                    "recall", 
                                    "f1"])
for file in files:
    filename = file.split("/")[-1].split(".")[0]
    test_sub_id = get_field_from_filename(filename, "sub")
    print(f"Test subject: {test_sub_id}")
    dataset = AAD_Dataset.load_from_file(file)
    
    train_dataset = dataset[dataset.sub_ids != test_sub_id]
    test_dataset = dataset[dataset.sub_ids == test_sub_id]
    print(train_dataset.features.shape, test_dataset.features.shape)

    clf = AAD_Classifier(model_name=config_clf['model_name'])
    optimization_results = clf.optimize_hyperparmeters(train_dataset, n_splits=10)
    optimization_results_filename = f"{config_id}_reports-clf-optimization_models-{config_clf['model_name']}_sub-{test_sub_id}"
    optimization_results.to_csv(os.path.join(path_dict['reports'], f"{optimization_results_filename}.csv"))

    clf.train(train_dataset)
    print(f"Accuracy Train: {clf.eval(train_dataset):.3f}")
    print(f"Accuracy Test: {clf.eval(test_dataset):.3f}")
    model_filename = f"{config_id}_models-{config_clf['model_name']}_sub-{test_sub_id}"
    clf.save(os.path.join(path_dict['models'], f"{model_filename}.pkl"))

    results_dict = {"test_sub_id": test_sub_id, 
                    "classifier": clf.model_name,
                    "accuracy_train": clf.eval(train_dataset),
                    "accuracy_test": clf.eval(test_dataset),
                    "precision": clf.eval(test_dataset, scoring="precision"), 
                    "recall": clf.eval(test_dataset, scoring="recall"), 
                    "f1": clf.eval(test_dataset, scoring="f1")
                    }
    clf_results = pd.concat([clf_results, pd.DataFrame([results_dict])])

clf_results_filename = f"{config_id}_reports-clf-performance_models-{config_clf['model_name']}"
clf_results.to_csv(os.path.join(path_dict['reports'], f"{clf_results_filename}.csv"))

# clf_results = pd.read_csv(os.path.join(results_path, f"classification_results_{config_clf['model_name']}.csv"))
ax = plot_clf_results(clf_results)
plt.savefig(os.path.join(path_dict['reports'], f"{clf_results_filename}.png"))