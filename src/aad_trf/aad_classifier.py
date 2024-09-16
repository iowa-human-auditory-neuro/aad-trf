import numpy as np
import pandas as pd

import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from aad_dataset import AAD_Dataset

class AAD_Classifier:
    def __init__(self, model_name, model=None, params=None):
        self.model_name = model_name
        self.model = model
        self.params = params

        if self.model_name == "logistic-regression":
            self.model = LogisticRegression()
        elif self.model_name == "random-forest":
            self.model = RandomForestClassifier()
        elif self.model_name == "svc-linear":
            self.model = SVC(kernel="linear", probability=True)
        elif self.model_name == "svc-rbf":
            self.model = SVC(kernel="rbf", probability=True)
        else:
            raise ValueError("Invalid classifier name")
    
    def optimize_hyperparmeters(self, 
                                dataset:AAD_Dataset, 
                                n_splits=10
                                ):
        grid = GridSearchCV(self.model, 
                            self.params, 
                            cv=n_splits, 
                            scoring="accuracy", 
                            n_jobs=-1, 
                            verbose=1
                            )
                            
        print("Finding best parameters...")
        features = dataset.get_features(normalize=True)
        grid.fit(features, dataset.labels)
        print(f"Best parameters: {grid.best_params_} with a score of {grid.best_score_:.3f}")
        self.model = clone(grid.best_estimator_)
        self.params = grid.best_params_

        return pd.DataFrame(grid.cv_results_)
    
    def train(self, dataset:AAD_Dataset):
        X = dataset.features
        y = dataset.labels
        self.model.fit(X, y)
    
    def eval(self, dataset:AAD_Dataset, scoring="accuracy"):
        y_true = dataset.labels
        y_pred = self.predict(dataset)
        if scoring == "accuracy":
            return accuracy_score(y_true, y_pred)
        elif scoring == "precision":
            return precision_score(y_true, y_pred)
        elif scoring == "recall":
            return recall_score(y_true, y_pred)
        elif scoring == "f1":
            return f1_score(y_true, y_pred)
        elif scoring == "roc_auc":
            return roc_auc_score(y_true, self.model.predict_proba(dataset.features)[:, 1])
        else:
            raise ValueError(f"Invalid scoring method {scoring}")
    
    def predict(self, dataset:AAD_Dataset):
        X = dataset.features
        return self.model.predict(X)
    
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
    
    def load(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)
        
    def main(args):
        import os
        import glob
        import matplotlib.pyplot as plt

        from aad_plotter import plot_clf_results
        from utils import load_config, set_paths_from_config, get_field_from_filename

        config_id = args.config_id
        config = load_config(config_id)
        config_id_trf = config['config_id_trf']
        config_trf = load_config(config_id_trf)

        dataset_name = config_trf['dataset']
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        path_dict = set_paths_from_config(base_path, config_trf)

        files = glob.glob(f"{path_dict['features']}/dataset-{dataset_name}_data-aad-trfscores_config-{config_id_trf}_sub-*.pkl")
        files.sort()
        model_name = config['model_name']
        results_by_trial_full = pd.DataFrame()
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

            clf = AAD_Classifier(model_name=config['model_name'])
            optimization_results = clf.optimize_hyperparmeters(train_dataset, n_splits=10)
            optimization_results_filename = f"dataset-{dataset_name}_reports-clf-optimization_models-{model_name}_config-{config_id}_sub-{test_sub_id}"
            # optimization_results.to_csv(os.path.join(path_dict['reports'], f"{optimization_results_filename}.csv"))

            clf.train(train_dataset)

            sub_ids = test_dataset.sub_ids
            trial = np.arange(1, len(test_dataset)+1)
            label = test_dataset.labels
            prob = clf.model.predict_proba(test_dataset.features)
            prediction = clf.predict(test_dataset)
            correct = label == prediction
            results_by_trial = pd.DataFrame({"sub_id": sub_ids, 
                                            "trial": trial, 
                                            "output1": prob[:, 0], 
                                            "output2": prob[:, 1], 
                                            "prediction": prediction, 
                                            "label": label, 
                                            "correct": correct})
            results_by_trial_full = pd.concat([results_by_trial_full, results_by_trial])

            print(f"Accuracy Train: {clf.eval(train_dataset):.3f}")
            print(f"Accuracy Test: {clf.eval(test_dataset):.3f}")
            model_filename = f"dataset-{dataset_name}_models-{model_name}_config-{config_id}_sub-{test_sub_id}"
            # clf.save(os.path.join(path_dict['models'], f"{model_filename}.pkl"))

            results_dict = {"test_sub_id": test_sub_id, 
                            "classifier": clf.model_name,
                            "accuracy_train": clf.eval(train_dataset),
                            "accuracy_test": clf.eval(test_dataset),
                            "precision": clf.eval(test_dataset, scoring="precision"), 
                            "recall": clf.eval(test_dataset, scoring="recall"), 
                            "f1": clf.eval(test_dataset, scoring="f1")
                            }
            clf_results = pd.concat([clf_results, pd.DataFrame([results_dict])])

        clf_results_by_trial_filename = f"dataset-{dataset_name}_reports-clf-results-by-trial_models-{model_name}_config-{config_id}"
        # results_by_trial_full.to_csv(os.path.join(path_dict['reports'], f"{clf_results_by_trial_filename}.csv"))

        clf_results_filename = f"dataset-{dataset_name}_reports-clf-performance_models-{model_name}_config-{config_id}"
        # clf_results.to_csv(os.path.join(path_dict['reports'], f"{clf_results_filename}.csv"))

        # clf_results = pd.read_csv(os.path.join(results_path, f"classification_results_{config['model_name']}.csv"))
        ax = plot_clf_results(clf_results)
        # plt.savefig(os.path.join(path_dict['reports'], f"{clf_results_filename}.png"))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_id", type=str, default="classifier-001", help="Configuration ID")
    args = parser.parse_args()
    AAD_Classifier.main(args)