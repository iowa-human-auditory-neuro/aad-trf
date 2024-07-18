import numpy as np
import pandas as pd
from mne import read_epochs
import os
import glob

import argparse
import json
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
        grid.fit(dataset.features, dataset.labels)
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

if __name__ == "__main__":
    pass