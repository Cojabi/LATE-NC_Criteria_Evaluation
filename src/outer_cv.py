from typing import Any, Dict
import numpy as np

import optuna
from optuna.trial import Trial
from optuna.distributions import FloatDistribution, IntDistribution
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

from inner_cv import Theory_driven_approach_inner_CV
from theory_based import Theory_Based_Approach
from data_loader import load_data
from constants_rosmap import DATA_PATH, LABEL_COL, ROSMAP_COLS


def main(n_cv_folds=5, n_trials=100, n_jobs=4):
    X, y = load_data(DATA_PATH, LABEL_COL)

    parameter_space = {"cog_ep_cutoff":FloatDistribution(0.1, 2.5), 
                    "cog_ps_cutoff":FloatDistribution(0.1, 2.5), 
                    "mri_ratio_cutoff":FloatDistribution(0.01, 1.5)}

    # containers 
    scores = {
            # train
            "train_AUPR" : [],
            "train_AUC" : [],
            # val
            "val_AUPR" : [],
            "val_AUC" : [],
            # store cut-offs
            "cog_ep_cutoff" : [],
            "cog_ps_cutoff" : [],
            "mri_ratio_cutoff" : []
            }

    # get cross-validated estimate for one particular hyperparameter set
    cv = StratifiedKFold(n_splits=n_cv_folds, shuffle=True)
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]    
        y_val = y.iloc[val_idx]

        objective = Theory_driven_approach_inner_CV(parameter_space, X_train, y_train, colname_dict=ROSMAP_COLS)


        study = optuna.create_study(study_name='xGBoost',
                                                storage=f'sqlite:///optuna__LATE_{fold}.db', 
                                                direction = "minimize", 
                                                load_if_exists = True)
        study.optimize(objective, 
                        n_trials = n_trials,
                        n_jobs = n_jobs)

        # Test outer fold
        print(study.best_params)
        model = Theory_Based_Approach(study.best_params)
        predicted_labels_train = model.predict(X_train, colname_dict=ROSMAP_COLS)
        predicted_labels_val = model.predict(X_val, colname_dict=ROSMAP_COLS)

        # get performance
        scores["train_AUPR"].append(average_precision_score(y_train, predicted_labels_train))
        scores["train_AUC"].append(roc_auc_score(y_train, predicted_labels_train))
        # validation scores
        scores["val_AUPR"].append(average_precision_score(y_val, predicted_labels_val))
        scores["val_AUC"].append(roc_auc_score(y_val, predicted_labels_val))

        print(f'Fold {fold}:\nTrain AUPR:{scores["train_AUPR"]}; Train AUC: {scores["train_AUC"]}; VAL AUPR: {scores["val_AUPR"]}; Val AUC: {scores["val_AUC"]}')

    # print final results
    print(f'Train AUPR:{np.mean(scores["train_AUPR"])}; Train AUC: {np.mean(scores["train_AUC"])}; VAL AUPR: {np.mean(scores["val_AUPR"])}; Val AUC: {np.mean(scores["val_AUC"])}')
    # print best cut-offs
    print(f'cog_ep_cutoffs:: {scores["cog_ep_cutoff"]}')
    print(f'cog_ps_cutoffs: {scores["cog_ps_cutoff"]}') 
    print(f'MRI_ratio_cutoffs:{scores["mri_ratio_cutoff"]}')
    print(f'Best cut-offs mean:\ncog_ep_cutoff: {np.mean(scores["cog_ep_cutoff"])}, cog_ps_cutoff: {np.mean(scores["cog_ps_cutoff"])}, MRI_ratio_cutoff:{np.mean(scores["mri_ratio_cutoff"])}')
    print(f'Best cut-offs std:\ncog_ep_cutoff: {np.std(scores["cog_ep_cutoff"])}, cog_ps_cutoff: {np.std(scores["cog_ps_cutoff"])}, MRI_ratio_cutoff:{np.std(scores["mri_ratio_cutoff"])}')


