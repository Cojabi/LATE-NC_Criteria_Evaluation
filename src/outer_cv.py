from typing import Any, Dict
import numpy as np

import optuna
from sklearn.model_selection import StratifiedKFold

from inner_cv import Theory_driven_approach_inner_CV
from theory_based import Theory_Based_Approach
from data_loader import load_data
from constants_rosmap import DATA_PATH, LABEL_COL, ROSMAP_COLS, PARAMETER_SPACE
from utils import get_performance


def main(n_cv_folds=5, n_trials=100, n_jobs=4):
    X, y = load_data(DATA_PATH, LABEL_COL)

    # containers 
    scores_opt = {
            # train
            "train_AUPR" : [],
            "train_AUC" : [],
            "train_Prec" : [],
            "train_Rec" : [],
            # val
            "val_AUPR" : [],
            "val_AUC" : [],
            "val_Prec" : [],
            "val_Rec" : [],
            # store cut-offs
            "cog_ep_cutoff" : [],
            "cog_ps_cutoff" : [],
            "mri_ratio_cutoff" : []
            }
    
    scores_tree = {
            # train
            "train_AUPR" : [],
            "train_AUC" : [],
            "train_Prec" : [],
            "train_Rec" : [],
            # val
            "val_AUPR" : [],
            "val_AUC" : [],
            "val_Prec" : [],
            "val_Rec" : []
            }
    
    scores_RF = {
            # train
            "train_AUPR" : [],
            "train_AUC" : [],
            "train_Prec" : [],
            "train_Rec" : [],
            # val
            "val_AUPR" : [],
            "val_AUC" : [],
            "val_Prec" : [],
            "val_Rec" : []
            }
    


    # get cross-validated estimate for one particular hyperparameter set
    cv = StratifiedKFold(n_splits=n_cv_folds, shuffle=True)
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        print(f'\nFold {fold}:')

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]    
        y_val = y.iloc[val_idx]


        ### HANDLE OPT LOGIC
        objective = Theory_driven_approach_inner_CV(PARAMETER_SPACE, X_train, y_train, colname_dict=ROSMAP_COLS)


        study = optuna.create_study(study_name='LATE',
                                    storage=f'sqlite:///optuna__LATE_{fold}.db', 
                                    direction = "maximize", 
                                    load_if_exists = True)
        study.optimize(objective, 
                        n_trials = n_trials,
                        n_jobs = n_jobs)
        
        # store best params
        scores_opt['cog_ep_cutoff'].append(study.best_params['cog_ep_cutoff'])
        scores_opt['cog_ps_cutoff'].append(study.best_params['cog_ps_cutoff'])
        scores_opt['mri_ratio_cutoff'].append(study.best_params['mri_ratio_cutoff'])

        # Test outer fold
        print(f'Best params: {study.best_params}')
        model = Theory_Based_Approach(study.best_params)
        predicted_labels_train = model.predict(X_train, colname_dict=ROSMAP_COLS)
        predicted_labels_val = model.predict(X_val, colname_dict=ROSMAP_COLS)

        scores_opt = get_performance(scores_opt, y_train, predicted_labels_train, y_val, predicted_labels_val, fold=fold)


        ### HANDLE TREE LOGIC
        model = FUNCTION FOR DEC TREE()
        predicted_labels_train = model.predict(X_train)
        predicted_labels_val = model.predict(X_val)

        scores_opt = get_performance(scores_opt, y_train, predicted_labels_train, y_val, predicted_labels_val, fold=fold)


        ### HANDLE RANDOMFOREST LOGIC
        model = RF()

        # implement hyperparameter opt reusing some of this
        # Theory_driven_approach_inner_CV(PARAMETER_SPACE, X_train, y_train, colname_dict=ROSMAP_COLS)

        predicted_labels_train = model.predict(X_train)
        predicted_labels_val = model.predict(X_val)

        scores_opt = get_performance(scores_opt, y_train, predicted_labels_train, y_val, predicted_labels_val, fold=fold)



    for scores, model_name in zip([scores_opt, scores_tree, scores_RF], ['Bayesian Opt.', 'Dec. tree', 'RandomForest']):
        # print final results
        print(f"\n#### Aggregated results {model_name}:")
        print(f'Train AUPR:{np.mean(scores["train_AUPR"])}; Train AUC: {np.mean(scores["train_AUC"])}; VAL AUPR: {np.mean(scores["val_AUPR"])}; Val AUC: {np.mean(scores["val_AUC"])}')
        print(f'Train Prec:{np.mean(scores["train_Prec"])}; Train Rec: {np.mean(scores["train_Rec"])}; VAL Prec: {np.mean(scores["val_Prec"])}; Val Rec: {np.mean(scores["val_Rec"])}\n')
        try:
            # print best cut-offs
            print(f'cog_ep_cutoffs:: {scores["cog_ep_cutoff"]}')
            print(f'cog_ps_cutoffs: {scores["cog_ps_cutoff"]}') 
            print(f'MRI_ratio_cutoffs:{scores["mri_ratio_cutoff"]}')
            print(f'Best cut-offs mean:\ncog_ep_cutoff: {np.mean(scores["cog_ep_cutoff"])}, cog_ps_cutoff: {np.mean(scores["cog_ps_cutoff"])}, MRI_ratio_cutoff:{np.mean(scores["mri_ratio_cutoff"])}')
            print(f'Best cut-offs std:\ncog_ep_cutoff: {np.std(scores["cog_ep_cutoff"])}, cog_ps_cutoff: {np.std(scores["cog_ps_cutoff"])}, MRI_ratio_cutoff:{np.std(scores["mri_ratio_cutoff"])}')
        except KeyError:
            pass




