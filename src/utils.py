import numpy as np

from optuna.trial import Trial

from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score

# utility functions to logg results in optuna study object
def store_scores(trial:Trial, scores:dict) -> None:
        """store scores in trial object, compute mean and std
        the array holds the values across folds"""
        
        for name, array in scores.items():
            
            if len(array) > 0:
                for i, score in enumerate(array):
                    trial.set_user_attr(f"split{i}_{name}", score)
            
            # creates the 'mean_val_score' that will be optimized.
            trial.set_user_attr(f"mean_{name}", np.nanmean(array))
            trial.set_user_attr(f"std_{name}", np.nanstd(array))

def get_performance(scores, y_train, predicted_labels_train, y_val, predicted_labels_val, fold=""):
     # get performance
        scores["train_AUPR"].append(average_precision_score(y_train, predicted_labels_train))
        scores["train_AUC"].append(roc_auc_score(y_train, predicted_labels_train))
        scores["train_Prec"].append(precision_score(y_train, predicted_labels_train))
        scores["train_Rec"].append(recall_score(y_train, predicted_labels_train))
        # validation scores
        scores["val_AUPR"].append(average_precision_score(y_val, predicted_labels_val))
        scores["val_AUC"].append(roc_auc_score(y_val, predicted_labels_val))
        scores["val_Prec"].append(precision_score(y_val, predicted_labels_val))
        scores["val_Rec"].append(recall_score(y_val, predicted_labels_val))

        print(f'\nTrain Prec:{scores["train_Prec"][fold]}; Train Rec: {scores["train_Rec"][fold]}; VAL Prec: {scores["val_Prec"][fold]}; Val Rec: {scores["val_Rec"][fold]}')
        print(f'Train AUPR:{scores["train_AUPR"][fold]}; Train AUC: {scores["train_AUC"][fold]}; VAL AUPR: {scores["val_AUPR"][fold]}; Val AUC: {scores["val_AUC"][fold]}\n')
        return scores