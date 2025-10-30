
import optuna
from optuna.trial import Trial

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

from theory_based import Theory_Based_Approach
from utils import store_scores


class Theory_driven_approach_inner_CV():
    """Inner loop for nested cross-validation of a XGBoost regressor.
    
    param_distributions : dict. Dictionary with hyperparameter names as keys and
        distributions as values.
    X : pd.DataFrame. Data.
    y : pd.Series. Labels.
    scoring : str. Metric to optimize (either "rmse" or "mape").
    cv : int. Number of folds for the inner cross-validation.
    """
    
    def __init__(
        self,
        param_space: dict,
        X,
        y,
        colname_dict,
        scoring : str = "AUPR",
        cv: int = 5,
        ) -> None:

        self.X = X
        self.y = y
        self.cv = cv
        self.colname_dict = colname_dict
        self.fit_params = None
        self.param_space = param_space
        self.scoring = scoring

    def __call__(self, trial: Trial) -> float:
        """callable for optuna search and returns the achieved score"""

        fit_params = self._get_params(trial) # sample suggested hyperparams
        cv = StratifiedKFold(n_splits = self.cv, shuffle=True)

        # containers 
        scores = {
           # train
           "train_AUPR" : [],
           "train_AUC" : [],
           # val
           "val_AUPR" : [],
           "val_AUC" : []
           }
        
        # get cross-validated estimate for one particular hyperparameter set
        for fold, (train_idx, val_idx) in enumerate(cv.split(self.X, self.y)):
        
            X_train = self.X.iloc[train_idx]
            y_train = self.y.iloc[train_idx]
            X_val = self.X.iloc[val_idx]    
            y_val = self.y.iloc[val_idx]
            
            # initialize model
            estimator = Theory_Based_Approach(fit_params)
            
            # predict
            y_train_pred = estimator.predict(X_train, self.colname_dict)
            y_val_pred = estimator.predict(X_val, self.colname_dict)

            # evaluate model performance
            ## train
            scores["train_AUC"].append(roc_auc_score(y_train, y_train_pred))
            scores["train_AUPR"].append(average_precision_score(y_train, y_train_pred))
            ## val
            scores["val_AUC"].append(roc_auc_score(y_val, y_val_pred))
            scores["val_AUPR"].append(average_precision_score(y_val, y_val_pred))

            # choose scoring to return for optuna to optimize
            if self.scoring == "AUC":
                scores["train_score"] = scores["train_AUC"]
                scores["val_score"] = scores["val_AUC"]
            elif self.scoring == "AUPR":
                scores["train_score"] = scores["train_AUPR"]
                scores["val_score"] = scores["val_AUPR"]
            else:
                raise ValueError("hp_objective must be either 'AUC' or 'AUPR'")

        # save scores in trial object
        store_scores(trial, scores)
        
        # the 'mean_' part is added in the store_scores function
        return trial.user_attrs["mean_val_score"]
    
    def _get_params(self, trial: Trial):
        """Get parameters from trial history"""

        return {name:trial._suggest(name, distribution) 
                for name, distribution in self.param_space.items()}