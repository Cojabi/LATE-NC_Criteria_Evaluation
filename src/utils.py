import numpy as np

from optuna.trial import Trial

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