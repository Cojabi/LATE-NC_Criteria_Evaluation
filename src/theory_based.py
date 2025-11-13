import pandas as pd
import numpy as np


class Theory_Based_Approach():

    def __init__(self, param_dict):
        self.cog_ep_cutoff = param_dict['cog_ep_cutoff']
        self.cog_ps_cutoff = param_dict['cog_ps_cutoff']
        self.mri_ratio_cutoff = param_dict['mri_ratio_cutoff']

 
    def set_params(self, param_dict):
        self.cog_ep_cutoff = param_dict['cog_ep_cutoff']
        self.cog_ps_cutoff = param_dict['cog_ps_cutoff']
        self.mri_ratio_cutoff = param_dict['mri_ratio_cutoff']


    def predict(self, X, colname_dict, check_ab_status=True, check_tau_status=True):
        """Run the data down the tree to get the predicted labels."""
        # Filter by episodic memory
        LATE_cases = X[X[colname_dict['cog_ep_col']] < self.cog_ep_cutoff]
        # Filter by perceptual speed
        LATE_cases = X[X[colname_dict['cog_ps_col']] > self.cog_ps_cutoff]
        # Filter by MRI atrophy
        LATE_cases = X[X[colname_dict['mri_ratio_col']] < self.mri_ratio_cutoff]
        
        # Filter by Ab status
        if check_ab_status: 
            LATE_cases = X[X[colname_dict['ab_status_col']] == True]
        # Filter by tau
        if check_tau_status:
            LATE_cases = X[X[colname_dict['tau_status_col']] == True]

        # get IDs from LATE cases and safe predictions
        LATE_cases_ids = LATE_cases.index
        predictions = pd.Series(np.zeros(X.shape[0]), index=X.index)
        predictions[LATE_cases_ids] = 1

        return list(predictions)
