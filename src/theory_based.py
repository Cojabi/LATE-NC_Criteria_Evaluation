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
        LATE_cases = X[X[colname_dict['cog_ep_col']] < self.cog_ep_cutoff].copy()
        # Filter by perceptual speed
        LATE_cases = LATE_cases[LATE_cases[colname_dict['cog_ps_col']] > self.cog_ps_cutoff]
        # Filter by MRI atrophy
        LATE_cases = LATE_cases[LATE_cases[colname_dict['mri_ratio_col']] < self.mri_ratio_cutoff]

        # Amyloid branch
        amyloid_no = LATE_cases[LATE_cases[colname_dict['ab_status_col']] == False]
        amyloid_yes = LATE_cases[LATE_cases[colname_dict['ab_status_col']] == True]
 
        # Tau branch (only for Amyloid yes)
        tau_no = amyloid_yes[amyloid_yes[colname_dict['tau_status_col']] == False]
 
        # Combine IDs for predicted Disease
        late_ids = pd.Index(amyloid_no.index.tolist() + tau_no.index.tolist()).unique()
 
        # Assign 1 to Disease_predicted

        predictions = pd.Series(np.zeros(X.shape[0]), index=X.index)
        predictions[late_ids] = 1
 
        return list(predictions)
    

