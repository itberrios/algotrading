'''
    Utility functions for motifs
'''

import numpy as np
import pandas as pd



def filter_motif_indexes(matrix_profile, thresh):
    ''' Removes motif indexes that fall withing the threshold distance 
        from eachother
        Inputs:
            matrix_profile (array) matrix profile output from stump()
            thresh (int) - threshold for similar motifs spaced in time 
        Outputs:    
            usefule_motif_indexes (list) filtered motif indexes
        '''
    # store motif indexes to ensure that matches close in time are not plotted
    motif_indexes = []
    useful_motif_indexes = []

    i = 0
    while i < matrix_profile.shape[0]:
        # get motifs
        motif_idx = np.argsort(matrix_profile[:, 0])[i]
        nearest_neighbor_idx = matrix_profile[motif_idx, 1]

        if i == 0:
            useful_motif_indexes.append(motif_idx)

            # store motif index
            motif_indexes.append(motif_idx)

            # increment i
            i += 1

            continue
        
        # only store motifs with adequate spacing
        if (np.min(np.abs(motif_idx - np.array(motif_indexes))) > thresh):
            useful_motif_indexes.append(motif_idx)

        # store motif index
        motif_indexes.append(motif_idx)
        
        # increment i
        i += 1

    return useful_motif_indexes
  
  
def clean_motifs(useful_motif_indexes, matrix_profile, df, window, ext_window):
    ''' Removes motifs that fall outside of the 1 day trading window. If a motif
        falls outside the 1 day regular trading hours, than it will have false
        implications of rising or falling prices.
        Inputs:
            useful_motif_indexes (list) motif indexes where closly spaced motifs
                                        have been removed
            matrix_profile (array) matrix_profile output of stump()
            df (DataFrame/Series) Contains time series for motifs (with datetime index)
            window (int) window used to obtain motifs
            ext_window (int) window for external motif look ahead/behind
        Outputs:
            motifs (array) - cleaned motifs (spaced apart and do not overlap days)
            motif_idxs (array) - start DateTime indexes for each motif
            ext_motifs (array) - extended motifs (spaced apart, may overlap days)
            ext_outside (array) indicates whether an extended motif overlaps a day
        '''
    motifs = []
    motif_idxs = []
    ext_motifs = []
    ext_outside = [] # indicates if extended motif is outside of the day window

    # motif extension window_4hr
    ext_wind = 3

    for motif_idx in useful_motif_indexes:
        nearest_neighbor_idx = matrix_profile[motif_idx, 1]
        motif = df.iloc[nearest_neighbor_idx:nearest_neighbor_idx + window]
        ext_motif = df.iloc[nearest_neighbor_idx - ext_window:nearest_neighbor_idx + window + ext_window]

        # only include motifs that take place over the course of multiple days
        if len(pd.to_datetime(motif.index).dayofweek.unique()) == 1:
            motifs.append(motif.to_numpy())
            motif_idxs.append(nearest_neighbor_idx) # start index of motif
            ext_motifs.append(ext_motif.to_numpy())

            # get flag to determine if extended motif fall outside of the day window_4hr
            if len(pd.to_datetime(ext_motif.index).dayofweek.unique()) == 1:
                ext_outside.append(0)
            else:
                ext_outside.append(1)

    motifs = np.array(motifs)
    motif_idxs = np.array(motif_idxs)
    ext_motifs = np.array(ext_motifs)
    ext_outside = np.array(ext_outside)

    return motifs, motif_idxs, ext_motifs, ext_outside
