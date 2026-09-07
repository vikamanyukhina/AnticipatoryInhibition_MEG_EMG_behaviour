#!/usr/bin/env python3

import mne
import numpy as np
from mne.preprocessing import maxwell_filter


# Paths and experiment
subj_type = 'Children/Control'
PATHfrom = 'pathfrom' + subj_type
PATHto = 'pathto' + subj_type
exp = 'naturel'

SUBJECTS = ['BM_C0' + str(ii) for ii in np.arange(1, 10)] + ['BM_C' + str(ii) for ii in np.arange(10, 25)]

# Load original and cleaned average head positions
oris = np.load(PATHfrom + 'Allsubjs_origin_headpos.npy')
new_pos = np.load(PATHfrom + 'Allsubjs_good_av_headpos.npy')


for s, subj in enumerate(SUBJECTS):

    # Load preprocessed raw MEG data
    raw_fname = PATHfrom + '/' + subj + '/ica_notch/' + subj + '_' + exp + '_indpos_ica-ref_thr2_raw.fif'
    raw = mne.io.Raw(raw_fname, preload=True)

    # Apply Maxwell filtering and move data to the cleaned average head position
    raw_new = maxwell_filter(raw, origin='auto', coord_frame='head', destination=new_pos[s], head_pos=None)

    # Save the head-position-corrected data
    raw_new.save(PATHto + subj + '_' + exp + '_indpos_ica-ref_thr2_newpos_raw.fif', overwrite=True)