#!/usr/bin/env python3

import numpy as np
from scipy.stats import spearmanr
import mne
import sys

PATHto = 'pathto'

# ---------------------------------------------------------------------
# Subject
# ---------------------------------------------------------------------
subj = sys.argv[1]

# ---------------------------------------------------------------------
# Load trial selection and inhibition measures
# ---------------------------------------------------------------------

# Number of valid epochs after head-position rejection
n_epo = np.load(
    PATHto + subj + '_epo_n_inh_headpos_auto.npy'
)

# Time of peak inhibition relative to unloading
inh_times = np.load(
    PATHto + subj +
    '_time_peakinh_unloading_and_inh_ontime_auto15_20ms_headpos1cm.npy'
)

# Select trials with inhibition peak occurring from -70 to +75 ms
# relative to unloading
n_early = np.argwhere(
    (inh_times >= -0.07) &
    (inh_times <= 0.075)
)[:, 0]

# Inhibition amplitude measure
inh = np.load(
    PATHto + subj +
    '_pampl_peakinh_unloading_and_inh_ontime_auto15_20ms_headpos1cm.npy'
)

print('Inhibition amplitude shape:', inh.shape)
print('Selected trials:', n_early)

# ---------------------------------------------------------------------
# Load source estimate 
# ---------------------------------------------------------------------
stc = mne.read_source_estimate(
    PATHto + subj + '_stc_mixed-stc.h5'
)

n_sources = sum(map(len, stc.vertices))
print('Number of sources:', n_sources)

# ---------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------

corr = []

for s in range(n_sources):

    corr_s = []

    # Alpha-beta source power:
    # 4–30 Hz, estimated with superlets
    data = np.load(
        PATHto +
        '/lcmv_periodic_alphabeta_sources_deep/' +
        subj + '/' +
        subj +
        '_periodic_power_4-30Hz_superlet_20_-03to03_epo_source_' +
        str(s) + '.npy'
    )

    print('Original data shape:', data.shape)

    # Keep only epochs corresponding to the valid MEG trials
    data = data[n_epo]

    print('After epoch selection:', data.shape)

    # -----------------------------------------------------------------
    # Restrict analysis to -250 to +100 ms relative to unloading
    # -----------------------------------------------------------------
    times = np.arange(-0.3, 0.3, 1 / 600)

    t1 = np.argmin(np.abs(times + 0.25))
    t2 = np.argmin(np.abs(times - 0.10))

    data = data[:, :, t1:t2]

    # -----------------------------------------------------------------
    # Spearman correlation between oscillatory power and inhibition
    #
    # Alpha-beta:
    # correlation is estimated separately for every frequency and time
    # point.
    #
    # Gamma:
    # the same procedure was applied to power averaged across the
    # 90–130 Hz gamma range, resulting in one gamma-power value per
    # time point.
    # -----------------------------------------------------------------

    for t in range(data.shape[2]):

        corr_t = []

        for f in range(data.shape[1]):

            # Correlation is calculated only for trials with inhibition
            # peaks occurring between -70 and +75 ms.
            sp = spearmanr(
                data[:, f, t][n_early],
                inh[n_early]
            )[0]

            corr_t.append(sp)

        corr_s.append(corr_t)

    corr.append(corr_s)

# ---------------------------------------------------------------------
# Save correlations
# ---------------------------------------------------------------------

corr = np.array(corr)

print('Correlation shape:', corr.shape)

np.save(
    PATHto + subj + '_corr_superlet_4-30Hz_-025to01_vs_inhampl_earlylateinh_2hemi.npy', corr)