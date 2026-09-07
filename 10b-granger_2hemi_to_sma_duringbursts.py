#!/usr/bin/env python3

import os
import sys
import numpy as np
import mne
from mne_connectivity import spectral_connectivity_epochs

print(mne.__version__)

PATHfrom = 'pathfrom'
PATHto = 'pathto'

subjects_dir = PATHfrom +'/MRI_freesurfer/'

# Analysis parameters
dop = 'reg01'
dia = '25-150'
f1, f2 = 25, 150
ica = 'ica'
type_movement = 'Natural_movement'
type_stimulus = 'naturel_action_start'
type_behav = 'naturel_elbow_rot'
epoch_type = '-lagcorrected-epo_prevstim-based.fif'
sfreq = 600

SUBJECTS = [f'BM_C{ii:02d}' for ii in range(1, 25)]



def match_length(noburst_trials, burst_trials):
    """Match the number of no-burst trials to the burst trials."""
    nob = list(noburst_trials)
    bur = list(burst_trials)

    if len(nob) >= len(bur):
        return nob[:len(bur)]

    extra = np.random.choice(nob, size=len(bur) - len(nob), replace=True)
    return nob + extra.tolist()


subject = sys.argv[1]

# -------------------------------------------------------------------------
# Load source-level data and inhibition timing
# -------------------------------------------------------------------------

data = np.load(PATHto + subject + '_all_lcmv_maxori_deep_allepo_indepo_reg01_for_alphabeta_nof_-17-12_epoxvertxtimes_goodheadpos_test_morphed.npy')
n_epo = np.load(PATHto + subject + '_epo_n_inh_headpos_auto.npy')
data = data[n_epo]

inh_times = np.load(PATHto + subject + '_time_peakinh_unloading_and_inh_ontime_auto15_20ms_headpos1cm.npy')

# -------------------------------------------------------------------------
# Load beta-burst information
# -------------------------------------------------------------------------

burst_times1 = np.load(PATHto + subject + '_betaburst20-24_times_new.npy')
burst_trials1 = np.load(PATHto + subject + '_betaburst20-24_trials_new.npy')
burst_verts1 = np.load(PATHto + subject + '_betaburst20-24_verts_new.npy')

burst_times2 = np.load(PATHto + subject + '_betaburst24-30_times_new.npy')
burst_trials2 = np.load(PATHto + subject + '_betaburst24-30_trials_new.npy')
burst_verts2 = np.load(PATHto + subject + '_betaburst24-30_verts_new.npy')

# Remove trials detected in both beta-frequency ranges
overlap_trials = np.intersect1d(burst_trials1, burst_trials2)

idx1 = np.isin(burst_trials1, overlap_trials)
idx2 = np.isin(burst_trials2, overlap_trials)

overlap_trials = burst_trials1[idx1].tolist() + burst_trials2[idx2].tolist()

idx1 = ~np.isin(burst_trials1, overlap_trials)
idx2 = ~np.isin(burst_trials2, overlap_trials)

burst_times1 = burst_times1[idx1]
burst_trials1 = burst_trials1[idx1]
burst_verts1 = burst_verts1[idx1]

burst_times2 = burst_times2[idx2]
burst_trials2 = burst_trials2[idx2]
burst_verts2 = burst_verts2[idx2]

# Use 20–24 Hz burst trials for the connectivity analysis
epo_max = burst_trials1
times_burst = burst_times1
verts_burst = burst_verts1

print('Number of burst trials:', len(epo_max))

# -------------------------------------------------------------------------
# Select SMA sources and construct burst-aligned epochs
# -------------------------------------------------------------------------

sma_sel = [3034, 3595, 3596, 3597]
data_sma = data[:, sma_sel, :]

data_max = data[epo_max]
label_max = data_sma[epo_max]

print('Data shape:', data.shape)
print('Burst data shape:', data_max.shape)
print('SMA data shape:', label_max.shape)

times = np.arange(-1.7, 1.2, 1 / sfreq)

contrast_max = []

for ei, e in enumerate(epo_max):
    t1 = np.argmin(np.abs(times - (inh_times[e] + times_burst[ei] - 1.5)))
    t2 = np.argmin(np.abs(times - (inh_times[e] + times_burst[ei] + 1.1)))

    sma_burst = np.expand_dims(label_max[ei, verts_burst[ei], t1:t2], axis=0)
    contrast_max.append(np.concatenate([data_max[ei, :, t1:t2], sma_burst], axis=0))

contrast_max = np.array(contrast_max)

# Time window for Granger connectivity: -100 to +50 ms around inhibition
times = np.arange(-1.5, 1.1, 1 / sfreq)
t1 = np.argmin(np.abs(times + 0.1))
t2 = np.argmin(np.abs(times - 0.05)) + 1

# -------------------------------------------------------------------------
# Calculate Granger connectivity for each source
# -------------------------------------------------------------------------

lll = np.array(list(range(0, 3597)) + list(range(3598, 6480)))

for s in lll:

    print('Processing source:', s)

    source_idx = np.array([[s]])
    sma_idx = np.array([[data.shape[1]]])

    # -------------------------------------------------------------
    # Granger: source -> SMA
    # -------------------------------------------------------------

    indices = (source_idx, sma_idx)

    coh = spectral_connectivity_epochs(
        contrast_max,
        method='gc',
        mode='cwt_morlet',
        indices=indices,
        sfreq=sfreq,
        fmin=3,
        fmax=33,
        gc_n_lags=30,
        cwt_n_cycles=4,
        cwt_freqs=np.arange(3, 34),
        faverage=False,
        n_jobs=1
    )

    coh = coh.get_data()[0, :, t1:t2]

    np.save(PATHfrom + '/granger_2hemi/max/' + subject + '/' + subject +
            '_granger_2hemi_vs_4SMA_in-01to005_aroundburst19-24_morlet4_30_3-33Hz_in_time_max_bursts2sd_epo_source' +
            str(s) + '.npy', coh)

    del coh

    # -------------------------------------------------------------
    # Granger: SMA -> source
    # -------------------------------------------------------------

    indices = (sma_idx, source_idx)

    coh = spectral_connectivity_epochs(
        contrast_max,
        method='gc',
        mode='cwt_morlet',
        indices=indices,
        sfreq=sfreq,
        fmin=3,
        fmax=33,
        gc_n_lags=30,
        cwt_n_cycles=4,
        cwt_freqs=np.arange(3, 34),
        faverage=False,
        n_jobs=1
    )

    coh = coh.get_data()[0, :, t1:t2]

    np.save(PATHfrom + '/granger_2hemi/min/' + subject + '/' + subject +
            '_granger_2hemi_vs_4SMA_in-01to005_aroundburst19-24_morlet4_30_3-33Hz_in_time_min_bursts2sd_epo_source' +
            str(s) + '.npy', coh)

    del coh

    # -------------------------------------------------------------
    # Time-reversed Granger: source -> SMA
    # -------------------------------------------------------------

    indices = (source_idx, sma_idx)

    coh = spectral_connectivity_epochs(
        contrast_max,
        method='gc_tr',
        mode='cwt_morlet',
        indices=indices,
        sfreq=sfreq,
        fmin=3,
        fmax=33,
        gc_n_lags=30,
        cwt_n_cycles=4,
        cwt_freqs=np.arange(3, 34),
        faverage=False,
        n_jobs=1
    )

    coh = coh.get_data()[0, :, t1:t2]

    np.save(PATHfrom + '/granger_2hemi/max_inv/' + subject + '/' + subject +
            '_granger_inv_2hemi_vs_4SMA_in-01to005_aroundburst19-24_morlet4_30_3-33Hz_in_time_max_bursts2sd_epo_source' +
            str(s) + '.npy', coh)

    del coh

    # -------------------------------------------------------------
    # Time-reversed Granger: SMA -> source
    # -------------------------------------------------------------

    indices = (sma_idx, source_idx)

    coh = spectral_connectivity_epochs(
        contrast_max,
        method='gc_tr',
        mode='cwt_morlet',
        indices=indices,
        sfreq=sfreq,
        fmin=3,
        fmax=33,
        gc_n_lags=30,
        cwt_n_cycles=4,
        cwt_freqs=np.arange(3, 34),
        faverage=False,
        n_jobs=1
    )

    coh = coh.get_data()[0, :, t1:t2]

    np.save(PATHfrom + '/granger_2hemi/min_inv/' + subject + '/' + subject +
            '_granger_inv_2hemi_vs_4SMA_in-01to005_aroundburst19-24_morlet4_30_3-33Hz_in_time_min_bursts2sd_epo_source' +
            str(s) + '.npy', coh)

    del coh
    
    
    
for subj in SUBJECTS:

    print(subj)

    contrast_s = []

    for s in range(6479):

        if s in [3034, 3595, 3596, 3597]:

            burst_max = np.ones(burst_max.shape)
            burst_max_inv = np.ones(burst_max.shape)
            noburst_min = np.ones(burst_max.shape)
            noburst_min_inv = np.ones(burst_max.shape)

        else:

            burst_max_inv = np.load(PATHfrom + 'granger_2hemi/max_inv/' + subj + '/' + subj + '_granger_inv_2hemi_vs_4SMA_in-01to005_aroundburst19-24_morlet6_30_3-33Hz_in_time_max_bursts2sd_epo_source' + str(s) + '.npy')
            burst_max = np.load(PATHfrom + 'granger_2hemi/max/' + subj + '/' + subj + '_granger_2hemi_vs_4SMA_in-01to005_aroundburst19-24_morlet6_30_3-33Hz_in_time_max_bursts2sd_epo_source' + str(s) + '.npy')

            burst_min_inv = np.load(PATHfrom + 'granger_2hemi/min_inv/' + subj + '/' + subj + '_granger_2hemi_vs_4SMA_in-01to005_aroundburst19-24_morlet6_30_3-33Hz_in_time_min_bursts2sd_epo_source' + str(s) + '.npy')
            burst_min = np.load(PATHfrom + 'granger_2hemi/min/' + subj + '/' + subj + '_granger_2hemi_vs_4SMA_in-01to005_aroundburst19-24_morlet6_30_3-33Hz_in_time_min_bursts2sd_epo_source' + str(s) + '.npy')

        #burst_max = np.log10(burst_max)
        #burst_min = np.log10(burst_min)
        #noburst_max = np.log10(noburst_max)
        #noburst_min = np.log10(noburst_min)

        contrast = (burst_max - burst_min) - (burst_max_inv - burst_min_inv)
        contrast_s.append(contrast)

    np.save(PATHfrom + subj + '_granger_bursts_inv_back_2hemi_vs_4SMA_in-01to005_aroundburstHz_morlet6_3-33Hz.npy', np.array(contrast_s))