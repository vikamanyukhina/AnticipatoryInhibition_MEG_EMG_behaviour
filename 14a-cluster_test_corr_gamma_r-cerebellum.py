#!/usr/bin/env python3

import numpy as np
import mne
from scipy import stats
from functools import partial
from mne.stats import spatio_temporal_cluster_1samp_test, combine_adjacency, ttest_1samp_no_p

PATHto = 'pathto'
subjects_dir = PATHto + '/MRI_freesurfer/'
subjects = [f'BM_C{ii:02d}' for ii in range(1, 25)]

# Right-cerebellum volume source space and spatial adjacency
bem = mne.make_bem_model('fsaverage', ico=4, conductivity=(0.3,), subjects_dir=subjects_dir)
vol_src = mne.setup_volume_source_space('fsaverage', bem=bem, volume_label=['Right-Cerebellum-Cortex'], pos=5, subjects_dir=subjects_dir)
adjacency = mne.spatial_src_adjacency(vol_src)

# Load morphed correlations: subjects × source × time (gamma averaged across 90–130 Hz)
X = np.array([np.load(PATHto + subj + '_corr_multitaper_bw4_gamma90-130av_-015to015_aroundinh_vs_earlylateinh_peak-onset0-based_pampl_auto_2hemi_morphed.npy') for subj in subjects])
times = np.arange(-0.15, 0.1501, 1 / 600)
t1, t2 = np.argmin(np.abs(times + 0.1)), np.argmin(np.abs(times - 0)) + 1
X = np.transpose(X[:, :, t1:t2], [0, 2, 1])  # subjects × time × source

# Restrict to right cerebellum
n_cereb = len(vol_src[0]['vertno'])
X = X[:, :, :n_cereb]

n_subjects, n_times = X.shape[:2]
adjacency = combine_adjacency(n_times, adjacency)
t_threshold = stats.t.ppf(1 - 0.001, df=n_subjects - 1)

T_obs, clusters, p_values, H0 = spatio_temporal_cluster_1samp_test(X, threshold=t_threshold, adjacency=adjacency, stat_fun=partial(ttest_1samp_no_p, sigma=1e-3), tail=1, n_permutations=5000, buffer_size=None)

np.save(PATHto + 'corr_gamma90-130av_-01to0_aroundinh_earlylateinh_vs_inhampl_r-cerebellum_noTFCE_p0001_tail1.npy', T_obs)