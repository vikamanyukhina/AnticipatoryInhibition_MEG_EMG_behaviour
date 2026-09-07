#!/usr/bin/env python3

import numpy as np
import mne
from scipy import stats
from scipy.sparse import coo_matrix
from functools import partial
from mne.stats import spatio_temporal_cluster_1samp_test, combine_adjacency, ttest_1samp_no_p

PATHto = 'pathto'
subjects_dir = PATHto + 'MRI_freesurfer/'
subjects = [f'BM_C{ii:02d}' for ii in range(1, 25)]


# Source-space adjacency and right-hemisphere motor ROI
src = mne.read_source_spaces(PATHto + 'src_fsaverage_ico4_test')
stc = mne.read_source_estimate(PATHto + 'stc_fsaverage_ico4_test')
adjacency = mne.spatial_src_adjacency(src)
labels = mne.read_labels_from_annot('fsaverage', 'HCPMMP1', 'rh', subjects_dir=subjects_dir)

lab_m1 = labels[27]
lab_sma = labels[124] + labels[35] + labels[36]
lab_pm = labels[29] + labels[34] + labels[33] + labels[62] + labels[38] + labels[37] + labels[97]
lab_pc = labels[96] + labels[40] + labels[43]
lab_cma = labels[10] + labels[11]
lab_pfc = labels[49] + labels[46] + labels[169] + labels[179] + labels[125] + labels[47] + labels[53] + labels[51] + labels[45] + labels[176] + labels[166] + labels[23] + labels[50]
lab_smar = labels[98] + labels[102] + labels[101] + labels[100] + labels[99]
lab_inf = labels[21] + labels[22] + labels[72] + labels[71] + labels[74] + labels[73] + labels[24] + labels[175]
sum_label = lab_m1 + lab_sma + lab_pm + lab_pc + lab_cma + lab_pfc + lab_smar + lab_inf

lab_ROI = stc.in_label(sum_label)
lab_idx = lab_ROI.vertices[1] + len(src[0]['vertno'])  # RH indices in the mixed surface source space
adjacency_lab = coo_matrix(adjacency.tocsr()[np.ix_(lab_idx, lab_idx)])


# Load morphed correlations: subjects × frequency × source × time
X = np.array([np.load(PATHto + subj + '_corr_superlet_4-30Hz_-025to01_vs_inhampl_earlylateinh_2hemi_morphed.npy') for subj in subjects])
times = np.arange(-0.25, 0.1001, 1 / 600)
t1, t2 = np.argmin(np.abs(times + 0.15)), np.argmin(np.abs(times - 0)) + 1
X = X[:, :, :, t1:t2]
X = np.transpose(X, [0, 3, 1, 2])[:, :, :, lab_idx]  # subjects × time × frequency × ROI sources


n_subjects, n_times, n_freqs = X.shape[:3]
adjacency = combine_adjacency(n_times, n_freqs, adjacency_lab)
t_threshold = stats.t.ppf(1 - 0.001, df=n_subjects - 1)

T_obs, clusters, p_values, H0 = spatio_temporal_cluster_1samp_test(X, threshold=t_threshold, adjacency=adjacency, stat_fun=partial(ttest_1samp_no_p, sigma=1e-3), tail=0, n_permutations=5000, buffer_size=None)

np.save(PATHto + 'corr_4-30Hz_-015to0_aroundinh_earlylateinh_vs_inh_pampl_rh_motor_noTFCE_p0001_tail0.npy', T_obs)