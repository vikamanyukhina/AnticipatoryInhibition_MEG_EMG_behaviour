#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Beta-burst detection and burst-related activity analyses.

The script performs Superlet-based time-frequency decomposition in the SMA,
extracts beta bursts, classifies trials by burst frequency, and compares
high-gamma and periodic power around burst peaks.

"""

import pickle

import jax.numpy as jnp
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from fooof import FOOOF
from matplotlib.colors import LinearSegmentedColormap
from mne.stats import combine_adjacency, spatio_temporal_cluster_1samp_test
from superlets import adaptive_superlet_transform

from burst_detection_newoverlap_noerp import extract_bursts


# Subject lists and data paths are kept as in the original analysis.
SUBJECTS = (
    ['BM_C0' + str(ii) for ii in np.arange(3, 10)]
    + ['BM_C' + str(ii) for ii in np.arange(10, 13)]
    + ['BM_C' + str(ii) for ii in np.arange(14, 28)]
)

SUBJECTS_earlylate = np.array([
    'BM_C03', 'BM_C04', 'BM_C05', 'BM_C07', 'BM_C08', 'BM_C09',
    'BM_C10', 'BM_C12', 'BM_C14', 'BM_C15', 'BM_C16', 'BM_C17',
    'BM_C18', 'BM_C20', 'BM_C22', 'BM_C23', 'BM_C24', 'BM_C25',
    'BM_C26'
])

PATHto = 'pathto'
sfreq = 600

# Superlet parameters used for beta-burst detection.
params = [3, 20]
freqs = np.arange(2, 50.25, 0.25)

# Parula-like colormap used for the group-level time-frequency plots.
cm_data = [
    [0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905],
    [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143],
    [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 0.779247619],
    [0.1252714286, 0.3242428571, 0.8302714286], [0.0591333333, 0.3598333333, 0.8683333333],
    [0.0116952381, 0.3875095238, 0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571],
    [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 0.8719571429],
    [0.0498142857, 0.4585714286, 0.8640571429], [0.0629333333, 0.4736904762, 0.8554380952],
    [0.0722666667, 0.4886666667, 0.8467], [0.0779428571, 0.5039857143, 0.8383714286],
    [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 0.8262714286],
    [0.0640571429, 0.5569857143, 0.8239571429], [0.0487714286, 0.5772238095, 0.8228285714],
    [0.0343428571, 0.5965809524, 0.819852381], [0.0265, 0.6137, 0.8135],
    [0.0238904762, 0.6286619048, 0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667],
    [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 0.7607190476],
    [0.0383714286, 0.6742714286, 0.743552381], [0.0589714286, 0.6837571429, 0.7253857143],
    [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429],
    [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 0.6424333333],
    [0.2178285714, 0.7250428571, 0.6192619048], [0.2586428571, 0.7317142857, 0.5954285714],
    [0.3021714286, 0.7376047619, 0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667],
    [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 0.5033142857],
    [0.4871238095, 0.7490619048, 0.4839761905], [0.5300285714, 0.7491142857, 0.4661142857],
    [0.5708571429, 0.7485190476, 0.4493904762], [0.609852381, 0.7473142857, 0.4336857143],
    [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333],
    [0.7184095238, 0.7411333333, 0.3904761905], [0.7524857143, 0.7384, 0.3768142857],
    [0.7858428571, 0.7355666667, 0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762],
    [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217],
    [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 0.2886428571],
    [0.9738952381, 0.7313952381, 0.266647619], [0.9937714286, 0.7454571429, 0.240347619],
    [0.9990428571, 0.7653142857, 0.2164142857], [0.9955333333, 0.7860571429, 0.196652381],
    [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857],
    [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309],
    [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 0.0948380952],
    [0.9661, 0.9514428571, 0.0755333333], [0.9763, 0.9831, 0.0538]
]
parula_map = LinearSegmentedColormap.from_list('parula', cm_data)


# -----------------------------------------------------------------------------
# 1. Superlet decomposition and beta-burst detection
# -----------------------------------------------------------------------------

for subj in SUBJECTS:
    print('Detecting bursts:', subj)

    # Keep only epochs retained after head-position and EMG-inhibition selection.
    n_epo = np.load(PATHto + subj + '_epo_n_inh_headpos_auto.npy')
    inh_times = np.load(
        PATHto + subj + '_time_peakinh_unloading_and_inh_ontime_auto15_20ms_headpos1cm.npy'
    )

    # Source time courses: trials x SMA vertices x time.
    data = np.load(
        PATHto + subj + '_all_lcmv_maxbetavert_deep_allepo_indepo_reg005_for_alphabeta_nof_-17-12_epoxvertxtimes_goodheadpos_morphed_4SMAvert.npy'
    )[:, n_epo, :]
    data = np.transpose(data, [1, 0, 2])

    power_subject = []
    periodic_power_subject = []
    bursts_subject = []

    for vertex in range(data.shape[1]):
        power_trials = []
        time_courses = []

        for trial in range(data.shape[0]):
            # Compute Superlet power for each trial and SMA vertex.
            scalogram = adaptive_superlet_transform(
                data[trial, vertex, :], freqs, sampling_freq=sfreq,
                base_cycle=params[0], min_order=1, max_order=params[1], mode='add'
            )
            power = jnp.abs(scalogram) ** 2

            # Retain a 2-s window centred on the individual EMG-inhibition peak.
            times = np.arange(-1.7, 1.2001, 1 / sfreq)
            t1 = np.argmin(np.abs(times - (inh_times[trial] - 1)))
            t2 = np.argmin(np.abs(times - (inh_times[trial] + 1)))
            power_trials.append(power[:, t1:t2])
            time_courses.append(data[trial, vertex, t1:t2])

        # Estimate and remove the aperiodic component before burst detection.
        fooof = FOOOF()
        fooof.fit(freqs, np.array(power_trials).mean(0).mean(1), [2, 50])
        aperiodic_power = 10 ** fooof._ap_fit
        periodic_power_trials = (
            np.array(power_trials) - aperiodic_power[np.newaxis, :, np.newaxis]
        )

        power_subject.append(power_trials)
        periodic_power_subject.append(periodic_power_trials)

        # Burst detection is performed in the 10–33 Hz range; the detector
        # applies the 13–30 Hz analysis range and the specified window size.
        beta_idx = np.where((freqs >= 10) & (freqs <= 33))[0]
        bursts = extract_bursts(
            np.array(time_courses),
            np.array(power_trials)[:, beta_idx, :],
            np.arange(-1, 1, 1 / sfreq),
            freqs[beta_idx],
            [13, 30],
            aperiodic_power[beta_idx].reshape(-1, 1),
            sfreq,
            w_size=0.26
        )
        bursts_subject.append(bursts)

    # Save burst dictionaries and total/periodic Superlet power for each subject.
    powers = np.array(power_subject)
    periodic_powers = np.array(periodic_power_subject)

    with open(
        PATHto + subj + '_beta_bursts_4vertSMA_epo_sup' + str(params[0]) + '-' + str(params[1])
        + '_allepo_1s_newoverlap_lcmv_nof_vector_noerp.pkl', 'wb') as f:
        pickle.dump(bursts_subject, f)

    np.save(PATHto + subj + '_total_superlet3-20_2-45_-05to05_aroundinh_goodheadpos_morphed_4SMAmed.npy', powers)
    np.save(PATHto + subj + '_periodic_superlet3-20_2-45_-05to05_aroundinh_goodheadpos_morphed_4SMAmed.npy', periodic_powers)

# -----------------------------------------------------------------------------
# 2. Classify trials by beta-burst frequency and save burst information
# -----------------------------------------------------------------------------

sup_type = '3-20'

for subj in SUBJECTS_earlylate:
    print('Classifying burst trials:', subj)

    n_epo = np.load(PATHto + subj + '_epo_n_inh_headpos_auto.npy')
    epos = np.arange(len(n_epo))
    inh_times = np.load(
        PATHto + subj + '_time_peakinh_unloading_and_inh_ontime_auto15_20ms_headpos1cm.npy'
    )

    # Restrict the main classification to trials with inhibition peaks in this interval.
    n_early = np.argwhere((inh_times >= -0.07) & (inh_times <= 0.075))[:, 0]

    burst_file = (
        PATHto + subj + '_beta_bursts_4vertSMA_epo_sup' + sup_type
        + '_allepo_1s_newoverlap_lcmv_nof_vector_noerp.pkl')
    
    with open(burst_file, 'rb') as f:
        burst_dict = pickle.load(f)

    # Combine detections from all SMA vertices into one table.
    burst_tables = []
    for vertex, vertex_bursts in enumerate(burst_dict):
        burst_tables.append(pd.DataFrame({
            'Trial': vertex_bursts['trial'],
            'PeakFreq': vertex_bursts['peak_freq'],
            'PeakTime': vertex_bursts['peak_time'],
            'PeakAmpl': vertex_bursts['peak_amp_base'],
            'TimeSpan': vertex_bursts['fwhm_time'],
            'FreqSpan': vertex_bursts['fwhm_freq'],
            'Vertex': vertex}))

    df_beta = pd.concat(burst_tables, ignore_index=True)
    df_beta['Subject'] = subj

    # Define the two beta-burst groups during the anticipatory interval.
    time_mask = (df_beta['PeakTime'] >= -0.08) & (df_beta['PeakTime'] <= 0)
    df_beta1 = df_beta[
        np.isin(df_beta['Trial'], n_early) & time_mask
        & (df_beta['PeakFreq'] >= 19) & (df_beta['PeakFreq'] <= 24)]
    
    df_beta2 = df_beta[
        np.isin(df_beta['Trial'], n_early) & time_mask
        & (df_beta['PeakFreq'] >= 24) & (df_beta['PeakFreq'] <= 29)]

    # Also retain detections across all eligible trials (without the early/late restriction).
    df_beta1_all = df_beta[
        time_mask & (df_beta['PeakFreq'] >= 19) & (df_beta['PeakFreq'] <= 24)]
    
    df_beta2_all = df_beta[
        time_mask & (df_beta['PeakFreq'] >= 24) & (df_beta['PeakFreq'] <= 29)]

    # Trials containing bursts in both frequency ranges.
    overlap_trials = np.intersect1d(df_beta1['Trial'].values, df_beta2['Trial'].values)

    # Identify trials without a burst in each band and without either burst type.
    noburst_epos1 = np.setdiff1d(epos, df_beta1['Trial'].values)
    noburst_epos2 = np.setdiff1d(epos, df_beta2['Trial'].values)
    noburst_epos = np.setdiff1d(
        epos, np.concatenate([df_beta1['Trial'].values, df_beta2['Trial'].values]))
    
    noburst_epos1_all = np.setdiff1d(epos, df_beta1_all['Trial'].values)
    noburst_epos2_all = np.setdiff1d(epos, df_beta2_all['Trial'].values)
    noburst_epos_all = np.setdiff1d(
        epos, np.concatenate([df_beta1_all['Trial'].values, df_beta2_all['Trial'].values]))

    # Early/late trial subsets are used in the subsequent burst-related analyses.
    noburst_epos1_earlylate = np.intersect1d(noburst_epos1, n_early)
    noburst_epos2_earlylate = np.intersect1d(noburst_epos2, n_early)
    noburst_epos_earlylate = np.intersect1d(noburst_epos, n_early)

    # Save burst characteristics for the 19–24 Hz group.
    for suffix, df_band, no_burst, no_burst_earlylate in [
        ('19-24', df_beta1, noburst_epos1, noburst_epos1_earlylate),
        ('24-29', df_beta2, noburst_epos2, noburst_epos2_earlylate)]:
        
        np.save(PATHto + subj + '_betaburst' + suffix + '_ampls_new.npy', df_band['PeakAmpl'].values)
        np.save(PATHto + subj + '_betaburst' + suffix + '_times_new.npy', df_band['PeakTime'].values)
        np.save(PATHto + subj + '_betaburst' + suffix + '_timespan_new.npy', df_band['TimeSpan'].values)
        np.save(PATHto + subj + '_betaburst' + suffix + '_freqspan_new.npy', df_band['FreqSpan'].values)
        np.save(PATHto + subj + '_betaburst' + suffix + '_trials_new.npy', df_band['Trial'].values)
        np.save(PATHto + subj + '_betaburst' + suffix + '_verts_new.npy', df_band['Vertex'].values)
        np.save(PATHto + subj + '_nobetaburst' + suffix + '_all_trials_new.npy', no_burst)
        np.save(PATHto + subj + '_nobetaburst' + suffix + '_earlylate_trials_new.npy', no_burst_earlylate)

    # Save overlap and combined no-burst trial indices.
    np.save(PATHto + subj + '_betaburst19-24-29_ovl_trials_new.npy', overlap_trials)
    np.save(PATHto + subj + '_nobetaburst19-24-29_all_trials_new.npy', noburst_epos)
    np.save(PATHto + subj + '_nobetaburst19-24-29_earlylate_trials_new.npy', noburst_epos_earlylate)
    np.save(PATHto + subj + '_nobetaburst19-29_all_trials_new.npy', noburst_epos)
    np.save(PATHto + subj + '_nobetaburst24-29_earlylate_trials_new.npy', noburst_epos_earlylate)

    # Save burst detections across all trials, independently of the early/late subset.
    for suffix, df_band, no_burst in [
        ('19-24', df_beta1_all, noburst_epos1_all),
        ('24-29', df_beta2_all, noburst_epos2_all)
    ]:
        np.save(PATHto + subj + '_betaburst' + suffix + '_all_times_new.npy', df_band['PeakTime'].values)
        np.save(PATHto + subj + '_betaburst' + suffix + '_all_trials_new.npy', df_band['Trial'].values)
        np.save(PATHto + subj + '_betaburst' + suffix + '_all_verts_new.npy', df_band['Vertex'].values)
        np.save(PATHto + subj + '_nobetaburst' + suffix + '_fromall_trials_new.npy', no_burst)

    np.save(PATHto + subj + '_nobetaburst19-24_fromall_trials_new.npy', noburst_epos_all)


# -----------------------------------------------------------------------------
# 3. Burst-centred high-gamma and periodic-power contrasts
# -----------------------------------------------------------------------------

# Each burst is represented by a 300-ms window centred on its detected peak.
# The no-burst trials provide the reference condition used in the contrasts.
ddds1, ddds2, ddds3 = [], [], []
ddds_pow1, ddds_pow2, ddds_pow3 = [], [], []

for subj in SUBJECTS_earlylate:
    print('Calculating burst-related contrasts:', subj)

    # High-gamma power: trials x SMA vertices x frequencies x time.
    gamma_power = np.log10(np.load(PATHto + subj + '_multitaper_power_60-130Hz_-05to05_aroundinh_epo_in4SMA.npy'))

    # Periodic Superlet power: SMA vertices x trials x frequencies x time.
    periodic_power = np.load(PATHto + subj + '_periodic_superlet3-20_2-45_-05to05_aroundinh_goodheadpos_morphed_4SMAmed.npy')
    
    times_full = np.arange(-1, 1, 1 / sfreq)
    t1 = np.argmin(np.abs(times_full + 0.5))
    t2 = np.argmin(np.abs(times_full - 0.5))
    periodic_power = periodic_power[:, :, :, t1:t2]
    times = np.arange(-0.5, 0.5, 1 / sfreq)

    # Load burst events and corresponding trial/vertex indices for both bands.
    burst_data_by_band = {}
    for band in ('19-24', '24-29'):
        burst_data_by_band[band] = {
            'times': np.load(PATHto + subj + '_betaburst' + band + '_times_new.npy'),
            'trials': np.load(PATHto + subj + '_betaburst' + band + '_trials_new.npy'),
            'vertices': np.load(PATHto + subj + '_betaburst' + band + '_verts_new.npy')}

    burst_times1 = burst_data_by_band['19-24']['times']
    burst_trials1 = burst_data_by_band['19-24']['trials']
    burst_verts1 = burst_data_by_band['19-24']['vertices']
    burst_times2 = burst_data_by_band['24-29']['times']
    burst_trials2 = burst_data_by_band['24-29']['trials']
    burst_verts2 = burst_data_by_band['24-29']['vertices']

    # Find events from both frequency ranges occurring in the same trial.
    overlap_trials_unique = np.intersect1d(burst_trials1, burst_trials2)
    idx1_overlap = np.isin(burst_trials1, overlap_trials_unique)
    idx2_overlap = np.isin(burst_trials2, overlap_trials_unique)

    overlap_trials = burst_trials1[idx1_overlap].tolist() + burst_trials2[idx2_overlap].tolist()
    overlap_times = burst_times1[idx1_overlap].tolist() + burst_times2[idx2_overlap].tolist()
    overlap_verts = burst_verts1[idx1_overlap].tolist() + burst_verts2[idx2_overlap].tolist()

    # Exclude overlapping trials from the band-specific burst conditions.
    idx1 = ~np.isin(burst_trials1, overlap_trials)
    burst_times1 = burst_times1[idx1]
    burst_trials1 = burst_trials1[idx1]
    burst_verts1 = burst_verts1[idx1]
    
    # Extract high-gamma and periodic power in windows centred on each burst peak.
    burst_gamma1, burst_gamma2, overlap_gamma, no_burst_gamma = [], [], [], []
    burst_periodic1, burst_periodic2, overlap_periodic, no_burst_periodic = [], [], [], []

    for event, trial in enumerate(burst_trials1):
        t1 = np.argmin(np.abs(times - (burst_times1[event] - 0.15)))
        t2 = np.argmin(np.abs(times - (burst_times1[event] + 0.15)))
        burst_gamma1.append(gamma_power[trial, burst_verts1[event], :, t1:t2])
        burst_periodic1.append(periodic_power[burst_verts1[event], trial, :, t1:t2])

    for event, trial in enumerate(burst_trials2):
        t1 = np.argmin(np.abs(times - (burst_times2[event] - 0.15)))
        t2 = np.argmin(np.abs(times - (burst_times2[event] + 0.15)))
        burst_gamma2.append(gamma_power[trial, burst_verts2[event], :, t1:t2])
        burst_periodic2.append(periodic_power[burst_verts2[event], trial, :, t1:t2])

    for event, trial in enumerate(overlap_trials):
        t1 = np.argmin(np.abs(times - (overlap_times[event] - 0.15)))
        t2 = np.argmin(np.abs(times - (overlap_times[event] + 0.15)))
        overlap_gamma.append(gamma_power[trial, overlap_verts[event], :, t1:t2])
        overlap_periodic.append(periodic_power[overlap_verts[event], trial, :, t1:t2])

    no_burst_trials = np.load(PATHto + subj + '_nobetaburst19-29_earlylate_trials_new.npy')
    
    for event, trial in enumerate(no_burst_trials):
        t1 = np.argmin(np.abs(times - (burst_times2[event] - 0.15)))
        t2 = np.argmin(np.abs(times - (burst_times2[event] + 0.15)))
        no_burst_gamma.append(gamma_power[trial, :, :, t1:t2].mean(0))
        no_burst_periodic.append(periodic_power[:, trial, :, t1:t2].mean(0))

    burst_gamma1 = np.asarray(burst_gamma1)
    burst_gamma2 = np.asarray(burst_gamma2)
    overlap_gamma = np.asarray(overlap_gamma)
    no_burst_gamma = np.asarray(no_burst_gamma)
    burst_periodic1 = np.asarray(burst_periodic1)
    burst_periodic2 = np.asarray(burst_periodic2)
    overlap_periodic = np.asarray(overlap_periodic)
    no_burst_periodic = np.asarray(no_burst_periodic)

    # Subtract the mean no-burst response from each burst condition.
    if len(burst_trials1) >= 3:
        ddds1.append((burst_gamma1 - no_burst_gamma.mean(0).mean(1)[np.newaxis, :, np.newaxis]).mean(0))
        ddds_pow1.append((burst_periodic1 - no_burst_periodic.mean(0).mean(1)[np.newaxis, :, np.newaxis]).mean(0))

    if len(burst_trials2) >= 3:
        ddds2.append((burst_gamma2 - no_burst_gamma.mean(0).mean(1)[np.newaxis, :, np.newaxis]).mean(0))
        ddds_pow2.append((burst_periodic2 - no_burst_periodic.mean(0).mean(1)[np.newaxis, :, np.newaxis]).mean(0))

    if len(overlap_trials) >= 3:
        ddds3.append((overlap_gamma - no_burst_gamma.mean(0).mean(1)[np.newaxis, :, np.newaxis]).mean(0))
        ddds_pow3.append((overlap_periodic - no_burst_periodic.mean(0).mean(1)[np.newaxis, :, np.newaxis]).mean(0))

ddds1, ddds2, ddds3 = np.asarray(ddds1), np.asarray(ddds2), np.asarray(ddds3)
ddds_pow1 = np.asarray(ddds_pow1)
ddds_pow2 = np.asarray(ddds_pow2)
ddds_pow3 = np.asarray(ddds_pow3)

# -----------------------------------------------------------------------------
# 4. Plot burst-related high-gamma and periodic-power contrasts
# -----------------------------------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Burst-related power contrasts (overlapping trials excluded from band-specific groups)')

data_list = [
    (ddds1, 'High-gamma: 19–24 Hz bursts vs no bursts (overlap excluded)', [-0.15, 0.15, 60, 130], 0.08, True),
    (ddds2, 'High-gamma: 24–29 Hz bursts vs no bursts (overlap retained)', [-0.15, 0.15, 60, 130], 0.08, True),
    (ddds3, 'High-gamma: overlapping burst trials vs no bursts', [-0.15, 0.15, 60, 130], 0.08, True),
    (ddds_pow1, 'Periodic power: 19–24 Hz bursts vs no bursts (overlap excluded)', [-0.15, 0.15, 2, 50], 1.05e-17, False),
    (ddds_pow2, 'Periodic power: 24–29 Hz bursts vs no bursts (overlap retained)', [-0.15, 0.15, 2, 50], 1.05e-17, False),
    (ddds_pow3, 'Periodic power: overlapping burst trials vs no bursts', [-0.15, 0.15, 2, 50], 1.05e-17, False)]

for ax, (data, title, extent, vmax, is_gamma) in zip(axes.ravel(), data_list):
    image = ax.imshow(
        data.mean(0), aspect='auto', origin='lower', extent=extent,
        cmap=parula_map, vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel('Time relative to burst peak (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_ylim(90, 130) if is_gamma else ax.set_ylim(3, 35)
    fig.colorbar(image, ax=ax)

plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# 5. Cluster-based permutation tests on high-gamma contrasts
# -----------------------------------------------------------------------------

def plot_gamma_cluster(data, time_window, title):
    """Run a one-sample spatiotemporal cluster test and plot significant effects."""
    gamma_freqs = np.arange(60, 131)
    f1 = np.argmin(np.abs(gamma_freqs - 90))
    f2 = np.argmin(np.abs(gamma_freqs - 130))
    times = np.arange(-0.15, 0.15, 1 / sfreq)

    # Input data: subjects x frequencies x time; transpose to subjects x time x frequencies.
    data = np.transpose(data, [0, 2, 1])
    t1 = np.argmin(np.abs(times - time_window[0]))
    t2 = np.argmin(np.abs(times - time_window[1])) + 1
    data = data[:, t1:t2, f1:f2]

    adjacency = combine_adjacency(data.shape[1], data.shape[2])
    threshold_tfce = dict(start=0, step=0.2)

    # One-sided test, 5,000 permutations
    t_obs, clusters, p_values, h0 = spatio_temporal_cluster_1samp_test(
        -data,
        n_jobs=None,
        threshold=threshold_tfce,
        adjacency=adjacency,
        buffer_size=None,
        tail=1,
        n_permutations=5000)

    mean_data = data.mean(0)
    significant = p_values.reshape(t_obs.shape).T < 0.05
    print(f'{title}: {significant.sum()} significant time-frequency points')
    masked_data = np.where(significant, mean_data.T, 0)

    info = mne.create_info(['Ch1'], sfreq, 'mag')
    tfr = mne.time_frequency.AverageTFR(
        info=info,
        data=np.expand_dims(mean_data.T, 0),
        times=times[t1:t2],
        freqs=gamma_freqs[f1:f2],
        nave=1)
    
    return tfr.plot(
        mask=masked_data != 0,
        cmap=parula_map,
        mask_style='contour',
        vlim=(-0.08, 0.08))


# 19–24 Hz burst-related high-gamma contrast: −50 to +50 ms around burst peak.
plot_gamma_cluster(ddds1, (-0.05, 0.05), '19–24 Hz bursts')

# 24–29 Hz burst-related high-gamma contrast: 0 to +150 ms around burst peak.
plot_gamma_cluster(ddds2, (0, 0.15), '24–29 Hz bursts')


          