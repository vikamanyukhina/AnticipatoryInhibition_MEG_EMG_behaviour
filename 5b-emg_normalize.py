#!/usr/bin/env python3

import mne
import numpy as np

# Paths and subjects
PATHfrom = 'pathfrom'
PATHto = 'pathto'
exp = 'naturel'
SUBJECTS = [f'BM_C{ii:02d}' for ii in range(1, 25)]

# Processing parameters
CHANNELS = ['BicG', 'TriG', 'SupG', 'UADC002-2800', 'UADC001-2800']
LOW_FREQ = 25
HIGH_FREQ = 150
TMIN = -1.7
TMAX = 1.2


# =========================================================================
# Log10-normalized Biceps EMG
# Baseline: -1.5 to -0.5 s
# Analysis window: -1 to 1 s
# =========================================================================

for subj in SUBJECTS:

    raw = mne.io.Raw(PATHfrom + subj + '_naturel_indpos_ica-ref_thr2_raw.fif', preload=True)
    raw = raw.copy().notch_filter([50, 100, 150])

    raw_5ch = raw.pick_channels(CHANNELS)
    raw_data = raw_5ch.get_data()
    raw_data = [raw_data[i] - np.mean(raw_data[i]) if 'UAD' not in raw_5ch.ch_names[i] else raw_data[i] for i in range(5)]
    raw_5ch = mne.io.RawArray(np.array(raw_data), raw_5ch.info)

    raw_5ch = raw_5ch.filter(LOW_FREQ, HIGH_FREQ)
    raw_data = raw_5ch.get_data()
    raw_data = [np.abs(raw_data[i]) if 'UAD' not in raw_5ch.ch_names[i] else raw_data[i] for i in range(5)]
    raw_5ch = mne.io.RawArray(np.array(raw_data), raw_5ch.info)

    # 'naturel_action_start': automatically identified unloading onset
    events = mne.events_from_annotations(raw, event_id={'naturel_action_start': 1})[0]

    epochs = mne.Epochs(raw_5ch, events, tmin=TMIN, tmax=TMAX, baseline=None,
                        proj=False, preload=True, reject_by_annotation=False)

    times = epochs.times
    b1 = np.argmin(np.abs(times + 1.5))
    b2 = np.argmin(np.abs(times + 0.5)) + 1
    t1 = np.argmin(np.abs(times + 1))
    t2 = np.argmin(np.abs(times - 1)) + 1

    emg = epochs.copy().pick_channels(['BicG']).get_data()[:, 0, :]
    emg_log_norm = np.log10(emg[:, t1:t2]) - np.log10(emg[:, b1:b2]).mean(axis=1)[:, None]

    np.save(PATHto + subj + '_inh_index_times_all_trials_log10_norm_BicL.npy', emg_log_norm)


# =========================================================================
# Unlogged, unnormalized Biceps EMG
# Window: -1.5 to 1 s
# =========================================================================

for subj in SUBJECTS:

    raw = mne.io.Raw(PATHfrom + subj + '_naturel_indpos_ica-ref_thr2_raw.fif', preload=True)
    raw = raw.copy().notch_filter([50, 100, 150])

    raw_5ch = raw.pick_channels(CHANNELS)
    raw_data = raw_5ch.get_data()
    raw_data = [raw_data[i] - np.mean(raw_data[i]) if 'UAD' not in raw_5ch.ch_names[i] else raw_data[i] for i in range(5)]
    raw_5ch = mne.io.RawArray(np.array(raw_data), raw_5ch.info)

    raw_5ch = raw_5ch.filter(LOW_FREQ, HIGH_FREQ)
    raw_data = raw_5ch.get_data()
    raw_data = [np.abs(raw_data[i]) if 'UAD' not in raw_5ch.ch_names[i] else raw_data[i] for i in range(5)]
    raw_5ch = mne.io.RawArray(np.array(raw_data), raw_5ch.info)

    # 'naturel_action_start': automatically identified unloading onset
    events = mne.events_from_annotations(raw, event_id={'naturel_action_start': 1})[0]

    epochs = mne.Epochs(raw_5ch, events, tmin=TMIN, tmax=TMAX, baseline=None,
                        proj=False, preload=True, reject_by_annotation=False)

    times = epochs.times
    t1 = np.argmin(np.abs(times + 1.5))
    t2 = np.argmin(np.abs(times - 1)) + 1

    emg = epochs.copy().pick_channels(['BicG']).get_data()[:, 0, t1:t2]

    np.save(PATHto + subj + '_inh_index_-15to1_times_all_trials_unlog_unnorm_BicL.npy', emg)