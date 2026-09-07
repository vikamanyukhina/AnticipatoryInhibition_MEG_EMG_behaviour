#!/usr/bin/env python3

import mne
import matplotlib.pyplot as plt
import numpy as np
from mne.report import Report
from mne.time_frequency import tfr_array_multitaper
from scipy.signal import find_peaks
from matplotlib.colors import LinearSegmentedColormap


# ============================================================
# Custom colormap
# ============================================================

cm_data = [
    [0.2081, 0.1663, 0.5292],
    [0.2116238095, 0.1897809524, 0.5776761905],
    [0.212252381, 0.2137714286, 0.6269714286],
    [0.2081, 0.2386, 0.6770857143],
    [0.1959047619, 0.2644571429, 0.7279],
    [0.1707285714, 0.2919380952, 0.779247619],
    [0.1252714286, 0.3242428571, 0.8302714286],
    [0.0591333333, 0.3598333333, 0.8683333333],
    [0.0116952381, 0.3875095238, 0.8819571429],
    [0.0059571429, 0.4086142857, 0.8828428571],
    [0.0165142857, 0.4266, 0.8786333333],
    [0.032852381, 0.4430428571, 0.8719571429],
    [0.0498142857, 0.4585714286, 0.8640571429],
    [0.0629333333, 0.4736904762, 0.8554380952],
    [0.0722666667, 0.4886666667, 0.8467],
    [0.0779428571, 0.5039857143, 0.8383714286],
    [0.079347619, 0.5200238095, 0.8311809524],
    [0.0749428571, 0.5375428571, 0.8262714286],
    [0.0640571429, 0.5569857143, 0.8239571429],
    [0.0487714286, 0.5772238095, 0.8228285714],
    [0.0343428571, 0.5965809524, 0.819852381],
    [0.0265, 0.6137, 0.8135],
    [0.0238904762, 0.6286619048, 0.8037619048],
    [0.0230904762, 0.6417857143, 0.7912666667],
    [0.0227714286, 0.6534857143, 0.7767571429],
    [0.0266619048, 0.6641952381, 0.7607190476],
    [0.0383714286, 0.6742714286, 0.743552381],
    [0.0589714286, 0.6837571429, 0.7253857143],
    [0.0843, 0.6928333333, 0.7061666667],
    [0.1132952381, 0.7015, 0.6858571429],
    [0.1452714286, 0.7097571429, 0.6646285714],
    [0.1801333333, 0.7176571429, 0.6424333333],
    [0.2178285714, 0.7250428571, 0.6192619048],
    [0.2586428571, 0.7317142857, 0.5954285714],
    [0.3021714286, 0.7376047619, 0.5711857143],
    [0.3481666667, 0.7424333333, 0.5472666667],
    [0.3952571429, 0.7459, 0.5244428571],
    [0.4420095238, 0.7480809524, 0.5033142857],
    [0.4871238095, 0.7490619048, 0.4839761905],
    [0.5300285714, 0.7491142857, 0.4661142857],
    [0.5708571429, 0.7485190476, 0.4493904762],
    [0.609852381, 0.7473142857, 0.4336857143],
    [0.6473, 0.7456, 0.4188],
    [0.6834190476, 0.7434761905, 0.4044333333],
    [0.7184095238, 0.7411333333, 0.3904761905],
    [0.7524857143, 0.7384, 0.3768142857],
    [0.7858428571, 0.7355666667, 0.3632714286],
    [0.8185047619, 0.7327333333, 0.3497904762],
    [0.8506571429, 0.7299, 0.3360285714],
    [0.8824333333, 0.7274333333, 0.3217],
    [0.9139333333, 0.7257857143, 0.3062761905],
    [0.9449571429, 0.7261142857, 0.2886428571],
    [0.9738952381, 0.7313952381, 0.266647619],
    [0.9937714286, 0.7454571429, 0.240347619],
    [0.9990428571, 0.7653142857, 0.2164142857],
    [0.9955333333, 0.7860571429, 0.196652381],
    [0.988, 0.8066, 0.1793666667],
    [0.9788571429, 0.8271428571, 0.1633142857],
    [0.9697, 0.8481380952, 0.147452381],
    [0.9625857143, 0.8705142857, 0.1309],
    [0.9588714286, 0.8949, 0.1132428571],
    [0.9598238095, 0.9218333333, 0.0948380952],
    [0.9661, 0.9514428571, 0.0755333333],
    [0.9763, 0.9831, 0.0538]
]

parula_map = LinearSegmentedColormap.from_list('parula', cm_data)


# ============================================================
# Paths and experiment
# ============================================================

PATHto = 'pathto'
exp = 'naturel'


# ============================================================
# Detect inhibition in children
# ============================================================

SUBJECTS = (
    ['BM_C0' + str(ii) for ii in np.arange(1, 10)]
    + ['BM_C' + str(ii) for ii in np.arange(10, 25)]
)

n_was_subj = []
n_inh_subj = []
n_clean_subj = []
events_difs = []
n_epo_inh_subj = []
betavolcor = []


for subj in SUBJECTS:

    pathto = 'pathto'

    raw_fname = pathto + subj + '_' + exp + '_indpos_ica-ref_thr2_raw.fif'
    raw = mne.io.Raw(raw_fname, preload=True)
    sfreq = raw.info['sfreq']

    events_true = mne.events_from_annotations(
        raw,
        event_id={'naturel_action_start': 1}
    )[0]

    n_was = events_true.shape[0]
    n_was_subj.append(n_was)

    report_fname = (
        PATHto + subj +
        '_annots_inhibitions_overview_multitaper_15std_nobase.html'
    )
    report = Report()

    # Epochs used to determine clean trials
    allepochs_nat_true = mne.Epochs(
        raw,
        events_true,
        tmin=-1.7,
        tmax=1.2,
        baseline=(-1.6, -1),
        proj=False,
        preload=True,
        reject_by_annotation=True
    )

    n_clean_subj.append(len(allepochs_nat_true))

    # Epochs including annotated segments
    allepochs_nat_true = mne.Epochs(
        raw,
        events_true,
        tmin=-1.7,
        tmax=1.2,
        baseline=(-1.6, -1),
        proj=False,
        preload=True,
        reject_by_annotation=False
    )

    # Biceps left channel
    ch_idx = np.argwhere(
        np.array(allepochs_nat_true.ch_names) == 'BicG'
    )[0][0]

    ddd = np.load(
        PATHto + subj +
        '_inh_index_-15to1_times_all_trials_unlog_unnorm_BicL.npy'
    )

    times_inh = np.arange(-1.5, 1.0001, 1 / 600)

    t1 = np.argmin(np.abs(times_inh + 0.15)) + 1
    t2 = np.argmin(np.abs(times_inh - 0.15)) + 1
    b1 = np.argmin(np.abs(times_inh + 1.5)) + 1
    b2 = np.argmin(np.abs(times_inh + 0.5)) + 1

    freqs = np.arange(30, 151)

    n_epo_inh = 0
    epo_inh = []
    char_inh = []
    sel_inh = []
    sel_inh_ontime = []

    # --------------------------------------------------------
    # Time-frequency analysis for each trial
    # --------------------------------------------------------

    for e in range(len(ddd)):

        tfr = tfr_array_multitaper(
            np.expand_dims(
                np.expand_dims(
                    np.pad(ddd[e], 600, mode='symmetric'),
                    0
                ),
                0
            ),
            sfreq,
            freqs,
            n_cycles=7,
            time_bandwidth=10,
            output='power'
        )

        tfr = tfr[0, 0, :, :][:, 600:-600]
        tfr = np.log10(tfr)

        # Baseline correction
        tfr = tfr[:, t1:t2] - np.expand_dims(
            np.mean(tfr[:, b1:b2], axis=1),
            axis=1
        )

        fig = plt.figure()

        plt.imshow(
            tfr,
            aspect='auto',
            origin='lower',
            extent=[-0.15, 0.15, 30, 150],
            cmap=parula_map,
            alpha=1
        )

        plt.title('Power BicepsL, epo ' + str(e))
        plt.xlabel('Time')
        plt.ylabel('Frequency')

        report.add_figure(fig, title='Epo ' + str(e))
        plt.close('all')

        # Identify time-frequency points below the inhibition threshold
        thr = (
            np.mean(tfr[:, b1:b2])
            - 1.5 * np.std(tfr[:, b1:b2])
        )

        mask = tfr < thr
        tfr = np.where(mask, tfr, 0)

        fig = plt.figure()

        plt.imshow(
            tfr,
            aspect='auto',
            origin='lower',
            extent=[-0.15, 0.15, 30, 150],
            cmap=parula_map,
            alpha=1
        )

        plt.title('Power BicepsL <1.5*SD, epo ' + str(e))
        plt.xlabel('Time')
        plt.ylabel('Frequency')

        report.add_figure(
            fig,
            title='Epo ' + str(e) + ' 1.5std'
        )

        plt.close('all')

        # Require a minimum temporal extent of 50 ms
        inh_freq = []
        inh_time = []
        times = times_inh[t1:t2]
        mask_sign = np.zeros(mask.copy().shape)

        for f in range(tfr.shape[0]):

            if sum(mask[f, :]) > 0:

                sel = np.argwhere(mask[f, :] == 1)[:, 0]

                for s in sel:

                    if np.sum(mask[f, :][s - 30:s + 30]) >= 12:
                        inh_time.append(s)
                        inh_freq.append(f)
                        epo_inh.append(e)

        if len(inh_time) != 0:
            n_epo_inh += 1

        for j in range(len(inh_time)):
            mask_sign[
                inh_freq[j],
                inh_time[j]
            ] = tfr[
                inh_freq[j],
                inh_time[j]
            ]

        fig = plt.figure()

        plt.imshow(
            mask_sign,
            aspect='auto',
            origin='lower',
            extent=[-0.15, 0.15, 30, 150],
            cmap=parula_map,
            alpha=1
        )

        plt.xlabel('Time')
        plt.ylabel('Frequency')

        report.add_figure(
            fig,
            title='Epo ' + str(e) + ' selected Inh+cluster'
        )

        plt.close('all')

        # Detect peaks in the frequency-averaged inhibition signal
        freqav = mask_sign.mean(0) * (-1)

        peaks, properties = find_peaks(
            freqav,
            distance=30,
            prominence=0.1,
            width=3
        )

        if len(peaks) != 0:

            sel_inh.append(
                allepochs_nat_true.selection[e]
            )

            fig = plt.figure()
            plt.plot(times, freqav)

            char_inh_peaks = []

            for ppp_i, ppp in enumerate(peaks):

                plt.axvline(
                    times[ppp],
                    color='red',
                    linestyle='--'
                )

                if (
                    times[properties['left_bases'][ppp_i]] > -0.1
                    and
                    times[properties['left_bases'][ppp_i]] < 0.125
                ):

                    left_i = properties['left_bases'][ppp_i]
                    right_i = properties['right_bases'][ppp_i]

                    plt.axvline(
                        times[left_i],
                        color='red'
                    )

                    plt.axvline(
                        times[right_i],
                        color='red'
                    )

                    wampl = freqav[
                        freqav > (2 / 3) *
                        np.max(freqav[left_i:right_i])
                    ].mean()

                    char_inh_peaks.append([
                        e,
                        times[ppp],
                        times[properties['left_bases'][ppp_i]],
                        (1 / 600) * (
                            times[properties['right_bases'][ppp_i]]
                            - times[properties['left_bases'][ppp_i]]
                        ),
                        (1 / 600) * properties['widths'][ppp_i],
                        properties['prominences'][ppp_i],
                        wampl,
                        np.sum(
                            freqav[
                                properties['left_bases'][ppp_i]:
                                properties['right_bases'][ppp_i] + 1
                            ]
                        )
                    ])

                    char_inh.append(char_inh_peaks)
                    sel_inh_ontime.append(
                        allepochs_nat_true.selection[e]
                    )

            report.add_figure(
                fig,
                title='Epo ' + str(e) + ' freq-av inh'
            )

            plt.close('all')

        else:
            char_inh.append([e])

    # ========================================================
    # Save inhibition detection results
    # ========================================================

    np.save(
        PATHto + subj +
        '_epo_selection_unloading_and_inh_auto15_20ms.npy',
        np.unique(sel_inh)
    )

    np.save(
        PATHto + subj +
        '_epo_selection_unloading_and_inh_ontime_auto15_20ms.npy',
        np.unique(sel_inh_ontime)
    )

    n_inh_subj.append(len(np.unique(sel_inh)))

    np.save(
        PATHto + subj +
        '_epo_inh_ontime_inhpeaktime_inhonset_duration_width_ampl_wampl_sum_20ms_15std_unload.npy',
        np.array(char_inh, dtype=object)
    )

    np.save(
        PATHto + subj +
        '_epo_inh_ontime_inh_20ms_15std_unload.npy',
        epo_inh
    )

    np.save(
        PATHto + subj +
        '_inh_inh_ontime_time_20ms_15std_unload.npy',
        inh_time
    )

    np.save(
        PATHto + subj +
        '_inh_inh_ontime_freq_20ms_15std_unload.npy',
        inh_freq
    )

    report.save(
        report_fname,
        open_browser=False,
        overwrite=True
    )


# ========================================================
# Load and save in a more appropriate format 
# ========================================================


SUBJECTS = (
    ['BM_C0' + str(ii) for ii in np.arange(1, 10)]
    + ['BM_C' + str(ii) for ii in np.arange(10, 25)]
)

biceps_epos = []
was = []
new = []
equal = []


for subj in SUBJECTS:

    pathto = 'pathto'

    raw_fname = pathto + subj + '_' + exp + '_indpos_ica-ref_thr2_raw.fif'
    raw = mne.io.Raw(raw_fname, preload=True)
    sfreq = raw.info['sfreq']

    # Remove line noise
    raw = raw.copy().notch_filter([50, 100, 150])

    # Select relevant EMG and analog channels
    raw_6ch = raw.pick_channels([
        'BicG',
        'TriG',
        'SupG',
        'UADC002-2800',
        'UADC001-2800'
    ])

    raw_data = raw_6ch.get_data()

    # Demean EMG channels; keep analog channels unchanged
    raw_data_demean = []

    for i in range(5):
        if 'UAD' not in raw_6ch.ch_names[i]:
            raw_data_demean.append(
                raw_data[i] - np.mean(raw_data[i])
            )
        else:
            raw_data_demean.append(raw_data[i])

    raw_data_demean_arr = np.array(raw_data_demean)
    raw_6ch = mne.io.RawArray(
        raw_data_demean_arr,
        raw_6ch.info
    )

    # Band-pass filter
    high = 150
    low = 25
    raw_6ch = raw_6ch.filter(low, high)

    # Rectify EMG channels; keep analog channels unchanged
    raw_data = raw_6ch.get_data()
    raw_data_abs = []

    for i in range(5):
        if 'UAD' not in raw_6ch.ch_names[i]:
            raw_data_abs.append(np.abs(raw_data[i]))
        else:
            raw_data_abs.append(raw_data[i])

    raw_data_abs_arr = np.array(raw_data_abs)

    raw_6ch = mne.io.RawArray(
        raw_data_abs_arr,
        raw_6ch.info
    )

    # Epoch the processed EMG data
    events_true = mne.events_from_annotations(
        raw,
        event_id={'naturel_action_start': 1}
    )[0]

    allepochs_nat_true = mne.Epochs(
        raw_6ch,
        events_true,
        tmin=-1.7,
        tmax=1.2,
        baseline=(-1.6, -1),
        proj=False,
        preload=True,
        reject_by_annotation=False
    )

    data = allepochs_nat_true.copy().pick_channels(
        ['BicG']
    ).get_data()

    # Load trials identified as inhibition trials
    n_epo = np.load(
        PATHto + subj +
        '_epo_inh_ontime_inhpeaktime_inhonset_duration_width_ampl_wampl_sum_20ms_15std_unload.npy',
        allow_pickle=True
    )

    epo_idx = []
    epo_inh_time = []
    epo_inh_time_onset = []
    epo_inh_pampl = []
    epo_inh_wampl = []
    epo_inh_sum = []
    epo_inh_len = []

    # Extract inhibition characteristics
    for e in range(len(n_epo)):

        if isinstance(n_epo[e][0], int):
            pass

        else:
            # Do not include duplicate inhibition events
            if n_epo[e][0][0] in epo_idx:
                pass

            else:
                lll = n_epo[e][0]

                epo_idx.append(lll[0])
                epo_inh_time.append(lll[1])
                epo_inh_time_onset.append(lll[2])
                epo_inh_len.append(lll[3])
                epo_inh_pampl.append(lll[5])
                epo_inh_wampl.append(lll[6])
                epo_inh_sum.append(lll[7])

    # Save inhibition characteristics
    np.save(PATHto + subj + '_epo_inh_ontime_time_inhpeak_auto15_20ms_unload.npy', np.array(epo_inh_time))
    np.save(PATHto + subj + '_epo_inh_ontime_time_inhonset_auto15_20ms_unload.npy', np.array(epo_inh_time_onset))
    np.save(PATHto + subj + '_epo_inh_ontime_duration_auto15_20ms_unload.npy', np.array(epo_inh_len))
    np.save(PATHto + subj + '_epo_inh_ontime_n_epos_auto15_20ms_unload.npy', np.array(epo_idx))
    np.save(PATHto + subj + '_epo_inh_ontime_peak_ampl_auto15_20ms_unload.npy', np.array(epo_inh_pampl))
    np.save(PATHto + subj + '_epo_inh_ontime_weighted_ampl_auto15_20ms_unload.npy', np.array(epo_inh_wampl))
    np.save(PATHto + subj + '_epo_inh_ontime_sum_ampl_auto15_20ms_unload.npy', np.array(epo_inh_sum))

    if len(epo_inh_time) != len(epo_idx) != len(epo_inh_pampl) != len(epo_inh_wampl):
        raise ValueError

    w = np.load(PATHto + subj + '_epo_selection_unloading_and_inh_ontime_auto15_20ms.npy')
    was.append(len(w))
    new.append(len(np.array(epo_idx)))
    equal.append(w.tolist() == epo_idx)
