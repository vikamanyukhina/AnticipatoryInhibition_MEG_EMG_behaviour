#!/usr/bin/env python3

import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mne.report import Report
from matplotlib.colors import LinearSegmentedColormap


# =============================================================================
# Colormap
# =============================================================================

cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905],
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
           [0.5708571429, 0.7485190476, 0.4493904767], [0.609852381, 0.7473142857, 0.4336857143],
           [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333],
           [0.7184095238, 0.7411333333, 0.3904761905], [0.7524857143, 0.7384, 0.3768142857],
           [0.7858428571, 0.7355666667, 0.3632714286], [0.8185047619, 0.7327333333, 0.3497904768],
           [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217],
           [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 0.2886428571],
           [0.9738952381, 0.7313952381, 0.266647619], [0.9937714286, 0.7454571429, 0.240347619],
           [0.9990428571, 0.7653142857, 0.2164142857], [0.9955333333, 0.7860571429, 0.196652381],
           [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857],
           [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309],
           [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 0.0948380952],
           [0.9661, 0.9514428571, 0.0755333333], [0.9763, 0.9831, 0.0538]]

parula_map = LinearSegmentedColormap.from_list('parula', cm_data)


# =============================================================================
# Functions
# =============================================================================

def drop_outliers(data):
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    return np.array([i for i in data if q25 - 1.5 * iqr < i < q75 + 1.5 * iqr])


def moving_average(data, window):
    return np.array([data[i:i + window + 1].mean()
                     for i in range(len(data) - window + 1)])


# =============================================================================
# Paths and subjects
# =============================================================================

subj_type = 'Children/Control'
PATHfrom = 'pathfrom'
PATHto = 'pathto'
exp = 'naturel'

SUBJECTS = ['BM_C0' + str(ii) for ii in np.arange(1, 10)] + ['BM_C' + str(ii) for ii in np.arange(10, 25)]

med_rot = pd.read_csv(PATHfrom + 'med_elbow_rot_imposed_children.csv')


# Subjects with inverted elbow-rotation polarity
INVERT_ROTATION = [
    'BM_C01', 'BM_C02', 'BM_C03', 'BM_C04', 'BM_C05', 'BM_C06', 'BM_C07',
    'BM_C08', 'BM_C09', 'BM_C11', 'BM_C12', 'BM_C13', 'BM_C15', 'BM_C16',
    'BM_C17', 'BM_C18', 'BM_C19', 'BM_C21', 'BM_C22', 'BM_C24'
]


# =============================================================================
# Subject loop
# =============================================================================

for subj in SUBJECTS:

    med_abs = med_rot.loc[med_rot['Subjects'] == subj, 'Median elbow rot ampl'].iloc[0]

    # Event definitions:
    # type_stimulus = onset of unloading, identified automatically from the
    #                 'naturel_action_start' annotation.
    # type_behav = peak elbow rotation, identified manually and marked by the
    #              'naturel_elbow_rot' annotation.
    type_stimulus = 'naturel_action_start'
    type_behav = 'naturel_elbow_rot'

    raw_fname = PATHfrom + '/' + subj + '_' + exp + '_indpos_ica-ref_thr2_raw.fif'
    raw = mne.io.Raw(raw_fname, preload=True)
    sfreq = raw.info['sfreq']

    events = mne.events_from_annotations(raw, event_id={type_stimulus: 1})[0]
    while events[0][0] == 0:
        events = events[1:]

    events_rot = mne.events_from_annotations(raw, event_id={type_behav: 2})[0]
    while events_rot[0][0] == 0:
        events_rot = events_rot[1:]

    # Match each manually identified elbow-rotation peak to the nearest
    # automatically identified unloading onset.
    sel = []
    for e in range(len(events_rot)):
        rot_idx = np.argmin(np.abs(events[:, 0] - events_rot[e][0]))
        sel.append(rot_idx)

    np.save(PATHto + subj + '_epo_unloading_with_elbrot.npy', sel)

    # =========================================================================
    # Epoch data around unloading and manually identified elbow-rotation peak
    # =========================================================================

    tmin = -1.7
    tmax = 1.2

    allepochs_nat_true = mne.Epochs(
        raw, events, tmin=tmin, tmax=tmax, baseline=None,
        proj=False, preload=True, reject_by_annotation=False
    )[sel]

    rot_epo = mne.Epochs(
        raw.copy().pick_channels(['UADC001-2800', 'UADC002-2800']),
        events_rot, tmin=tmin, tmax=tmax, baseline=None,
        proj=False, preload=True, reject_by_annotation=False
    )

    if len(rot_epo.events) != len(allepochs_nat_true.events):
        raise ValueError

    # Elbow rotation immediately before movement and during baseline period.
    rot_epo_start_epo = allepochs_nat_true.copy().crop(
        tmin=-0.1, tmax=0
    ).pick_channels(['UADC001-2800'])

    rot_epo_base = allepochs_nat_true.copy().crop(
        tmin=-1.6, tmax=-1
    ).pick_channels(['UADC001-2800'])

    if len(rot_epo) != len(rot_epo_start_epo):
        raise ValueError

    unload_epo = rot_epo.copy().pick_channels(['UADC002-2800']).get_data()
    rot_epo = rot_epo.copy().pick_channels(['UADC001-2800'])

    unload_epo_based = allepochs_nat_true.copy().crop(
        0, 0.1
    ).pick_channels(['UADC002-2800'])

    # =========================================================================
    # Identify unloading onset in the elbow-rotation epochs
    # =========================================================================

    unload_time_idx = []
    base_rot_idx1 = []
    base_rot_idx2 = []

    times_ma = moving_average(rot_epo.times, 50)

    for i in range(len(rot_epo.events)):
        time_dif = (rot_epo.events[i][0] - allepochs_nat_true.events[i][0]) / sfreq

        if time_dif < 0:
            raise ValueError

        time_dif = -time_dif
        idx = np.argmin(np.abs(times_ma - time_dif))
        idx_b1 = np.argmin(np.abs(times_ma - (time_dif - 0.6)))
        idx_b2 = np.argmin(np.abs(times_ma - (time_dif - 0.1)))

        unload_time_idx.append(idx)
        base_rot_idx1.append(idx_b1)
        base_rot_idx2.append(idx_b2)

        print(times_ma[idx])

    # =========================================================================
    # Initialize variables
    # =========================================================================

    times01, times0 = [], []
    rot_norm_plot, rot_norm = [], []
    vels = []
    vel_peaks, vel_peak_epo = [], []
    vel_peaksel_epo, vel_maxpeak_epo = [], []
    argss, argss_base = [], []
    movestart, excs = [], []
    unloads = []
    inhrots_idx, inhrot_bases_idx = [], []
    inh_rot_norms = []
    rot_norm_auto, rot_10max_auto = [], []
    rot_stab, bad_stab, stab_yes_no = [], [], []
    rot_stab_short = []

    # =========================================================================
    # Trial loop
    # =========================================================================

    for e in range(len(rot_epo_start_epo)):

        # In some subjects, elbow rotation polarity was inverted.
        if subj in INVERT_ROTATION:
            rot = -rot_epo.get_data()
            rot_epo_start = -rot_epo_start_epo.get_data()
        else:
            rot = rot_epo.get_data()
            rot_epo_start = rot_epo_start_epo.get_data()

        bas_ma = moving_average(rot_epo_start[e, 0, :], 50)
        bas = np.mean(bas_ma)
        unload = unload_epo

        unl = moving_average(unload[e, 0, :], 50)
        epo = moving_average(rot[e, 0, :], 50)
        times_ma = moving_average(rot_epo.times, 50)
        testn = bas_ma

        time0 = np.argmin(np.abs(times_ma - 0))       # peak elbow rotation
        time05 = np.argmin(np.abs(times_ma - 0.5))
        timeb = np.argmin(np.abs(times_ma + 0.25))

        # Normalize elbow rotation using the period -0.1 to 0 s relative to
        # the automatically identified unloading onset.
        bas_true = epo[base_rot_idx1[e]:base_rot_idx2[e]].mean()
        epo_norm = epo - bas_true

        args = drop_outliers(np.argsort(epo_norm[time0 - 50:time0 + 50])[-20:])

        rot_epo_zero = np.where(
            rot[e, 0, :] == rot_epo_start[e, 0, 0]
        )[0][0]
        time01 = np.argmin(np.abs(times_ma - rot_epo.times[rot_epo_zero]))

        movestart.append(times_ma[time01])
        times01.append(time01)

        # ---------------------------------------------------------------------
        # Elbow-rotation decrease / inhibition
        # ---------------------------------------------------------------------

        inhrot = drop_outliers(
            np.argsort(epo_norm[unload_time_idx[e] - 100:time0 - 50])[:20]
        )
        inhrotmin = np.max(inhrot) + unload_time_idx[e] - 100

        inhrot_base = drop_outliers(
            np.argsort(epo_norm[unload_time_idx[e] - 100:inhrotmin])[-20:]
        )

        inhrots_idx.append(inhrot)
        inhrot_bases_idx.append(inhrot_base)

        inh_rot_norm = (
            epo_norm[unload_time_idx[e] - 100:time0][inhrot].mean()
            - epo_norm[unload_time_idx[e] - 100:inhrotmin][inhrot_base].mean()
        )
        inh_rot_norms.append(inh_rot_norm)

        # ---------------------------------------------------------------------
        # Estimate baseline around the elbow-rotation peak
        # ---------------------------------------------------------------------

        exc = ''

        if (bas > epo[time0 - 50:time0 + 50][args].mean()) and (
            bas / epo[time0 - 50:time0 + 50][args].mean() < 0.9
            or bas / epo[time0 - 50:time0 + 50][args].mean() > 0.9
        ):

            args_base = drop_outliers(
                np.argsort(epo[time0 - 100:time0 - 25])[:len(testn)]
            )
            bas = epo[time0 - 100:time0 - 25][args_base].mean()
            epo_norm = epo - bas_true

            if bas > epo[time0 - 50:time0 + 50][args].mean():
                args_base = drop_outliers(
                    np.argsort(epo[time0 - 100:time0 + 100])[:len(testn)]
                )
                bas = epo[time0 - 100:time0 + 100][args_base].mean()
                epo_norm = epo - bas_true

            args = drop_outliers(
                np.argsort(epo_norm[time0 - 50:time0 + 50])[-20:]
            )
            exc = 'exc'

            if (bas > epo[time0 - 50:time0 + 50][args].mean()) and (
                bas / epo[time0 - 50:time0 + 50][args].mean() < 0.9
                or bas / epo[time0 - 50:time0 + 50][args].mean() > 0.9
            ):

                args_base = drop_outliers(
                    np.argsort(epo[time0 - 200:time0 - 25])[:len(testn)]
                )
                bas = epo[time0 - 200:time0 - 25][args_base].mean()
                epo_norm = epo - bas_true

                if bas > epo[time0 - 50:time0 + 50][args].mean():
                    args_base = drop_outliers(
                        np.argsort(epo[time0 - 200:time0 + 200])[:len(testn)]
                    )
                    bas = epo[time0 - 200:time0 + 200][args_base].mean()
                    epo_norm = epo - bas_true

                args = drop_outliers(
                    np.argsort(epo_norm[time0 - 50:time0 + 50])[-20:]
                )

            argss_base.append(args_base)

        else:
            argss_base.append([])

        argss.append(args)
        excs.append(exc)

        # ---------------------------------------------------------------------
        # Elbow-rotation velocity
        # ---------------------------------------------------------------------

        vel = np.diff(epo_norm)
        vels.append(vel)

        unl0 = np.argmin(np.abs(times_ma + 0.5))
        vel_peak = np.argmax(vel[unl0:time0])
        vel_lim1 = np.argmin(
            np.abs(times_ma - (times_ma[unl0:time0][vel_peak] - 0.015))
        )
        vel_lim2 = np.argmin(
            np.abs(times_ma - (times_ma[unl0:time0][vel_peak] + 0.015))
        ) + 1

        vel_maxpeak = vel[unl0:time0][vel_peak]

        vel_peaks.append(vel_peak)
        vel_peak_epo.append(vel[vel_lim1:vel_lim2].mean(0))
        vel_peaksel_epo.append(
            np.partition(vel[vel_lim1:vel_lim2], -10)[-10:].mean(0)
        )
        vel_maxpeak_epo.append(vel_maxpeak)

        rot_norm_plot.append(epo_norm)

        # ---------------------------------------------------------------------
        # Automatically estimated elbow-rotation peak
        # ---------------------------------------------------------------------

        auto1 = np.argmin(
            np.abs(times_ma - (times_ma[unload_time_idx[e]] + 0.5))
        )
        auto_max10 = np.argsort(
            epo_norm[unload_time_idx[e]:auto1]
        )[-10:]
        rot_try_auto = np.mean(
            epo_norm[unload_time_idx[e]:auto1][auto_max10]
        )

        rot_norm_auto.append(rot_try_auto)
        rot_10max_auto.append(auto_max10)

        # ---------------------------------------------------------------------
        # Manually identified elbow-rotation peak
        # ---------------------------------------------------------------------

        rot_try = np.mean(
            epo_norm[time0 - 50:time0 + 50][args]
        )
        rot_norm.append(rot_try)
        unloads.append(unl)

        # ---------------------------------------------------------------------
        # Rotation stability
        # ---------------------------------------------------------------------

        stab_idx = np.argsort(epo_norm[time0 + 50:time0 + 200])
        stab_idx_short = np.argsort(
            epo_norm[time0 + 50:time0 + 200]
        )[:len(args) * 2]

        stab = epo_norm[time0 + 50:time0 + 200][stab_idx]
        stab_short_unnorm = epo_norm[
            time0 + 50:time0 + 200
        ][stab_idx_short]

        stab = np.mean(stab) / np.abs(rot_try)
        stab_short = np.mean(stab_short_unnorm) / np.abs(rot_try)

        rot_stab.append(stab)
        rot_stab_short.append(stab_short)

        if (
            stab_short > 1.05
            or (np.abs(rot_try) / med_abs < 0.07 and stab > 0)
            or np.abs(stab_short_unnorm.mean(0)) > med_abs
        ):
            bad_stab.append(e)
            stab_yes_no.append(0)
        else:
            stab_yes_no.append(1)

    # =========================================================================
    # Save trial-level measures
    # =========================================================================

    np.save(PATHto + subj + '_peak_elbowrot_auto_basecor_epos.npy', np.array(rot_norm_auto))
    np.save(PATHto + subj + '_peak_elbowrot_basecor_epos.npy', np.array(rot_norm))
    np.save(PATHto + subj + '_decr_elbowrot_basecor_epos.npy', np.array(inh_rot_norms))
    np.save(PATHto + subj + '_vel_elbowrot_nobase_epos.npy', np.array(vel_peak_epo))
    np.save(PATHto + subj + '_vel_elbowrotmax_nobase_epos.npy', np.array(vel_maxpeak_epo))
    np.save(PATHto + subj + '_vel_elbowrotsel_nobase_epos.npy', np.array(vel_peaksel_epo))
    np.save(PATHto + subj + '_stab_elbowrot_basecor_epos.npy', np.array(rot_stab))
    np.save(PATHto + subj + '_stab_min_elbowrot_basecor_epos.npy', np.array(rot_stab_short))
    np.save(PATHto + subj + '_stab_elbowrot_true_false.npy', np.array(stab_yes_no))
    np.save(PATHto + subj + '_vel_elbrot_intime_ma_nobase_epos.npy', vels)
    np.save(PATHto + subj + '_elbow_rotation_decrease_ampl_withbase.npy', np.array(inh_rot_norms))
    np.save(PATHto + subj + '_elbowrot_norm_for_in_time_epos.npy', np.array(rot_norm_plot))

    # =========================================================================
    # Report
    # =========================================================================

    report = Report()
    report_fname = PATHto + 'html_files/' + subj + '_elbow_rotation_2peaks_estimation.html'

    rot_norm_plot_array = np.array(rot_norm_plot)
    maxplot = np.max(rot_norm_plot_array[:, time0:]) + 0.15 * np.max(rot_norm_plot_array[:, time0:])
    minplot = np.min(rot_norm_plot_array)

    if minplot < -0.015:
        minplot = -0.02

    # -------------------------------------------------------------------------
    # Trial-level figures
    # -------------------------------------------------------------------------

    for e in range(rot.shape[0]):

        add_stab = ': BAD STAB' if e in bad_stab else ''

        # Manually identified elbow-rotation peak
        fig = plt.figure(figsize=(6, 4))
        plt.axhline(med_abs, color='grey', label='Imposed Med', linestyle='--')
        plt.axhline(0, color='grey', linestyle=':', alpha=1)
        plt.axvline(0, color='grey', label='zero', linestyle=':', alpha=1)
        plt.ylim(minplot, maxplot)
        plt.plot(times_ma, rot_norm_plot_array[e, :], color='black')
        plt.axvline(times_ma[time0 - 50], color='red')
        plt.axvline(times_ma[time0 + 50], color='red')
        plt.axvline(times_ma[time0 + 50], color='maroon')
        plt.axvline(times_ma[time0 + 200], color='maroon')
        plt.axvline(times_ma[unload_time_idx[e]], color='grey', label='move start')
        plt.axvline(times_ma[unload_time_idx[e] - 100], color='grey', linestyle='--')
        plt.axvline(times_ma[time0], color='grey', linestyle='--')

        plt.plot(
            times_ma[unload_time_idx[e] - 100:time0][inhrots_idx[e]],
            [minplot / 2] * len(times_ma[unload_time_idx[e] - 100:time0][inhrots_idx[e]]),
            marker='o', color='green', label='elbow inh', markersize=5, linewidth=0
        )
        plt.plot(
            times_ma[unload_time_idx[e] - 100:time0][inhrot_bases_idx[e]],
            [minplot / 2] * len(times_ma[unload_time_idx[e] - 100:time0][inhrot_bases_idx[e]]),
            marker='o', color='orange', label='elbow inh base', markersize=5, linewidth=0
        )
        plt.plot(
            times_ma[time0 - 50:time0 + 50][argss[e]],
            [rot_norm[e]] * len(times_ma[time0 - 50:time0 + 50][argss[e]]),
            marker='o', color='red', label='elbow rot', markersize=5, linewidth=0
        )

        if excs[e] == '':
            plt.plot(
                times_ma[times01[e]:times01[e] + len(testn)],
                [0] * len(times_ma[times01[e]:times01[e] + len(testn)]),
                marker='o', color='blue', label='baseline', markersize=5, linewidth=0
            )
        else:
            plt.plot(
                times_ma[time0 - 100:time0 + 100][argss_base[e]],
                [0] * len(times_ma[time0 - 100:time0 + 100][argss_base[e]]),
                marker='o', color='blue', label='baseline', markersize=5, linewidth=0
            )

        plt.xlabel('Times, 0 - peak elbow rot')
        plt.ylabel('Elbow rotation')
        plt.title('Elbow rot = average(red) - average(blue)' + add_stab)
        plt.legend()
        plt.xlim(-1, 0.5)

        report.add_figure(
            fig, title=subj + ' estimation of max elbow rotation',
            section=subj + ' epo ' + str(e)
        )
        plt.close('all')

        # Automatically identified elbow-rotation peak
        fig = plt.figure(figsize=(6, 4))
        plt.axhline(med_abs, color='grey', label='Imposed Med', linestyle='--')
        plt.axhline(0, color='grey', linestyle=':', alpha=1)
        plt.axvline(0, color='grey', label='zero', linestyle=':', alpha=1)
        plt.ylim(minplot, maxplot)
        plt.plot(times_ma, rot_norm_plot_array[e, :], color='black')
        plt.axvline(times_ma[time0 - 50], color='red')
        plt.axvline(times_ma[time0 + 50], color='red')
        plt.axvline(times_ma[time0 + 50], color='maroon')
        plt.axvline(times_ma[time0 + 200], color='maroon')
        plt.axvline(times_ma[unload_time_idx[e]], color='grey', label='move start')
        plt.axvline(times_ma[unload_time_idx[e] - 100], color='grey', linestyle='--')
        plt.axvline(times_ma[time0], color='grey', linestyle='--')

        plt.plot(
            times_ma[unload_time_idx[e] - 100:time0][inhrots_idx[e]],
            [minplot / 2] * len(times_ma[unload_time_idx[e] - 100:time0][inhrots_idx[e]]),
            marker='o', color='green', label='elbow inh', markersize=5, linewidth=0
        )
        plt.plot(
            times_ma[unload_time_idx[e] - 100:time0][inhrot_bases_idx[e]],
            [minplot / 2] * len(times_ma[unload_time_idx[e] - 100:time0][inhrot_bases_idx[e]]),
            marker='o', color='orange', label='elbow inh base', markersize=5, linewidth=0
        )
        plt.plot(
            times_ma[unload_time_idx[e]:auto1 + 1][rot_10max_auto[e]],
            [rot_norm_auto[e]] * len(times_ma[unload_time_idx[e]:auto1 + 1][rot_10max_auto[e]]),
            marker='o', color='red', label='elbow rot AUTO', markersize=5, linewidth=0
        )

        if excs[e] == '':
            plt.plot(
                times_ma[times01[e]:times01[e] + len(testn)],
                [0] * len(times_ma[times01[e]:times01[e] + len(testn)]),
                marker='o', color='blue', label='baseline', markersize=5, linewidth=0
            )
        else:
            plt.plot(
                times_ma[time0 - 100:time0 + 100][argss_base[e]],
                [0] * len(times_ma[time0 - 100:time0 + 100][argss_base[e]]),
                marker='o', color='blue', label='baseline', markersize=5, linewidth=0
            )

        plt.xlabel('Times, 0 - peak elbow rot')
        plt.ylabel('Elbow rotation')
        plt.title('Elbow rot = average(red) - average(blue)' + add_stab)
        plt.legend()
        plt.xlim(-1, 0.5)

        report.add_figure(
            fig, title=subj + ' estimation of max elbow rotation AUTO',
            section=subj + ' epo ' + str(e)
        )
        plt.close('all')

        # Elbow-rotation velocity
        fig = plt.figure(figsize=(6, 4))
        plt.plot(times_ma[1:], np.array(vels)[e, :], color='black')
        plt.axvline(times_ma[unl0:][vel_peaks[e]], color='red')
        plt.axvline(times_ma[unl0:][vel_peaks[e]] - 0.015, color='red', linestyle='--')
        plt.axvline(times_ma[unl0:][vel_peaks[e]] + 0.015, color='red', linestyle='--')
        plt.xlabel('Times, 0 - peak elbow rot')
        plt.ylabel('Elbow rotation velocity')
        plt.title('Elbow rot velocity')
        plt.xlim(-1, 0.5)

        report.add_figure(
            fig, title=subj + ' estimation of rotation velocity',
            section=subj + ' epo ' + str(e)
        )
        plt.close('all')

    report.save(report_fname, overwrite=True, open_browser=True)