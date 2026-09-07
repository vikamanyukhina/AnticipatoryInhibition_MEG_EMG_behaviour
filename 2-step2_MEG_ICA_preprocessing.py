#!/usr/bin/env python3

import mne
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs, corrmap
from mne.report import Report
import math
import pickle
import time


# ============================================================
# General settings
# ============================================================

task_type_name = 'naturel'
subj_type = 'Children'
subj_type2 = 'Control'

pathto = 'pathto' + subj_type + '/' + subj_type2

# Run for each subject separately
SUBJECTS = ['BM_C01']

ica = 'yes'  # no
notch = 'yes'
info = 'indpos'

plt.close('all')

task_type = 'naturel'


# ============================================================
# Process each subject
# ============================================================

for s, subj in enumerate(SUBJECTS):

    # --------------------------------------------------------
    # IMPORTANT:
    # Check that the correct markers are present and that
    # bad segments have been correctly marked.
    # This should be checked manually before continuing.
    # --------------------------------------------------------

    # raw_motor = mne.io.read_raw(
    #     pathto + '/' + subj + '/' + subj + '_' +
    #     task_type + '_indpos_raw.fif',
    #     preload=True
    # )

    # Opens interactive window
    # raw_motor.plot(block=True)

    # raw_motor.save(
    #     pathto + '/' + subj + '/' + subj + '_' +
    #     task_type + '_indpos_raw.fif',
    #     overwrite=True
    # )

    # --------------------------------------------------------
    # Load the raw MEG data
    # --------------------------------------------------------

    raw_motor = mne.io.read_raw(
        pathto + '/' + subj + '/' + subj + '_' +
        task_type + '_indpos_raw.fif',
        preload=True
    )

    # Select MEG and reference MEG channels for visual inspection
    select_picks = np.concatenate(
        (
            mne.pick_types(raw_motor.info, meg=True)[-32:],
            mne.pick_types(raw_motor.info, meg=False, ref_meg=True),
        )
    )

    # Parameters for interactive raw-data visualization
    plot_kwargs = dict(
        duration=100,
        order=select_picks,
        n_channels=len(select_picks),
        scalings={"mag": 8e-13, "ref_meg": 2e-11},
    )

    raw_motor.plot(**plot_kwargs)

    # Inspect the power spectrum up to 30 Hz
    raw_motor.compute_psd(fmax=30).plot(
        picks="data",
        exclude="bads"
    )


    # --------------------------------------------------------
    # Reload the unaffected raw file before preprocessing
    # --------------------------------------------------------

    raw_motor = mne.io.read_raw(
        pathto + '/' + subj + '/' + subj + '_' +
        task_type + '_indpos_raw.fif',
        preload=True
    )

    # High-pass filter the data at 1 Hz
    raw_tog = raw_motor.copy().filter(
        l_freq=1.,
        h_freq=None
    )

    # Apply notch filtering if enabled
    if notch == 'yes':
        raw_tog.notch_filter(freqs=[50, 100])

        os.chdir(pathto + '/' + subj + '/ica_notch/')


    # ========================================================
    # ICA decomposition
    # ========================================================

    ica_kwargs = dict(
        method="picard",
    )

    # Select MEG and reference MEG channels for ICA
    all_picks = mne.pick_types(
        raw_tog.info,
        meg=True,
        ref_meg=True
    )

    # Fit ICA
    ica_tog = ICA(
        n_components=70,
        max_iter="auto",
        random_state=42,
        allow_ref_meg=True,
        **ica_kwargs
    )

    ica_tog.fit(
        raw_tog,
        picks=all_picks
    )

    # Save the ICA solution
    ica_tog.save(
        subj + '_' + task_type + '_' + info + '.fif',
        overwrite=True
    )


    # ========================================================
    # Create ICA quality-control report
    # ========================================================

    report = Report()

    report_fname = (
        subj + '_ica_' + task_type + '_' +
        info + '_comp_ref_report.html'
    )

    # Plot ICA component topographies
    figs = ica_tog.plot_components()

    for j in range(len(figs)):

        report.add_figure(
            figs[j],
            title=str(j + 1) + 'th 20 ICs topoplots',
            section='ICs topoplots'
        )

        # Select good magnetometer channels for PSD plots
        take = mne.pick_types(
            raw_tog.info, meg='mag', eeg=False, stim=False, eog=False,
            ecg=False, emg=False, ref_meg=False, misc=False, resp=False,
            chpi=False, exci=False, ias=False, syst=False,
            include=[], exclude='bads', selection=None
        )


    # Plot magnetometer spectra from 1-140 Hz
    fig = raw_tog.plot_psd(
        tmin=0.0,
        tmax=raw_tog.n_times,
        fmin=1,
        fmax=140,
        proj=False,
        n_fft=2048,
        picks=take,
        area_mode='std'
    )

    report.add_figure(
        fig,
        title='Plot magnetometers spectra for each channel, 1-140Hz',
        section='Mag plot'
    )

    # Plot magnetometer spectra from 1-45 Hz
    fig = raw_tog.plot_psd(
        tmin=0.0,
        tmax=raw_tog.n_times,
        fmin=1,
        fmax=45,
        proj=False,
        n_fft=2048,
        picks=take,
        area_mode='std'
    )

    report.add_figure(
        fig,
        title='Plot magnetometers spectra for each channel, 1-45Hz',
        section='Mag plot'
    )


    # Save the ICA quality-control report
    report.save(
        report_fname,
        open_browser=True,
        overwrite=True
    )

    plt.close('all')


    # ========================================================
    # Inspect ICA components and manually identify artifacts
    # ========================================================

    # IMPORTANT:
    # Inspect the ICA components carefully before selecting
    # components for exclusion.

    sources = ica_tog.get_sources(raw_tog)
    sources.plot(block=True)


    # Specify biological components to exclude manually
    ecg = []
    veog = []
    heog = []
    emg = []

    # ECG components
    ecg_inp = input('Which ECG components to exclude?')

    if ecg_inp != '':
        ecg = list(map(int, (ecg_inp).split(',')))

    # Vertical EOG components
    veog_i = input('Which vertical EOG components to exclude?')

    if veog_i != '':
        veog = list(map(int, (veog_i).split(',')))

    # Horizontal EOG components
    heog_i = input('Which horizontal EOG components to exclude?')

    if heog_i != '':
        heog = list(map(int, (heog_i).split(',')))

    # EMG components
    emg_i = input('Which EMG components to exclude?')

    if emg_i != '':
        emg = list(map(int, (emg_i).split(',')))


    # Combine manually selected EOG components
    eog_comps = veog + heog

    # Combine all manually selected biological components
    bio_comps = ecg + eog_comps + emg

    # Save manually selected component indices
    np.save(
        'dropped_eog_comps.npy',
        np.array(eog_comps)
    )

    np.save(
        'dropped_ecg_comps.npy',
        np.array(ecg)
    )

    np.save(
        'dropped_emg_comps.npy',
        np.array(emg)
    )


    # ========================================================
    # Identify bad reference components automatically
    # ========================================================

    bad_comps, scores = ica_tog.find_bads_ref(
        raw_tog,
        threshold=2
    )

    # Save automatically identified reference components
    np.save(
        'dropped_ref_2_comps.npy',
        np.array(bad_comps)
    )


    # Combine automatically and manually excluded components
    excl = bad_comps + bio_comps


    # ========================================================
    # Save information about excluded ICA components
    # ========================================================

    dd = {
        'Subject': subj,
        'Condition': task_type,
        'N excl comp': len(excl),
        'Ref excl': bad_comps,
        'EOGv excl': veog,
        'EOGh excl': heog,
        'ECG excl': ecg,
        'EMG excl': emg
    }

    # Create a binary pickle file
    f = open(
        subj + "_ICA_" + task_type + "_" +
        info + "_info_log_ref-thr2.pkl",
        "wb"
    )

    pickle.dump(dd, f)

    f.close()


    # ========================================================
    # Apply ICA and save cleaned data
    # ========================================================

    raw_motor_ica = ica_tog.apply(
        raw_motor.copy(),
        exclude=excl
    )


    # --------------------------------------------------------
    # Optional comparison of cleaned and original data
    # --------------------------------------------------------

    # raw_motor_ica.plot()
    # raw_motor.plot(block=True)


    # --------------------------------------------------------
    # Select bad segments manually
    # Use the interactive window to identify muscle or other
    # artifacts that should be annotated.
    # --------------------------------------------------------

    # raw_motor_ica.plot(block=True)


    # Save ICA-cleaned raw data
    raw_motor_ica.save(
        subj + '_' + task_type +
        '_indpos_ica-ref_thr2_raw.fif',
        overwrite=True
    )


# ============================================================
# Optional check of ICA components across subjects
# ============================================================

# dfs = []

# for subj in SUBJECTS:

#     filename = (
#         pathto + '/' + subj + '/ica_notch/' +
#         subj + "_ICA_" + task_type + "_" +
#         info + "_info_log_ref-thr2.pkl"
#     )

#     with open(filename, "rb") as f:
#         dd = pickle.load(f)

#     for comp_type in ['EOGv excl']:
#         # ["Ref excl", "EOGv excl", "EOGh excl", "ECG excl", "EMG excl"]

#         # for comp in dd[comp_type]:

#         dfs.append({
#             "Subject": dd["Subject"],
#             "Condition": dd["Condition"],
#             "Component_type": comp_type,
#             "Component": len(dd[comp_type])
#         })

# df_eogv = pd.DataFrame(dfs)