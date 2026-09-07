#!/usr/bin/env python3

import mne
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import scipy.integrate as integrate
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs, corrmap
from mne.report import Report
import math
from mne.preprocessing import annotate_muscle_zscore


# Paths and subject selection
subj_type = 'Children'

pathfrom = 'pathfrom'
pathto = 'pathto'

info = 'indpos'
SUBJECTS = ['BM_C0' + str(ii) for ii in np.arange(1, 10)] + ['BM_C' + str(ii) for ii in np.arange(10, 25)]
ica = 'yes'

conds = ['impose']

# Store subject-level and trial-level elbow-rotation measures
imposed_elbows = []
all_epochs_rot = []
imposed_rows = []

imposed_rot_ampl = []
imposed_vel_ampl = []
imposed_vel_time = []

inh_subjs = []

bics = []
bracs = []
trics = []

report = Report()
report_fname = pathfrom + '/Allsubj_asd_imposed_elbow_rotation_peak_estimation_and_EMG_inhibition.html'


for s, subj in enumerate(SUBJECTS):

    for cond in conds:
        cond = 'imposed'
        type_movement = 'Imposed unloading'

        # Load raw data and epoch around unloading onset (delestage)
        raw = mne.io.read_raw(pathto + '/' + subj + '/' + subj + '_' + cond + '_indpos_raw.fif', preload=True)
        events_true = mne.events_from_annotations(raw, event_id={'delestage': 1})[0]
        allepochs = mne.Epochs(raw, events_true, tmin=-1, tmax=1, baseline=None,
                               proj=False, preload=True, reject_by_annotation=False)

        # Extract and rectify EMG
        allepo_muscles = allepochs.copy().pick_channels(['BicG', 'TriG', 'SupG'])
        data_muscles = allepo_muscles.get_data()

        data_demean_epochs = []
        for i in range(len(data_muscles)):
            data_demean_ch = []
            for j in range(data_muscles.shape[1]):
                data_demean_ch.append(data_muscles[i, j, :] - np.mean(data_muscles[i, j, :]))
            data_demean_epochs.append(data_demean_ch)

        allepo_demean = mne.EpochsArray(np.array(data_demean_epochs), allepo_muscles.info,
                                        events=allepo_muscles.events, tmin=allepo_muscles.tmin)

        high = 150
        low = 25
        allepo_demean = allepo_demean.filter(low, high)
        allepo_data_filter = np.abs(allepo_demean.get_data())

        ch_names_emg = allepo_muscles.ch_names
        muscles_names = ['Biceps L', 'Triceps L', 'Brachioradialis L']

        # Plot subject-level average EMG for quality control
        fig = plt.figure()
        plt.title('Biceps L')
        plt.plot(allepochs.times, allepo_data_filter[:, 0, :].mean(0), color='black')
        plt.xlim(-0.5, 0.5)
        report.add_figure(fig, title='Epoch-averaged EMG, Biceps L',
                          section=subj + ' EMG epoch average ' + str(len(allepochs)))
        plt.close('all')

        fig = plt.figure()
        plt.title('Triceps L')
        plt.plot(allepochs.times, allepo_data_filter[:, 1, :].mean(0), color='black')
        plt.xlim(-0.5, 0.5)
        report.add_figure(fig, title='Epoch-averaged EMG, Triceps L',
                          section=subj + ' EMG epoch average ' + str(len(allepochs)))
        plt.close('all')

        fig = plt.figure()
        plt.title('Brachioradialis L')
        plt.plot(allepochs.times, allepo_data_filter[:, 2, :].mean(0), color='black')
        plt.xlim(-0.5, 0.5)
        report.add_figure(fig, title='Epoch-averaged EMG, Brachioradialis L',
                          section=subj + ' EMG epoch average ' + str(len(allepochs)))
        plt.close('all')

        bics.append(allepo_data_filter[:, 0, :].mean(0))
        bracs.append(allepo_data_filter[:, 2, :].mean(0))
        trics.append(allepo_data_filter[:, 1, :].mean(0))

        # Extract elbow rotation and invert sign to obtain the desired direction
        rot = allepochs.copy().pick_channels(['UADC001-2800']).get_data() * (-1)

        def drop_outliers(data):
            new_data = []
            q75, q25 = np.percentile(data, [75, 25])
            iqr = q75 - q25
            for i in data:
                if i > q25 - (1.5 * iqr) and i < q75 + (1.5 * iqr):
                    new_data.append(i)
            return np.array(new_data)

        def moving_average(data, window):
            new_data = []
            i = 0
            while i < len(data) - window + 1:
                new_data.append(data[i:i + window + 1].mean())
                i += 1
            return np.array(new_data)

        vel_rot_max = []
        vel_rot_time = []
        imposed_rot = []
        zero = np.argwhere(allepochs.times == 0)[0][0]
        argss = []
        maxess = []
        bass = []
        epos = []

        # Estimate post-unloading elbow rotation and maximal rotation velocity
        for e in range(rot.shape[0]):

            times_ma = moving_average(allepochs.times, 50)
            zero_ma = np.argmin(np.abs(times_ma - 0))

            bas_ma = moving_average(rot[e, 0, zero - 100:zero], 50)
            bas = np.mean(bas_ma)
            bass.append(bas)

            epo = moving_average(rot[e, 0, :], 50)
            epos.append(epo)

            args = drop_outliers(np.argsort(epo[zero_ma:])[-20:])
            argss.append(args)
            maxes = np.mean(epo[zero_ma:][args])
            maxess.append(maxes)

            imposed_rot.append(maxes - bas)

            vel = np.diff(epo)
            max_vel = np.argmin(np.abs(vel - np.mean(sorted(vel)[-10:])))
            vel_rot_max.append(np.mean(sorted(vel)[-10:]))
            vel_rot_time.append(times_ma[1:][max_vel])

        mmm = np.mean(drop_outliers(imposed_rot))
        imposed_elbows.append(mmm)
        imposed_rows.append(drop_outliers(imposed_rot))

        imposed_rot_ampl.append(np.median(drop_outliers(imposed_rot)))
        imposed_vel_ampl.append(np.median(drop_outliers(vel_rot_max)))
        imposed_vel_time.append(np.median(drop_outliers(vel_rot_time)))

        # Plot distribution of estimated elbow rotation
        fig = plt.figure(figsize=(10, 8))
        plt.hist(drop_outliers(imposed_rot), bins=40, color='darkblue', alpha=0.5)
        plt.xlabel('Elbow rotation amplitude, baseline-corrected')
        plt.ylabel('N; from ' + str(rot.shape[0]) + ' epochs')
        plt.axvline(np.median(drop_outliers(imposed_rot)), color='red',
                    label=str(round(np.median(drop_outliers(imposed_rot)), 5)))
        plt.legend()
        report.add_figure(fig, title='Elbow rotation amplitude after unloading',
                          section='Elbow rotation amplitude - ' + type_movement)

        # Plot distribution of maximal elbow-rotation velocity
        fig = plt.figure(figsize=(10, 8))
        plt.hist(drop_outliers(vel_rot_max), bins=40, color='darkblue', alpha=0.5)
        plt.xlabel('Maximal elbow-rotation velocity')
        plt.ylabel('N; from ' + str(rot.shape[0]) + ' epochs')
        plt.axvline(np.median(drop_outliers(vel_rot_max)), color='red',
                    label=round(np.median(drop_outliers(vel_rot_max)), 5))
        plt.legend()
        report.add_figure(fig, title='Maximal velocity of elbow rotation',
                          section='Elbow rotation amplitude - ' + type_movement)

        # Plot distribution of timing of maximal elbow-rotation velocity
        fig = plt.figure(figsize=(10, 8))
        plt.hist(drop_outliers(vel_rot_time), bins=40, color='darkblue', alpha=0.5)
        plt.xlabel('Time of maximal elbow-rotation velocity (s)')
        plt.ylabel('N; from ' + str(rot.shape[0]) + ' epochs')
        plt.axvline(np.median(drop_outliers(vel_rot_time)), color='red',
                    label=round(np.median(drop_outliers(vel_rot_time)), 5))
        plt.legend()
        report.add_figure(fig, title='Timing of maximal elbow-rotation velocity',
                          section='Elbow rotation amplitude - ' + type_movement)

        # Plot individual trials and the points used to estimate rotation and velocity
        for e in range(rot.shape[0]):

            fig = plt.figure()
            plt.axvline(0, color='red', label='delestage', alpha=0.5)
            plt.plot(times_ma, epos[e], color='black')
            plt.plot(times_ma[zero_ma:][argss[e]], [maxess[e]] * len(times_ma[zero_ma:][argss[e]]),
                     marker='o', color='red', label='elbow rotation', markersize=5, linewidth=0)
            plt.plot(times_ma[zero_ma - len(bas_ma):zero_ma], [bass[e]] * len(bas_ma),
                     marker='o', color='blue', label='baseline', markersize=5, linewidth=0)
            plt.xlabel('Time relative to delestage (s)')
            plt.ylabel('Elbow rotation')
            plt.title('Elbow rotation = mean post-unloading rotation - mean baseline')
            plt.legend()
            report.add_figure(fig, title=subj + ' estimation of maximal elbow rotation',
                              section=subj + ' epoch ' + str(e) + ' imposed')

            fig = plt.figure()
            plt.axvline(0, color='red', label='delestage', alpha=0.5)
            plt.plot(times_ma[1:], np.diff(epos[e]), color='black')
            max_vel = np.argmin(np.abs(vel - np.mean(sorted(vel)[-10:])))
            plt.axvline(times_ma[1:][max_vel], color='blue', linestyle='--')
            plt.xlabel('Time relative to delestage (s)')
            plt.ylabel('Elbow-rotation velocity')
            plt.title('Maximal elbow-rotation velocity at ' + str(times_ma[1:][max_vel]))
            plt.legend()
            report.add_figure(fig, title=subj + ' estimation of elbow-rotation velocity',
                              section=subj + ' epoch ' + str(e) + ' imposed')
            plt.close('all')


# Average EMG across subjects
bics = np.array(bics).mean(0)
bracs = np.array(bracs).mean(0)
trics = np.array(trics).mean(0)


# Smooth EMG using a moving-average window
window_size = 18  # 15 ms
overlap_samples = 12  # 10 ms

def data_ma(inh, times_inh):
    inh_ma = []
    for e in range(len(inh)):
        inhma = np.array([np.mean(inh[e, i:i + window_size])
                          for i in range(0, inh.shape[1] - window_size + 1, overlap_samples)])
        inh_ma.append(inhma)
    times_ma = np.array([np.mean(times_inh[i:i + window_size])
                         for i in range(0, times_inh.shape[0] - window_size + 1, overlap_samples)])
    return np.array(inh_ma), times_ma


# Plot grand-average EMG around imposed unloading
fig = plt.figure()
data, time = data_ma(bics, allepochs.times)
plt.title('Biceps L')
plt.plot(time, -data.mean(0), color='black')
plt.xlim(-0.5, 0.5)
plt.axvline(0.05, color='red', label='0.05')
plt.axvline(0.075, color='red', label='0.075')
plt.axvline(0.1, color='red', label='0.1')
plt.legend()
report.add_figure(fig, title='Subject-averaged EMG, Biceps L',
                  section='Subject-averaged EMG ' + str(len(allepochs)))
plt.close('all')

fig = plt.figure()
data, time = data_ma(trics, allepochs.times)
plt.title('Triceps L')
plt.plot(time, data.mean(0), color='black')
plt.xlim(-0.5, 0.5)
plt.axvline(0.05, color='red', label='0.05')
plt.axvline(0.075, color='red', label='0.075')
plt.axvline(0.1, color='red', label='0.1')
plt.legend()
report.add_figure(fig, title='Subject-averaged EMG, Triceps L',
                  section='Subject-averaged EMG ' + str(len(allepochs)))
plt.close('all')

fig = plt.figure()
data, time = data_ma(bracs, allepochs.times)
plt.title('Brachioradialis L')
plt.plot(time, data.mean(0), color='black')
plt.xlim(-0.5, 0.5)
plt.axvline(0.05, color='red', label='0.05')
plt.axvline(0.075, color='red', label='0.075')
plt.axvline(0.1, color='red', label='0.1')
plt.legend()
report.add_figure(fig, title='Subject-averaged EMG, Brachioradialis L',
                  section='Subject-averaged EMG ' + str(len(allepochs)))
plt.close('all')


# Save report and subject-level elbow-rotation measures
report.save(report_fname, overwrite=True, open_browser=True)

df_rot = pd.DataFrame({'Subjects': SUBJECTS, 'Mean elbow rot': imposed_elbows})
df_rot.to_csv(pathto + '/elbow_rot_imposed_children.csv')

dfs = pd.DataFrame({'Epo norm ampl': all_epochs_rot})
dfs.to_csv(pathto + '/elbow_rot_normalized_all_epos_children.csv')

df_rot = pd.DataFrame({'Subjects': SUBJECTS, 'Median elbow rot ampl': imposed_rot_ampl})
df_rot.to_csv(pathto + '/med_elbow_rot_imposed_children.csv')

df_rot = pd.DataFrame({'Subjects': SUBJECTS, 'Median elbow vel rot': imposed_vel_ampl})
df_rot.to_csv(pathto + '/med_elbow_vel_imposed_children.csv')

df_rot = pd.DataFrame({'Subjects': SUBJECTS, 'Median elbow vel time': imposed_vel_time})
df_rot.to_csv(pathto + '/med_elbow_vel_time_imposed_children.csv')