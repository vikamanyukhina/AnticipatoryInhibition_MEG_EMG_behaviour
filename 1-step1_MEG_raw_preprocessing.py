#!/usr/bin/env python3

import mne
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mne.io import read_raw_ctf
import glob
import os
from mne.preprocessing import annotate_movement, compute_average_dev_head_t
from mne.report import Report
import sys
import os
import copy
import datetime


# ============================================================
# General settings
# ============================================================

task = ['volontaire']
task_type_name = 'naturel'

pathfrom = 'pathfrom'
pathto = 'pathto'

SUBJECTS = [f'BM_C{ii:02d}' for ii in range(1, 25)]

# ============================================================
# Initialize containers
# ============================================================

raws_motor = []
raws_rest = []

bad_chpi_subj = []
bad_chpi_raw = []

type_stimulus = 'naturel_action_start'
annots = []

raws_events = []


# ============================================================
# Process each participant
# ============================================================

for s, subj in enumerate(SUBJECTS):

    add_nat = []

    for task_type in task:

        # ----------------------------------------------------
        # Locate and load the raw MEG data
        # ----------------------------------------------------

        path = pathfrom + '/' + subj + '/'
        os.chdir(path)

        files = sorted(glob.glob(subj + '_' + task_type + '*' + '.ds'))

        raws_type = []
        raws_dates = []

        if True:

            raw_fname = path + subj + '_' + task_type + '.ds'
            raw = mne.io.read_raw_ctf(raw_fname, preload=True)

            # Extract events from annotations and obtain sampling frequency
            events_true = mne.events_from_annotations(raw)

            sfreq = raw.info['sfreq']

            concat_raws_type = raw

        # ----------------------------------------------------
        # Create participant output directory and save raw data
        # ----------------------------------------------------

        try:
            os.chdir(pathto + '/' + subj)
        except:
            os.mkdir(pathto + '/' + subj)
            os.chdir(pathto + '/' + subj)

        concat_raws_type.save(
            pathto + '/' + subj + '/' + subj + '_' + task_type_name + '_indpos_raw.fif',
            overwrite=True
        )

        # ----------------------------------------------------
        # Initialize head-position information
        # ----------------------------------------------------

        av_pos = []
        init_pos = []

        report = Report()

        report_fname = (
            pathto + '/' + subj + '/' +
            subj + '_' + task_type_name +
            '_raw_parts_head_pos_report.html'
        )

        # ----------------------------------------------------
        # Extract CHPI locations and estimate head position
        # ----------------------------------------------------

        chpi_locs = mne.chpi.extract_chpi_locs_ctf(concat_raws_type)

        head_pos = mne.chpi.compute_head_pos(raw.info, chpi_locs)

        original_head_dev_t = mne.transforms.invert_transform(
            raw.info['dev_head_t']
        )

        if subj == 'BM_C13':
            pass

        else:

            # Calculate the average head position
            average_head_dev_t = mne.transforms.invert_transform(
                compute_average_dev_head_t(raw, head_pos)
            )

            av_pos.append(
                average_head_dev_t['trans'][:3, 3].tolist()
            )

            init_pos.append(
                original_head_dev_t['trans'][:3, 3].tolist()
            )

            # ------------------------------------------------
            # Plot estimated head positions
            # ------------------------------------------------

            fig = mne.viz.plot_head_positions(head_pos)

            for ax, val, val_ori in zip(
                fig.axes[::2],
                average_head_dev_t['trans'][:3, 3],
                original_head_dev_t['trans'][:3, 3]
            ):
                ax.axhline(1000 * val, color='r')
                ax.axhline(1000 * val_ori, color='g')

            fig.suptitle(
                "Red: average head pos; Green: initial head pos",
                fontsize=10,
                y=1
            )

            report.add_figure(
                fig,
                title=raw.filenames[0],
                section='Raw CHPI info'
            )

            # Calculate mean average and initial head positions
            new_av_pos = np.array(av_pos).mean(0)
            new_init_pos = np.array(init_pos).mean(0)

            # ------------------------------------------------
            # Plot head positions of all raw parts
            # ------------------------------------------------

            fig = plt.figure(figsize=(8, 6))

            pos = av_pos + init_pos

            for i in range(len(av_pos)):
                plt.scatter(np.array(av_pos[i]) * 1000, [0, 1, 2])
                plt.scatter(np.array(init_pos[i]) * 1000, [3, 4, 5])

            plt.scatter(
                new_av_pos * 1000,
                [0, 1, 2],
                label='Mean av',
                color='black'
            )

            plt.scatter(
                new_init_pos * 1000,
                [3, 4, 5],
                label='Mean init',
                color='lightblue'
            )

            plt.yticks(
                [0, 1, 2, 3, 4, 5],
                labels=['X', 'Y', 'Z', 'Xorig', 'Yorig', 'Zorig']
            )

            plt.xlabel('mm')

            # Mark 1 mm and 2 cm deviations around the mean positions
            for c in np.array(
                new_av_pos.tolist() + new_init_pos.tolist()
            ) * 1000:

                plt.axvline(
                    c + 1,
                    color='grey',
                    linestyle=':'
                )

                plt.axvline(
                    c - 1,
                    color='grey',
                    linestyle=':'
                )

                plt.axvspan(
                    c - 20,
                    c + 20,
                    color='red',
                    alpha=0.1
                )

            plt.legend()

            plt.title(
                'Grey: 1mm, Pink: 2cm, bad_chpi: '
                + ', '.join([name[-7:-5] for name in bad_chpi_raw])
            )

            report.add_figure(
                fig,
                title='Av and Init head pos',
                section='Raw CHPI info'
            )

        # ----------------------------------------------------
        # Close figures and save the quality-control report
        # ----------------------------------------------------

        plt.close('all')

        report.save(
            report_fname,
            overwrite=True,
            open_browser=True
        )

    # Free memory before processing the next participant
    del raw, concat_raws_type, raws_type
