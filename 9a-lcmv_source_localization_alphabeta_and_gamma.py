#!/usr/bin/env python3
"""
LCMV source reconstruction for the bimanual movement MEG data.

The same pipeline is used for the alpha/beta and gamma analyses; only the
frequency range and output filename differ.

Set ANALYSIS to "alphabeta" or "gamma" before running.
"""

import os
import numpy as np
import mne
from mne.beamformer import make_lcmv, apply_lcmv_epochs


# ---------------------------------------------------------------------
# Analysis configuration
# ---------------------------------------------------------------------

ANALYSIS = "alphabeta"  # "alphabeta" or "gamma"

path = "path"
SUBJECTS_DIR = path + "MRI_freesurfer/"

SUBJECTS = ['BM_C0' + str(ii) for ii in np.arange(1, 10)] + ['BM_C' + str(ii) for ii in np.arange(10, 25)]

CONFIG = {
    "alphabeta": {
        "subjects": SUBJECTS,
        "fmin": 1,
        "fmax": 150,
        "output_suffix": "for_alphabeta_f1-150_-17-12_epoxvertxtimes_goodheadpos",
    },
    "gamma": {
        "subjects": SUBJECTS,
        "fmin": 25,
        "fmax": 150,
        "output_suffix": "for_gamma_f25-150_-17-12_epoxvertxtimes_goodheadpos",
    },
}

cfg = CONFIG[ANALYSIS]

# Epoching parameters
TMIN = -1.7
TMAX = 1.2
BASELINE = None
EVENT_NAME = "naturel_action_start"

# LCMV parameters
BEM_CONDUCTIVITY = (0.3,)  # single-layer BEM
SOURCE_SPACING = "ico4"
VOLUME_GRID = 5  # mm
LCMV_REG = 0.1
NOTCH_FREQS = (50, 100)


# Subcortical structures included in the mixed source space.
VOLUME_LABELS = [
    "Left-Cerebellum-Cortex", "Left-Thalamus-Proper",
    "Left-Caudate", "Left-Putamen", "Left-Pallidum",
    "Right-Cerebellum-Cortex", "Right-Thalamus-Proper",
    "Right-Caudate", "Right-Putamen", "Right-Pallidum",
]


# ---------------------------------------------------------------------
# Subject-level processing
# ---------------------------------------------------------------------

for subject in cfg["subjects"]:
    print(f"\nProcessing {subject} ({ANALYSIS})")

    # Input files
    raw_fname = os.path.join(
        path,
        f"{subject}_naturel_indpos_ica-ref_thr2_newpos_raw.fif",
    )
    trans_fname = os.path.join(
        path,
        f"{subject}_indpos_ica-ref_thr2_newpos_raw_trans.fif",
    )
    epoch_selection_fname = os.path.join(
        path,
        f"{subject}_epo_selection_unloading_and_no_emg_bad_and_no_headmove1.npy",
    )

    # -----------------------------------------------------------------
    # Build a mixed cortical + deep source space and the forward model.
    # -----------------------------------------------------------------
    bem_model = mne.make_bem_model(
        subject=subject,
        ico=4,
        conductivity=BEM_CONDUCTIVITY,
        subjects_dir=SUBJECTS_DIR,
    )
    bem = mne.make_bem_solution(bem_model)

    volume_src = mne.setup_volume_source_space(
        subject,
        mri=os.path.join(SUBJECTS_DIR, subject, "mri", "aseg.mgz"),
        pos=VOLUME_GRID,
        bem=bem,
        volume_label=VOLUME_LABELS,
        subjects_dir=SUBJECTS_DIR,
        add_interpolator=True,
    )

    surface_src = mne.setup_source_space(
        subject,
        spacing=SOURCE_SPACING,
        subjects_dir=SUBJECTS_DIR,
        add_dist=False,
        n_jobs=-1,
    )

    src = surface_src + volume_src

    # The forward model is calculated from the unfiltered raw data.
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    fwd = mne.make_forward_solution(
        raw.info,
        trans=trans_fname,
        src=src,
        bem=bem,
        meg=True,
        eeg=False,
        mindist=5.0,
        n_jobs=2,
    )

    del raw, bem, src

    # -----------------------------------------------------------------
    # Filter the data and create epochs around the action onset.
    # -----------------------------------------------------------------
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    raw.filter(cfg["fmin"], cfg["fmax"])
    raw.notch_filter(freqs=NOTCH_FREQS)

    events = mne.events_from_annotations(
        raw, event_id={EVENT_NAME: 1}
    )[0]

    epochs = mne.Epochs(
        raw,
        events,
        tmin=TMIN,
        tmax=TMAX,
        baseline=BASELINE,
        proj=False,
        preload=True,
        reject_by_annotation=False,
    )

    # Apply the previously defined trial-selection mask.
    epoch_selection = np.load(epoch_selection_fname)
    print(f"  Epochs before selection: {len(epochs)}")
    epochs = epochs[epoch_selection]
    print(f"  Epochs after selection:  {len(epochs)}")

    del raw

    # -----------------------------------------------------------------
    # Compute the LCMV spatial filters.
    # -----------------------------------------------------------------
    mag_epochs = epochs.copy().pick("mag")

    rank = mne.compute_rank(
        mag_epochs,
        tol=1e-6,
        tol_kind="relative",
    )

    data_cov = mne.compute_covariance(
        epochs,
        tmin=-1,
        tmax=0.1,
        method="empirical",
        rank=rank,
    )

    filters = make_lcmv(
        epochs.info,
        fwd,
        data_cov,
        reg=LCMV_REG,
        reduce_rank=True,
        rank=rank,
        pick_ori="max-power",
        weight_norm=None,
    )

    # -----------------------------------------------------------------
    # Apply the filters to every trial and average the source estimates.
    # -----------------------------------------------------------------
    stcs = apply_lcmv_epochs(mag_epochs, filters)

    # Each SourceEstimate has shape (sources, time).
    # Averaging here reproduces the original trial-averaging step.
    source_data = np.mean([stc.data for stc in stcs], axis=0)

    output_fname = os.path.join(
        path,
        f"{subject}_all_lcmv_maxori_deep_allepo_indepo_reg01_"
        f"{cfg['output_suffix']}.npy",
    )

    np.save(output_fname, source_data)

    print(f"  Source data shape: {source_data.shape}")
    print(f"  Saved: {output_fname}")

    del epochs, mag_epochs, filters, stcs, source_data, fwd, data_cov
