#!/usr/bin/env python3

import numpy as np
import mne
import os
import sys

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PATHto = 'pathto'
subjects_dir = 'pathto/MRI_freesurfer/'

# ---------------------------------------------------------------------
# Create fsaverage mixed source space
# ---------------------------------------------------------------------
def create_good_src_only(subject):

    conductivity = (0.3,)  # single-layer BEM

    model = mne.make_bem_model(
        subject=subject,
        ico=4,
        conductivity=conductivity,
        subjects_dir=subjects_dir
    )
    bem = mne.make_bem_solution(model)

    labels_vol = [
        'Left-Cerebellum-Cortex',
        'Left-Thalamus-Proper',
        'Left-Caudate',
        'Left-Putamen',
        'Left-Pallidum',
        'Right-Cerebellum-Cortex',
        'Right-Thalamus-Proper',
        'Right-Caudate',
        'Right-Putamen',
        'Right-Pallidum'
    ]

    # Cortical source space
    src = mne.setup_source_space(
        subject,
        spacing='ico4',
        subjects_dir=subjects_dir,
        add_dist=True,
        n_jobs=-1
    )

    # Subcortical/cerebellar volume source space
    vol_src = mne.setup_volume_source_space(
        subject,
        mri=subjects_dir + subject + '/mri/aseg.mgz',
        pos=5,
        bem=bem,
        volume_label=labels_vol,
        subjects_dir=subjects_dir,
        add_interpolator=True,
        verbose=True
    )

    # Combined surface + volume source space
    src += vol_src

    return src


# ---------------------------------------------------------------------
# Estimate subject-specific forward model
# ---------------------------------------------------------------------
def estimate_fwd(subject):

    conductivity = (0.3,)  # single-layer BEM

    model = mne.make_bem_model(
        subject=subject,
        ico=4,
        conductivity=conductivity,
        subjects_dir=subjects_dir
    )
    bem = mne.make_bem_solution(model)

    labels_vol = [
        'Left-Cerebellum-Cortex',
        'Left-Thalamus-Proper',
        'Left-Caudate',
        'Left-Putamen',
        'Left-Pallidum',
        'Right-Cerebellum-Cortex',
        'Right-Thalamus-Proper',
        'Right-Caudate',
        'Right-Putamen',
        'Right-Pallidum'
    ]

    vol_src = mne.setup_volume_source_space(
        subject,
        mri=subjects_dir + subject + '/mri/aseg.mgz',
        pos=5,
        bem=bem,
        volume_label=labels_vol,
        subjects_dir=subjects_dir,
        add_interpolator=True,
        verbose=True
    )

    src = mne.setup_source_space(
        subject,
        spacing='ico4',
        subjects_dir=subjects_dir,
        add_dist=False,
        n_jobs=-1
    )

    src += vol_src

    # Raw data used to obtain sensor information
    exp = 'naturel'
    ica = 'ica'

    raw_fname = (
        PATHto + subject +
        '_naturel_indpos_ica-ref_thr2_newpos_raw.fif'
    )

    raw = mne.io.Raw(raw_fname, preload=True)

    trans = (
        PATHto + subject +
        '_indpos_ica-ref_thr2_newpos_raw_trans.fif'
    )

    fwd = mne.make_forward_solution(
        raw.info,
        trans=trans,
        src=src,
        bem=bem,
        meg=True,
        eeg=False,
        mindist=5.0,
        n_jobs=2
    )

    return fwd


# ---------------------------------------------------------------------
# fsaverage target source space
# ---------------------------------------------------------------------
src_to = create_good_src_only('fsaverage')


# ---------------------------------------------------------------------
# Subjects
# ---------------------------------------------------------------------
SUBJECTS = [f'BM_C{ii:02d}' for ii in range(1, 25)]


# ---------------------------------------------------------------------
# Morph subject-level correlations to fsaverage
# ---------------------------------------------------------------------
for subj in SUBJECTS:

    print('\nProcessing:', subj)

    # ---------------------------------------------------------------
    # Load subject-level alpha-beta correlations
    #
    # Dimensions:
    #   source × frequency × time
    #
    # These correlations were calculated separately at each frequency
    # and time point.
    #
    # The same correlation analysis was also performed for gamma power
    # averaged across the 90–130 Hz range. In that case, the resulting
    # correlation data are source × time.
    # ---------------------------------------------------------------
    corr = np.load(
        PATHto + subj +
        '_corr_superlet_4-30Hz_-025to01_vs_inhampl_earlylateinh_2hemi.npy'
    )

    print('Correlation shape:', corr.shape)

    # ---------------------------------------------------------------
    # Subject source space and source estimate
    # ---------------------------------------------------------------
    src_ind = mne.read_source_spaces(
        PATHto + subj + '_src_ico4_mixed.fif'
    )

    stc_ind = mne.read_source_estimate(
        PATHto + subj + '_stc_mixed-stc.h5'
    )

    # ---------------------------------------------------------------
    # Subject-specific forward model
    # ---------------------------------------------------------------
    fwd_ind = estimate_fwd(subj)

    # ---------------------------------------------------------------
    # Compute morph from subject space to fsaverage
    # ---------------------------------------------------------------
    morph = mne.compute_source_morph(
        fwd_ind['src'],
        src_to=src_to,
        subject_from=subj,
        subject_to='fsaverage',
        subjects_dir=subjects_dir
    )

    # ---------------------------------------------------------------
    # Morph each alpha-beta frequency separately
    # ---------------------------------------------------------------
    morphed_corr = []

    for f in range(corr.shape[1]):

        # Extract correlation map for one frequency:
        # source × time
        corr_freq = corr[:, f, :]

        # Create source estimate containing the correlation values
        stc_corr = mne.MixedSourceEstimate(
            corr_freq,
            stc_ind.vertices,
            tmin=-0.25,
            tstep=stc_ind.tstep,
            subject=subj
        )

        # Morph to fsaverage
        stc_morph = morph.apply(stc_corr)

        morphed_corr.append(stc_morph.data)

    # Convert to:
    # frequency × source × time
    morphed_corr = np.array(morphed_corr)

    print('Morphed correlation shape:', morphed_corr.shape)

    # ---------------------------------------------------------------
    # Save morphed alpha-beta correlations
    # ---------------------------------------------------------------
    np.save(
        PATHto + subj +
        '_corr_superlet_4-30Hz_-025to01_vs_inhampl_earlylateinh_2hemi_morphed.npy',
        morphed_corr
    )