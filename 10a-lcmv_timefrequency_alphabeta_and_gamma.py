"""
Time-frequency analysis of source-reconstructed LCMV data.

This script performs the post-LCMV time-frequency analyses used for the
children/ASD MEG dataset:

1. Alpha-beta range:
   - Adaptive superlet transform
   - Power estimation
   - FOOOF-based aperiodic correction
   - Saving total and periodic 4-30 Hz power

2. High-gamma range:
   - Multitaper time-frequency decomposition
   - Log10 transformation
   - Saving 90-130 Hz average power


Notes
-----
The numerical operations, parameters, frequency ranges, time windows,
padding, scaling, and output naming conventions are retained from the
original analysis scripts. The code is reorganized and documented for
clarity and reproducibility without changing the analysis procedure.
"""

import os
import sys

import jax.numpy as jnp
import mne
import numpy as np
from fooof import FOOOF
from mne.time_frequency import tfr_array_multitaper

from superlet_complex import adaptive_superlet_transform


# =============================================================================
# Configuration
# =============================================================================

PATH_TO_DATA = "pathfrom"

ALPHA_BETA_OUTPUT_DIR = "/pathto/lcmv_periodic_alphabeta_sources_deep/"

GAMMA_OUTPUT_DIR = "/pathto/lcmv_periodic_sources_deep/"

SAMPLING_FREQUENCY = 600

# Time axis of the original LCMV source data.
TIMES = np.linspace(-1.7, 1.2, 1741)

# -------------------------------------------------------------------------
# Alpha-beta / superlet analysis
# -------------------------------------------------------------------------
SUPERLET_FREQS = np.arange(1, 51)

SUPERLET_BASE_CYCLE = 3
SUPERLET_MIN_ORDER = 1
SUPERLET_MAX_ORDER = 20

# Time window retained after the superlet transform.
SUPERLET_WINDOW = (-0.3, 0.3)

# Frequency range used for the periodic component.
PERIODIC_FREQ_RANGE = (4, 30)

# Frequency range used for the FOOOF fit.
FOOOF_FREQ_RANGE = [2, 50]

# -------------------------------------------------------------------------
# High-gamma / multitaper analysis
# -------------------------------------------------------------------------
GAMMA_FREQS = np.arange(90, 131)

# Symmetric padding used before the multitaper transform.
TFR_PADDING = 300

# Scaling applied to source time series before multitaper analysis.
SOURCE_SCALING = 10e5

# Time window used for high-gamma averaging.
GAMMA_WINDOW = (-0.5, 0.5)

# Multitaper time-bandwidth parameter.
TIME_BANDWIDTH = 4


# =============================================================================
# Input data
# =============================================================================

def load_lcmv_source_data(subject, analysis):
    """
    Load previously computed LCMV source-reconstructed time series.

    Parameters
    ----------
    subject : str
        Subject identifier.
    analysis : {"alpha_beta", "gamma"}
        Selects the LCMV input file corresponding to the analysis.

    Returns
    -------
    data_dif : numpy.ndarray
        LCMV source data with dimensions:
        epochs × sources × time.
    """
    if analysis == "alpha_beta":
        filename = (
            PATH_TO_DATA
            + subject
            + "_all_lcmv_maxori_deep_allepo_indepo_reg01_"
            "for_alphabeta_f1-150_-17-12_epoxvertxtimes_goodheadpos.npy"
        )
    elif analysis == "gamma":
        filename = (
            PATH_TO_DATA
            + subject
            + "_all_lcmv_maxori_deep_allepo_indepo_reg01_"
            "for_gamma_f25-150_-17-12_epoxvertxtimes_goodheadpos.npy"
        )
    else:
        raise ValueError(
            "analysis must be either 'alpha_beta' or 'gamma'"
        )

    data_dif = np.load(filename)
    print(f"{subject} ({analysis}): LCMV data shape = {data_dif.shape}")

    return data_dif


# =============================================================================
# Time-window helper
# =============================================================================

def get_time_indices(times, window):
    """
    Find the sample indices corresponding to a requested time window.

    The index selection reproduces the original analysis:
    the nearest sample to the lower bound is selected, and the nearest
    sample to the upper bound is selected with the end index incremented
    by one for slicing.
    """
    t1 = np.argmin(np.abs(times + abs(window[0])))
    t2 = np.argmin(np.abs(times - window[1])) + 1

    return t1, t2


# =============================================================================
# Alpha-beta analysis: adaptive superlet
# =============================================================================

def compute_superlet_power(data, sfreq, freqs):
    """
    Compute adaptive-superlet power for one source time series.

    Parameters
    ----------
    data : numpy.ndarray
        One source time series.
    sfreq : float
        Sampling frequency in Hz.
    freqs : numpy.ndarray
        Frequencies of interest.

    Returns
    -------
    tfr_power : numpy.ndarray
        Superlet power, frequency × time.
    """
    tfr = adaptive_superlet_transform(
        data,
        freqs,
        sampling_freq=sfreq,
        base_cycle=SUPERLET_BASE_CYCLE,
        min_order=SUPERLET_MIN_ORDER,
        max_order=SUPERLET_MAX_ORDER,
        mode="add",
    )

    tfr_power = jnp.abs(tfr) ** 2

    return tfr_power


def run_alpha_beta_analysis(data_dif, subject):
    """
    Run the adaptive-superlet analysis and FOOOF-based periodic correction.

    The analysis is performed separately for every source location.
    """
    t1, t2 = get_time_indices(TIMES, SUPERLET_WINDOW)
    f1 = np.argmin(np.abs(SUPERLET_FREQS - PERIODIC_FREQ_RANGE[0]))
    f2 = np.argmin(np.abs(SUPERLET_FREQS - PERIODIC_FREQ_RANGE[1])) + 1

    print(
        f"Superlet time window: {SUPERLET_WINDOW[0]} to "
        f"{SUPERLET_WINDOW[1]} s"
    )
    print(
        f"Periodic frequency range: {PERIODIC_FREQ_RANGE[0]}-"
        f"{PERIODIC_FREQ_RANGE[1]} Hz"
    )

    for source_idx in range(data_dif.shape[1]):
        print(f"{subject}: alpha-beta source {source_idx}")

        tfr_lab = []
        tfr_lablog10 = []

        for epoch_idx in range(len(data_dif)):
            stc_lab_data = data_dif[epoch_idx, source_idx, :]

            # Adaptive superlet transform.
            tfr_data = compute_superlet_power(
                stc_lab_data,
                SAMPLING_FREQUENCY,
                SUPERLET_FREQS,
            )

            print(f"Superlet power shape: {tfr_data.shape}")

            # Retain the predefined analysis window.
            tfr_lab.append(tfr_data[:, t1:t2])

        epo_power = np.array(tfr_lab)
        print(f"Total superlet power shape: {epo_power.shape}")

        # -----------------------------------------------------------------
        # Estimate the aperiodic component from the average spectrum.
        # -----------------------------------------------------------------
        epo_av = epo_power.mean(0).mean(1)

        pers = []
        apers = []

        ff = FOOOF()
        ff.fit(SUPERLET_FREQS, epo_av, FOOOF_FREQ_RANGE)

        ap = 10 ** ff._ap_fit

        print(f"FOOOF frequency shape: {ff.freqs.shape}")

        # -----------------------------------------------------------------
        # Remove the FOOOF-estimated aperiodic component.
        # -----------------------------------------------------------------
        per = (
            epo_power[:, f1:f2, :]
            - np.expand_dims(
                np.expand_dims(ap, axis=0),
                axis=2,
            )[:, f1:f2, :]
        )

        print(f"Periodic power shape: {np.array(per).shape}")

        # -----------------------------------------------------------------
        # Save periodic and total power.
        # -----------------------------------------------------------------
        np.save(
            ALPHA_BETA_OUTPUT_DIR
            + subject
            + "/"
            + subject
            + "_periodic_power_4-30Hz_superlet_20_-03to03_epo_source_"
            + str(source_idx)
            + ".npy",
            per,
        )

        np.save(
            ALPHA_BETA_OUTPUT_DIR
            + subject
            + "/"
            + subject
            + "_total_power_4-30Hz_superlet_20_-03to03_epo_source_"
            + str(source_idx)
            + ".npy",
            epo_power[:, f1:f2, :],
        )


# =============================================================================
# High-gamma analysis: multitaper
# =============================================================================

def compute_multitaper_power(data, sfreq, freqs):
    """
    Compute multitaper power for one padded source time series.

    Parameters
    ----------
    data : numpy.ndarray
        One source time series.
    sfreq : float
        Sampling frequency in Hz.
    freqs : numpy.ndarray
        Frequencies of interest.

    Returns
    -------
    tfr_power : numpy.ndarray
        Log10 multitaper power, frequency × time.
    """
    ddd = np.pad(
        data * SOURCE_SCALING,
        TFR_PADDING,
        mode="symmetric",
    )

    tfr_data = tfr_array_multitaper(
        np.expand_dims(np.expand_dims(ddd, 0), 0),
        sfreq,
        freqs,
        time_bandwidth=TIME_BANDWIDTH,
        output="power",
        n_jobs=-1,
    )

    # Remove the padding and convert power to log10 scale.
    tfr_data = np.log10(
        tfr_data[0, 0, :, TFR_PADDING:-TFR_PADDING]
    )

    return tfr_data


def run_gamma_analysis(data_dif, subject):
    """
    Run the high-gamma multitaper analysis.

    Power is averaged over the -0.5 to 0.5 s window for each epoch and
    source location.
    """
    t1, t2 = get_time_indices(TIMES, GAMMA_WINDOW)

    print(
        f"Multitaper time window: {GAMMA_WINDOW[0]} to "
        f"{GAMMA_WINDOW[1]} s"
    )
    print("High-gamma frequency range: 90-130 Hz")

    for source_idx in range(data_dif.shape[1]):
        print(f"{subject}: high-gamma source {source_idx}")

        stc_lab_data = data_dif[:, source_idx, :]

        tfr_lab = []
        tfr_lablog10 = []

        for epoch_idx in range(len(data_dif)):
            tfr_data = compute_multitaper_power(
                stc_lab_data[epoch_idx, :],
                SAMPLING_FREQUENCY,
                GAMMA_FREQS,
            )

            # Average power over the -0.5 to 0.5 s window.
            tfr_lab.append(
                tfr_data[:, t1:t2].mean(0)
            )

        tfr_lab = np.array(tfr_lab)

        print(f"High-gamma power shape: {tfr_lab.shape}")

        np.save(
            GAMMA_OUTPUT_DIR
            + subject
            + "/"
            + subject
            + "_pow_90-130av_multitaper_20_-05-05_aroundunload_epo_"
            "goodheadpos_mixed_source_"
            + str(source_idx)
            + ".npy",
            tfr_lab,
        )


# =============================================================================
# Main
# =============================================================================

def main():
    if len(sys.argv) < 2:
        raise ValueError(
            "Please provide a subject ID, e.g. "
            "python lcmv_time_frequency_analysis.py BM_C01"
        )

    subject = sys.argv[1]

    print("=" * 72)
    print(f"Running post-LCMV time-frequency analysis for {subject}")
    print("=" * 72)

    # Alpha-beta: adaptive superlet + FOOOF periodic correction.
    alpha_beta_data = load_lcmv_source_data(subject, "alpha_beta")
    run_alpha_beta_analysis(alpha_beta_data, subject)

    # High-gamma: multitaper power.
    gamma_data = load_lcmv_source_data(subject, "gamma")
    run_gamma_analysis(gamma_data, subject)

    print("=" * 72)
    print(f"Finished time-frequency analysis for {subject}")
    print("=" * 72)


if __name__ == "__main__":
    main()
