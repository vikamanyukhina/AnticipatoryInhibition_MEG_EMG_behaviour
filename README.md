# MEG analysis pipeline

The scripts provided in this repository implement the main analyses of the MEG study, covering raw-data preprocessing, behavioural and EMG analyses, source-level analyses, brain–behaviour correlations, and burst-related analyses. The repository also includes scripts implementing GAM-based analyses to further investigate relationships between neural activity, EMG inhibition, and motor behaviour.

**Manyukhina, V., Barlaam, F., Vergne, J., Bain, A., Abdoun, O., Daligault, S., Delpuech, C., Jerbi, K., Sonié, S., Bonnefond, M., & Schmitz, C.**

*SMA beta bursts reveal distinct mechanisms of anticipatory motor control in children.*

**doi:** https://doi.org/10.64898/2026.03.17.712353

The scripts are also based on the initial study but are not identical to those used in it:

**Manyukhina, V., Abdoun, O., Di Rienzo, F., Barlaam, F., Daligault, S., Delpuech, C., ... & Schmitz, C. (2026).** Beta bursts in SMA mediate anticipatory muscle inhibition. *Cerebral Cortex, 36*(6), bhag054.

**doi:** https://doi.org/10.1093/cercor/bhag054


## Pipeline

### 1. MEG preprocessing

- `1-step1_MEG_raw_preprocessing.py` — combines MEG files and performs initial preprocessing.
- `2-step2_MEG_ICA_preprocessing.py` — ICA-based artifact removal.

### 2. Behavioural and EMG analysis

- `3-imposed_emg_and_rotation.py` — analysis of elbow rotation in the imposed-unloading condition.
- `4-voluntary_rotation_peak_estimation.py` — estimation of peak elbow rotation during voluntary unloading.
- `5a-manual_annotate_inhibitions_from_emg.py` — manual annotation of EMG inhibition.
- `5b-emg_normalize.py` — EMG normalization for automatic inhibition detection.
- `6-auto_inh_detection_and_report.py` — automatic detection and reporting of EMG inhibition.
- `7-headpos_and_badtrials.py` — head-position estimation and bad-trial identification based on manual cleaning.
- `8-maxfilter_to_indheadpos.py` — MaxFilter alignment to the optimal individual head position.
- `9b-correlation_matrix_behav_children.py` — correlations between behavioural/EMG measures and individual characteristics (e.g., age and body parameters).

### 3. Source-level analysis

- `9a-lcmv_source_localization_alphabeta_and_gamma.py` — LCMV source localization for alpha/beta and gamma activity.
- `10a-lcmv_timefrequency_alphabeta_and_gamma.py` — source-level time–frequency analysis for alpha/beta and gamma.
- `11a-corr_lcmv_alphabeta_vs_inhampl.py` — source-level correlation between alpha/beta activity and EMG inhibition amplitude. The analogous gamma analysis was also performed but is not shown separately.
- `12a-morph_lcmv_corr_alphabeta_fsaverage.py` — morphing source-level correlations to `fsaverage`.
- `13a-cluster_test_corr_alphabeta_rh_motor.py` — cluster-based analysis of alpha/beta–inhibition correlations in the right motor cortex.
- `14a-cluster_test_corr_gamma_r-cerebellum.py` — example cluster-based analysis of gamma–inhibition correlations in the right cerebellum.

### 4. Burst and connectivity analysis

- `10b-burst_detection_and_contrasts.py` — Beta-burst detection and burst-related analyses in the SMA, including Superlet time–frequency decomposition, aperiodic activity estimation, burst classification, burst-centred power analyses, and cluster-based permutation tests.
- `10c-granger_2hemi_to_sma_duringbursts.py` — Granger causality analysis on selected beta-burst trials.

Beta bursts were extracted using the pipeline of Szul et al. (2023) within the SMA label identified from the gamma–EMG inhibition correlation. The burst-detection script implements the extraction and subsequent analyses reported here.

Cluster-based analyses are shown for representative frequency bands/regions and are not repeated when the same analysis was performed analogously elsewhere.

### 5. GAM-based analyses

The following R scripts implement generalized additive models (GAMs) to investigate relationships between EMG inhibition, neural activity, and motor behaviour:

- `behav_model_elbrot_vs_inh_to_github.R` — uses GAMs to test which EMG inhibition parameters predict elbow rotation (forearm stabilization).
- `SMAlabel_gamma_beta_inh_to_github.R` — uses GAMs to validate findings from the correlation analyses on relationships between right SMA/IFC high-gamma power, beta power, and EMG inhibition, and to investigate how these relationships vary with EMG inhibition timing.
- `SMAlabel_gammaburst_behav_td_children_to_github.R` — uses GAMs to investigate whether high-gamma power suppression at two burst times is associated with EMG inhibition parameters and elbow rotation.


## Automatic EMG inhibition detection

The supplementary figures illustrate examples of Biceps brachii inhibition events identified by the automatic detection algorithm and their correspondence with manual annotations.

**Figure S1.** Biceps brachii inhibition events identified by both manual and automatic detection methods.

**Figure S2.** Biceps brachii inhibition events identified by the automatic detection method but not detected manually.

**Figure S3.** Trials 1–2: Examples of long Biceps brachii inhibition events identified manually but missed by the automatic detection method. Trial 3: An example of a long inhibition event successfully detected by both methods.

## Software and library versions

The analyses were performed using the following software and library versions:

- Python 3.7
- MNE-Python 1.7.1
- SciPy 1.13.1
- scikit-learn 1.5.0
- R 4.5.2
- `mgcv` 1.9.4
- FreeSurfer 6.0.0 (`recon-all` pipeline)
