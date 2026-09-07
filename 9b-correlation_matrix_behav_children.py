#!/usr/bin/env python3


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, wilcoxon
from mne.stats import fdr_correction


# ============================================================
# Calculate correlation matrix and significance annotations
# ============================================================

def corr_sig(df):
    cols = df.columns
    n = len(cols)
    corr = np.zeros((n, n))
    annot = np.empty((n, n), dtype=object)

    for i in range(n):
        for j in range(n):
            x = df[cols[i]].to_numpy()
            y = df[cols[j]].to_numpy()

            # Pairwise removal of NaNs
            valid = ~np.isnan(x) & ~np.isnan(y)
            xv = x[valid]
            yv = y[valid]

            # Calculate Spearman correlation
            r, p = spearmanr(xv, yv)
            corr[i, j] = r

            # Add significance/trend annotation
            if np.isnan(r):
                annot[i, j] = ""
            else:
                star = "*" if p < 0.05 else "~" if p < 0.15 else ""
                annot[i, j] = f"{r:.2f}{star}"

    return pd.DataFrame(corr, index=cols, columns=cols), annot


# ============================================================
# Calculate FDR-corrected significance annotations
# ============================================================

def get_annots(data):
    # Flatten correlations across subjects
    corr_flattened = data.reshape(data.shape[0], -1)
    n_voxels = corr_flattened.shape[1]
    p_vals = np.zeros(n_voxels)

    # Calculate p-values for each correlation
    for i in range(n_voxels):
        x = corr_flattened[:, i]
        _, p_vals[i] = wilcoxon(
            x,
            zero_method='wilcox',
            alternative='two-sided'
        )

    # Reshape p-values back to matrix
    p_vals_matrix = p_vals.reshape(data.shape[1], data.shape[2])

    # FDR correction on unique pairs only
    tri_inds = np.triu_indices_from(p_vals_matrix, k=1)
    p_unique = p_vals_matrix[tri_inds]

    reject, p_fdr_unique = fdr_correction(
        p_unique,
        alpha=0.05,
        method='indep'
    )

    # Rebuild full corrected matrix
    p_vals_matrix_fdr = np.ones_like(p_vals_matrix)
    p_vals_matrix_fdr[tri_inds] = p_fdr_unique
    p_vals_matrix_fdr[(tri_inds[1], tri_inds[0])] = p_fdr_unique

    # Exclude self-correlations
    np.fill_diagonal(p_vals_matrix_fdr, np.nan)
    p_vals_matrix = p_vals_matrix_fdr

    # Mean correlations for display
    annot_cors = data.mean(0)

    # Build annotation labels
    annot = np.full(p_vals_matrix.shape, '', dtype=object)
    annot[p_vals_matrix < 0.05] = '*'
    annot[(p_vals_matrix >= 0.05) & (p_vals_matrix < 0.15)] = '~'
    annot = np.array(np.round(annot_cors, 2), dtype=str) + annot

    return annot


# ============================================================
# Subjects and file paths
# ============================================================

SUBJECTS = (['BM_C0' + str(ii) for ii in np.arange(1, 10)] + ['BM_C' + str(ii) for ii in np.arange(10, 25)])

PATHto = 'pathto'
PATHfrom = 'pathfrom'


# ============================================================
# Load behavioral and participant information
# ============================================================

med_rot = pd.read_csv(PATHfrom + '/med_elbow_rot_imposed_children.csv')

xlsx_path = PATHfrom + '/MEG_Children_data.xlsx'
df_descr = pd.read_excel(xlsx_path, sheet_name="MEG ALL NT", skiprows=1)

sel_labels = [
    'Code sujet', 'Age (mois)', 'Taille', 'Poids', 'IMC',
    'Poids utilisé', 'Tête', "Longueur de l'avant-bras",
    'Latéralité 2\n%', 'IRP', 'IMT', 'IVT', 'Total',
    'Dext. Manuelle', 'Maîtrise balle', 'Equilibre', 'Total.1'
]

df_descr = df_descr[sel_labels]


# ============================================================
# Initialize subject-level measures
# ============================================================

cor_inhs_matrices = []

inh_ampls = []
inh_times = []
inh_onsets = []
inh_durs = []

rots = []
decls = []

ages = []
taille = []
poids = []
imc = []
loads = []
tetes = []
bras = []
lats = []
totals = []
dexts = []
balles = []
equils = []
totals1 = []


# ============================================================
# Process each subject
# ============================================================

for subj in SUBJECTS:

    # Load subject-specific median elbow rotation
    med_abs = med_rot[med_rot['Subjects'] == subj]['Median elbow rot ampl'].iloc[0]

    # Select participant description
    df_subj = df_descr[df_descr['Code sujet'] == subj]

    # Load trial selection information
    n_epos = np.load(PATHto + subj + '_epo_n_inh_headpos_auto.npy')
    n_epo_man = np.load(PATHto + subj + '_epo_n_inh_headpos_man.npy')
    rot_epo = np.load(PATHto + subj + '_epo_n_rot_headpos.npy')

    # Load automatic inhibition measures
    inh = np.load(
        PATHto + subj +
        '_pampl_peakinh_unloading_and_inh_ontime_auto15_20ms_headpos1cm.npy'
    )

    inh_time = np.load(
        PATHto + subj +
        '_time_peakinh_unloading_and_inh_ontime_auto15_20ms_headpos1cm.npy'
    )

    inh_onset = np.load(
        PATHto + subj +
        '_time_onsetinh_unloading_and_inh_ontime_auto15_20ms_headpos1cm.npy'
    )

    inh_dur = np.load(
        PATHto + subj +
        '_duration_unloading_and_inh_ontime_auto15_20ms_headpos1cm.npy'
    ) * 600

    df_auto = pd.DataFrame({
        'Epo': n_epos,
        'Ampl': -inh,
        'PeakTime': inh_time,
        'Len': inh_dur,
        'Onset': inh_onset
    })

    # Load manual inhibition measures
    inh_dur_man = np.load(
        PATHto + subj +
        '_duration_unloading_and_inh_ontime_manual_headpos1cm.npy'
    )

    inh_onset_man = np.load(
        PATHto + subj +
        '_time_onsetinh_unloading_and_inh_ontime_manual_headpos1cm.npy'
    )

    df_man = pd.DataFrame({
        'Epo': n_epo_man,
        'Len': inh_dur_man,
        'Onset': inh_onset_man
    })

    # Load rotation measures
    rot_peak = np.abs(
        np.load(
            PATHto + subj +
            '_elbow_peak_unloading_and_rot_headpos1cm_new.npy'
        ) / med_abs
    )

    rot_decl = np.load(
        PATHto + subj +
        '_elbow_decl_unloading_and_rot_headpos1cm_new.npy'
    ) / med_abs

    df_rot = pd.DataFrame({
        'Epo': rot_epo,
        'Peak': rot_peak,
        'Decl': rot_decl
    })

    # Find trials with all required measures
    epo_both_early = np.intersect1d(n_epo_man, n_epos)
    epo_both_early = np.intersect1d(epo_both_early, rot_epo)

    # Select automatic inhibition measures for overlapping trials
    epo_ovl = df_auto[df_auto['Epo'].isin(epo_both_early)]['Epo'].values
    inh_ovl = df_auto[df_auto['Epo'].isin(epo_both_early)]['Ampl'].values
    inh_time_ovl = df_auto[df_auto['Epo'].isin(epo_both_early)]['PeakTime'].values
    inh_dur_ovl = df_auto[df_auto['Epo'].isin(epo_both_early)]['Len'].values
    inh_onset_ovl = df_auto[df_auto['Epo'].isin(epo_both_early)]['Onset'].values

    # Select rotation measures for overlapping trials
    rot_peak_ovl = df_rot[df_rot['Epo'].isin(epo_both_early)]['Peak'].values
    rot_decl_ovl = df_rot[df_rot['Epo'].isin(epo_both_early)]['Decl'].values

    # Select manual inhibition measures for overlapping trials
    inh_dur_man_ovl = df_man[df_man['Epo'].isin(epo_both_early)]['Len'].values
    inh_onset_man_ovl = df_man[df_man['Epo'].isin(epo_both_early)]['Onset'].values

    # Combine trial-level measures
    df_all = pd.DataFrame({
        'Trial': epo_ovl,
        'Inhibition': inh_ovl,
        'Peak_Time': inh_time_ovl,
        'Duration': inh_dur_man_ovl,
        'Onset_Time': inh_onset_man_ovl,
        'Peak_Rot': rot_peak_ovl,
        'Rot_Decl': rot_decl_ovl
    })

    # Calculate within-subject correlation matrix
    corr, annot = corr_sig(df_all)
    cor_inhs_matrices.append(corr)

    # Calculate subject-level mean measures
    inh_ampls.append(inh_ovl.mean())
    inh_times.append(inh_time_ovl.mean())
    inh_onsets.append(inh_onset_man_ovl.mean())
    inh_durs.append(inh_dur_man_ovl.mean())
    rots.append(rot_peak_ovl.mean())
    decls.append(rot_decl_ovl.mean())

    # Extract participant characteristics
    ages.append(df_subj['Age (mois)'].values[0])
    taille.append(df_subj['Taille'].values[0])
    poids.append(df_subj['Poids'].values[0])
    imc.append(df_subj['IMC'].values[0])
    loads.append(df_subj['Poids utilisé'].values[0])
    tetes.append(df_subj['Tête'].values[0])
    bras.append(df_subj["Longueur de l'avant-bras"].values[0])
    lats.append(df_subj['Latéralité 2\n%'].values[0])
    totals.append(df_subj['Total'].values[0])
    dexts.append(df_subj['Dext. Manuelle'].values[0])
    balles.append(df_subj['Maîtrise balle'].values[0])
    equils.append(df_subj['Equilibre'].values[0])
    totals1.append(df_subj['Total.1'].values[0])


# ============================================================
# Average correlation matrices across subjects
# ============================================================

cor_inhs_matrices = np.array(cor_inhs_matrices)
cor_inhs_matrices_av = cor_inhs_matrices.mean(0)


# ============================================================
# Create subject-level dataframe
# ============================================================

df_subjs = pd.DataFrame({
    'PeakInh': inh_ampls,
    'PeakTime': inh_times,
    'Duration': inh_durs,
    'OnsetTime': inh_onsets,
    'PeakRot': rots,
    'RotDecl': decls,
    'Age': ages,
    'Height': taille,
    'Weight': poids,
    'IMC': imc,
    'Load_weight': loads,
    'Head': tetes,
    'Forearm_Len': bras,
    'Laterality': lats,
    'TotalIQ': totals,
    'Dexterity': dexts,
    'Ball': balles,
    'Equilibre': equils,
    'Total1': totals1
})


# ============================================================
# Correlations with age
# ============================================================

df_subjs = pd.DataFrame({
    'PeakInh': inh_ampls,
    'PeakTime': inh_times,
    'Duration': inh_durs,
    'OnsetTime': inh_onsets,
    'PeakRot': rots,
    'RotDecl': decls,
    'Age': ages
})

vars_to_test = df_subjs.columns.drop('Age')

r_vals = []
p_vals = []

# Calculate Spearman correlations with age
for var in vars_to_test:
    r, p = spearmanr(df_subjs['Age'], df_subjs[var])
    r_vals.append(r)
    p_vals.append(p)


# ============================================================
# FDR correction for age correlations
# ============================================================

reject, p_vals_fdr = fdr_correction(
    p_vals,
    0.05,
    'indep'
)[1]

results = pd.DataFrame({
    'Variable': vars_to_test,
    'r': r_vals,
    'p_uncorrected': p_vals,
    'p_FDR': p_vals_fdr,
    'significant_FDR': reject
})

print(results)


# ============================================================
# Correlation matrix across subjects
# ============================================================

corr_all, annot_all = corr_sig(df_subjs)

fig = plt.figure(figsize=(20, 10))

ax = sns.heatmap(
    corr_all,
    annot=annot_all,
    fmt='',
    cmap='coolwarm',
    center=0,
    vmin=-0.5,
    vmax=0.5,
    xticklabels=df_subjs.columns.values,
    yticklabels=df_subjs.columns.values,
    cbar_kws={'label': "Spearman's R, group mean"},
    annot_kws={'size': 12}
)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Adjust existing seaborn colorbar
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=14)
cbar.set_label("Spearman's R", fontsize=16)

plt.title('Correlation Matrix', fontsize=20)
plt.show()

# Save correlation matrix
plt.savefig(
    PATHto + '/paper_figs_children/' +
    'Correlation_matrix_oversubj.png',
    bbox_inches='tight',
    pad_inches=1
)

plt.savefig(
    PATHto + '/paper_figs_children/' +
    'Correlation_matrix_oversubj.svg',
    bbox_inches='tight',
    pad_inches=1
)


# ============================================================
# Correlation matrix across trials
# ============================================================

fig = plt.figure(figsize=(8, 6))

annot = get_annots(cor_inhs_matrices)

ax = sns.heatmap(
    cor_inhs_matrices_av,
    annot=annot,
    fmt='',
    cmap='coolwarm',
    center=0,
    vmin=-0.5,
    vmax=0.5,
    xticklabels=corr.columns.values,
    yticklabels=corr.columns.values,
    cbar_kws={'label': "Spearman's R, group mean"},
    annot_kws={'size': 12}
)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Adjust existing seaborn colorbar
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=14)
cbar.set_label("Spearman's R, group mean", fontsize=16)

plt.title('Correlation Matrix, group mean', fontsize=20)
plt.show()

# Save correlation matrix
plt.savefig(
    PATHto + '/paper_figs_children/' +
    'Correlation_matrix_bysubj.png',
    bbox_inches='tight',
    pad_inches=1
)

plt.savefig(
    PATHto + '/paper_figs_children/' +
    'Correlation_matrix_bysubj.svg',
    bbox_inches='tight',
    pad_inches=1
)