#!/usr/bin/env python3

import mne
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mne.preprocessing import annotate_movement, compute_average_dev_head_t
from mne.report import Report
from matplotlib.colors import LinearSegmentedColormap


# Custom colormap
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
 [0.5708571429, 0.7485190476, 0.4493904762], [0.609852381, 0.7473142857, 0.4336857143],
 [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333],
 [0.7184095238, 0.7411333333, 0.3904761905], [0.7524857143, 0.7384, 0.3768142857],
 [0.7858428571, 0.7355666667, 0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762],
 [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217],
 [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 0.2886428571],
 [0.9738952381, 0.7313952381, 0.266647619], [0.9937714286, 0.7454571429, 0.240347619],
 [0.9990428571, 0.7653142857, 0.2164142857], [0.9955333333, 0.7860571429, 0.196652381],
 [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857],
 [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309],
 [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 0.0948380952],
 [0.9661, 0.9514428571, 0.0755333333], [0.9763, 0.9831, 0.0538]]

parula_map = LinearSegmentedColormap.from_list('parula', cm_data)


# Paths and experiment
subj_type = 'Children/Control'
PATHfrom = 'pathfrom'
PATHto = 'pathto' + subj_type
exp = 'naturel'

SUBJECTS = ['BM_C0' + str(ii) for ii in np.arange(1, 10)] + ['BM_C' + str(ii) for ii in np.arange(10, 25)]

n_epo_subj = []
n_epo_clean_subj = []
n_epo_1cm_subj = []
n_epo_inh_ontime_subj = []
n_epo_inh_ontime_manual_subj = []

ori_poss = []
new_av_poss = []
inters_perc = []
subjs = []
onsets = []

report_fname = PATHfrom + 'Children_control_head_positions.html'
report = Report()


for subj in SUBJECTS:

    subjs.append(subj)

    # Load raw data and identify unloading events
    raw_fname = PATHfrom + '/' + subj + '_' + exp + '_indpos_ica-ref_thr2_raw.fif'
    raw = mne.io.Raw(raw_fname, preload=True)
    events_true = mne.events_from_annotations(raw, event_id={'naturel_action_start': 1})[0]

    # Count all unloading trials and trials without existing annotations marked as bad
    allepochs_nat_true = mne.Epochs(raw, events_true, tmin=-0.5, tmax=0.1, baseline=None,
                                    proj=False, preload=True, reject_by_annotation=False)
    n_epo_subj.append(len(allepochs_nat_true))
    np.save(PATHfrom + subj + '_epo_selection_unloading.npy', allepochs_nat_true.selection)

    allepochs_nat_true = mne.Epochs(raw, events_true, tmin=-0.5, tmax=0.1, baseline=None,
                                    proj=False, preload=True, reject_by_annotation=True)
    n_epo_clean_subj.append(len(allepochs_nat_true))
    np.save(PATHfrom + subj + '_epo_selection_unloading_and_no_emg_bad.npy', allepochs_nat_true.selection)

    # Estimate continuous head position from the CTF cHPI coils
    chpi_locs = mne.chpi.extract_chpi_locs_ctf(raw)
    head_pos = mne.chpi.compute_head_pos(raw.info, chpi_locs)

    original_head_dev_t = mne.transforms.invert_transform(raw.info['dev_head_t'])
    ori_poss.append(original_head_dev_t['trans'][:3, 3])

    # Calculate the average head position across the recording
    average_head_dev_t = mne.transforms.invert_transform(compute_average_dev_head_t(raw, head_pos))
    av_pos = average_head_dev_t['trans'][:3, 3]

    fig = mne.viz.plot_head_positions(head_pos)
    for ax, val, val_ori in zip(fig.axes[::2], average_head_dev_t['trans'][:3, 3], original_head_dev_t['trans'][:3, 3]):
        ax.axhline(1000 * val, color='r')
        ax.axhline(1000 * val_ori, color='g')

    report.add_figure(fig, title='Head position', section=subj)
    plt.close('all')

    # Mark periods with head-position changes exceeding 1 cm
    mean_distance_limit = 0.01  # 1 cm, in meters
    annotation_movement, hpi_disp = annotate_movement(
        raw, head_pos, mean_distance_limit=mean_distance_limit, use_dev_head_trans='average'
    )
    annotation_movement.save(PATHfrom + subj + '_head_pos_deflection_1cm_bad.csv', overwrite=True)

    # Combine head-movement annotations with the original annotations
    raw.set_annotations(annotation_movement + raw.annotations)

    # Keep only trials without EMG or >1 cm head-movement artifacts
    allepochs_nat_true = mne.Epochs(raw, events_true, tmin=-0.5, tmax=0.1, baseline=None,
                                    proj=False, preload=True, reject_by_annotation=True)
    n_epo_1cm_subj.append(len(allepochs_nat_true))
    np.save(PATHfrom + subj + '_epo_selection_unloading_and_no_emg_bad_and_no_headmove1cm.npy', allepochs_nat_true.selection)

    # Remove manually identified elbow-rotation artifacts
    bads_dict = {
        'BM_C01': [6, 10, 46],
        'BM_C03': [25],
        'BM_C04': [15, 36],
        'BM_C05': [2, 3, 7, 11, 43],
        'BM_C07': [2],
        'BM_C10': [32, 38, 53],
        'BM_C12': [5, 16, 35, 40, 48],
        'BM_C15': [30, 46],
        'BM_C18': [32],
        'BM_C19': [4, 18, 21, 24, 26, 31, 48, 49, 52],
        'BM_C20': [10, 24],
        'BM_C21': [23, 47],
        'BM_C23': [30],
        'BM_C24': [23]
    }

    n_epo_rot = np.load(PATHfrom + subj + '_epo_unloading_with_elbrot.npy')
    bads = bads_dict.get(subj, [])
    sel = np.ones(len(n_epo_rot), dtype=bool)
    sel[bads] = False
    n_epo_rot = n_epo_rot[sel]

    rotstab = np.load(PATHfrom + subj + '_stab_elbowrot_basecor_epos.npy')[sel]
    rotpeak = np.load(PATHfrom + subj + '_peak_elbowrot_basecor_epos.npy')[sel]
    rotdecl = np.load(PATHfrom + subj + '_decr_elbowrot_basecor_epos.npy')[sel]
    rotvel = np.load(PATHfrom + subj + '_vel_elbowrotmax_nobase_epos.npy')[sel]

    df_rot = pd.DataFrame({'Epo': n_epo_rot, 'Peak': rotpeak, 'Decl': rotdecl, 'Vel': rotvel, 'Stab': rotstab})

    # Load manually identified inhibition trials
    n_epo_inh_manual = np.load(PATHfrom + subj + '_epo_selection_unloading_and_inh_ontime_manual.npy')
    inh_time_manual = np.load(PATHfrom + subj + '_epo_inh_ontime_time_inhonset_manual_unload.npy')
    inh_len_manual = np.load(PATHfrom + subj + '_epo_inh_ontime_duration_manual_unload.npy')
    onsets.append(inh_time_manual.mean())

    if len(n_epo_inh_manual) != len(inh_time_manual):
        raise ValueError

    df_man = pd.DataFrame({'Time': inh_time_manual, 'Epo': n_epo_inh_manual, 'Len': inh_len_manual})

    # Load automatically detected inhibition trials and their characteristics
    n_epo_inh_auto = np.load(PATHfrom + subj + '_epo_inh_ontime_n_epos_auto15_20ms_unload.npy')
    inh_time_auto = np.load(PATHfrom + subj + '_epo_inh_ontime_time_inhpeak_auto15_20ms_unload.npy')
    inh_time_onset_auto = np.load(PATHfrom + subj + '_epo_inh_ontime_time_inhonset_auto15_20ms_unload.npy')
    dur = np.load(PATHfrom + subj + '_epo_inh_ontime_duration_auto15_20ms_unload.npy')
    pampl = np.load(PATHfrom + subj + '_epo_inh_ontime_peak_ampl_auto15_20ms_unload.npy')
    wampl = np.load(PATHfrom + subj + '_epo_inh_ontime_weighted_ampl_auto15_20ms_unload.npy')
    sampl = np.load(PATHfrom + subj + '_epo_inh_ontime_sum_ampl_auto15_20ms_unload.npy')

    if len(n_epo_inh_auto) != len(inh_time_auto):
        raise ValueError

    df_auto = pd.DataFrame({'Time': inh_time_auto, 'Time_onset': inh_time_onset_auto, 'Epo': n_epo_inh_auto,
                            'pAmpl': pampl, 'wAmpl': wampl, 'sAmpl': sampl, 'Len': dur})

    # Convert original epoch indices to indices of the final clean dataset
    n_inh_auto = []
    n_inh_man = []
    n_rot = []

    for e in range(len(allepochs_nat_true.selection)):
        if allepochs_nat_true.selection[e] in n_epo_inh_auto:
            n_inh_auto.append(e)
        if allepochs_nat_true.selection[e] in n_epo_inh_manual:
            n_inh_man.append(e)
        if allepochs_nat_true.selection[e] in n_epo_rot:
            n_rot.append(e)

    np.save(PATHfrom + subj + '_epo_n_inh_headpos_auto.npy', np.unique(n_inh_auto))
    np.save(PATHfrom + subj + '_epo_n_inh_headpos_man.npy', np.unique(n_inh_man))
    np.save(PATHfrom + subj + '_epo_n_rot_headpos.npy', np.unique(n_rot))

    # Keep only inhibition and rotation trials that survived the 1 cm head-movement/EMG cleaning
    int_auto = np.intersect1d(n_epo_inh_auto, allepochs_nat_true.selection)
    int_man = np.intersect1d(n_epo_inh_manual, allepochs_nat_true.selection)
    int_rot = np.intersect1d(n_epo_rot, allepochs_nat_true.selection)

    int_time_auto = df_auto[df_auto['Epo'].isin(int_auto)]['Time'].values
    int_time_onset_auto = df_auto[df_auto['Epo'].isin(int_auto)]['Time_onset'].values
    pampl_auto = df_auto[df_auto['Epo'].isin(int_auto)]['pAmpl'].values
    wampl_auto = df_auto[df_auto['Epo'].isin(int_auto)]['wAmpl'].values
    sampl_auto = df_auto[df_auto['Epo'].isin(int_auto)]['sAmpl'].values
    dur_auto = df_auto[df_auto['Epo'].isin(int_auto)]['Len'].values

    int_time_man = df_man[df_man['Epo'].isin(int_man)]['Time'].values
    int_len_man = df_man[df_man['Epo'].isin(int_man)]['Len'].values

    int_rot_peak = df_rot[df_rot['Epo'].isin(int_rot)]['Peak'].values
    int_rot_decl = df_rot[df_rot['Epo'].isin(int_rot)]['Decl'].values
    int_rot_vel = df_rot[df_rot['Epo'].isin(int_rot)]['Vel'].values
    int_rot_stab = df_rot[df_rot['Epo'].isin(int_rot)]['Stab'].values

    # Save cleaned trial selections and behavioral/inhibition measures
    np.save(PATHfrom + subj + '_epo_selection_unloading_and_inh_ontime_auto15_20ms_headpos1cm.npy', int_auto)
    np.save(PATHfrom + subj + '_epo_selection_unloading_and_inh_ontime_manual_headpos1cm.npy', int_man)
    np.save(PATHfrom + subj + '_epo_selection_unloading_and_rot_headpos1cm.npy', int_rot)

    np.save(PATHfrom + subj + '_elbow_peak_unloading_and_rot_headpos1cm_new.npy', int_rot_peak)
    np.save(PATHfrom + subj + '_elbow_decl_unloading_and_rot_headpos1cm_new.npy', int_rot_decl)
    np.save(PATHfrom + subj + '_elbow_vel_unloading_and_rot_headpos1cm_new.npy', int_rot_vel)
    np.save(PATHfrom + subj + '_elbow_stab_unloading_and_rot_headpos1cm_new.npy', int_rot_stab)

    np.save(PATHfrom + subj + '_time_onsetinh_unloading_and_inh_ontime_auto15_20ms_headpos1cm.npy', int_time_onset_auto)
    np.save(PATHfrom + subj + '_time_peakinh_unloading_and_inh_ontime_auto15_20ms_headpos1cm.npy', int_time_auto)
    np.save(PATHfrom + subj + '_time_onsetinh_unloading_and_inh_ontime_manual_headpos1cm.npy', int_time_man)
    np.save(PATHfrom + subj + '_duration_unloading_and_inh_ontime_manual_headpos1cm.npy', int_len_man)

    np.save(PATHfrom + subj + '_pampl_peakinh_unloading_and_inh_ontime_auto15_20ms_headpos1cm.npy', pampl_auto)
    np.save(PATHfrom + subj + '_wampl_peakinh_unloading_and_inh_ontime_auto15_20ms_headpos1cm.npy', wampl_auto)
    np.save(PATHfrom + subj + '_sampl_peakinh_unloading_and_inh_ontime_auto15_20ms_headpos1cm.npy', sampl_auto)
    np.save(PATHfrom + subj + '_duration_unloading_and_inh_ontime_auto15_20ms_headpos1cm.npy', dur_auto)

    n_epo_inh_ontime_subj.append(len(int_auto))
    n_epo_inh_ontime_manual_subj.append(len(int_man))

    # Recalculate average head position after applying the movement criterion
    new_av_pos = compute_average_dev_head_t(raw, head_pos)['trans'][:3, 3]
    new_av_poss.append(new_av_pos)

    # Proportion of clean trials detected as inhibition both automatically and manually
    n_both = len(np.intersect1d(int_auto, int_man))
    inters_perc.append(n_both / len(allepochs_nat_true.selection))


# Save summary report and subject-level trial counts
report.save(report_fname, open_browser=True, overwrite=True)

df = pd.DataFrame({
    'Subject': subjs,
    'N was': n_epo_subj,
    'N clean': n_epo_clean_subj,
    'N headstab': n_epo_1cm_subj,
    'N inh on-time auto': n_epo_inh_ontime_subj,
    'N inh on-time manual': n_epo_inh_ontime_manual_subj
})

df.to_csv(PATHfrom + 'Children_n_epo_left_clean_manual_and_autoinh15_ontime_1cm.csv')

np.save(PATHfrom + 'Allsubjs_origin_headpos.npy', np.array(ori_poss))
np.save(PATHfrom + 'Allsubjs_good_av_headpos.npy', np.array(new_av_poss))
