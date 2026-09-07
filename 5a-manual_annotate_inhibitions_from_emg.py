#!/usr/bin/env python3

import mne
import numpy as np
import matplotlib.pyplot as plt
from mne import io
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
           [0.9697, 0.8481380952, 0.147452381,], [0.9625857143, 0.8705142857, 0.1309],
           [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 0.0948380952],
           [0.9661, 0.9514428571, 0.0755333333], [0.9763, 0.9831, 0.0538]]

parula_map = LinearSegmentedColormap.from_list('parula', cm_data)


PATHfrom = 'pathfrom'
PATHto = 'pathto'
exp = 'naturel'

subject = 'BM_C01'


# Load preprocessed data
raw_fname = PATHfrom + subject + '/ica_notch/' + subject + '_' + exp + '_indpos_ica-ref_thr2_raw.fif'
raw = mne.io.Raw(raw_fname, preload=True)


# Apply notch filter and select EMG/kinematic channels
raw_orig = io.Raw(raw_fname, preload=True)
raw = raw_orig.copy().notch_filter([50, 100, 150])
raw_6ch = raw.pick_channels(['BicG', 'TriG', 'SupG', 'UADC002-2800', 'UADC001-2800'])


# Demean EMG channels
raw_data = raw_6ch.get_data()
raw_data_demean = []

for i in range(5):
    if 'UAD' not in raw_6ch.ch_names[i]:
        raw_data_demean.append(raw_data[i] - np.mean(raw_data[i]))
    else:
        raw_data_demean.append(raw_data[i])

raw_6ch = mne.io.RawArray(np.array(raw_data_demean), raw_6ch.info)


# Band-pass filter and rectify EMG channels
raw_6ch = raw_6ch.filter(25, 150)
raw_data = raw_6ch.get_data()
raw_data_abs = []

for i in range(5):
    if 'UAD' not in raw_6ch.ch_names[i]:
        raw_data_abs.append(np.abs(raw_data[i]))
    else:
        raw_data_abs.append(raw_data[i])

raw_6ch = mne.io.RawArray(np.array(raw_data_abs), raw_6ch.info)


# Load existing annotations or create annotation labels
try:
    annot = mne.read_annotations(PATHto + '/' + subject + '_' + exp + '_indpos_ica_raw_annots_my.csv')
    raw_6ch.set_annotations(annot)
except:
    annot = mne.Annotations(
        [0, 0, 0], [0, 0, 0],
        ['_BicG_naturel_inhibition', '_SupG_naturel_inhibition',
         '_TriG_naturel_cocontraction'],
        orig_time=raw.info['meas_date']
    )
    raw_6ch.set_annotations(annot + raw.annotations)


# Manually annotate muscle inhibition
raw_6ch.plot(block=True)


# Save annotations
save_annot = raw_6ch.annotations
save_annot.save(PATHto + subject + '_' + exp + '_indpos_ica_raw_annots_my.csv', overwrite=True)


# Check timing of inhibition annotations relative to action onset
events_inh_other = mne.events_from_annotations(
    raw_6ch, event_id={'_BicG_naturel_inhibition': 1})[0]
events_rot = mne.events_from_annotations(
    raw_6ch, event_id={'naturel_action_start': 1})[0]

sfreq = 600

inh_time_all = []
for i in range(len(events_inh_other)):
    ttt = np.argmin(np.abs(events_rot[:, 0] - events_inh_other[i][0]))
    inh_time_all.append((events_inh_other[i][0] - events_rot[ttt][0]) / sfreq)

inh_time_all = np.array(inh_time_all)

plt.hist(inh_time_all, bins=200)


# Reload annotations and verify they were saved correctly
load_annot = mne.read_annotations(PATHto + subject + '_' + exp + '_indpos_ica_raw_annots_my.csv')
raw_6ch.set_annotations(load_annot)
raw_6ch.plot()