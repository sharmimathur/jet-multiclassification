import tensorflow.keras as keras
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import uproot

# 27 jet-level features
features = ['fj_jetNTracks',
            'fj_nSV',
            'fj_tau0_trackEtaRel_0',
            'fj_tau0_trackEtaRel_1',
            'fj_tau0_trackEtaRel_2',
            'fj_tau1_trackEtaRel_0',
            'fj_tau1_trackEtaRel_1',
            'fj_tau1_trackEtaRel_2',
            'fj_tau_flightDistance2dSig_0',
            'fj_tau_flightDistance2dSig_1',
            'fj_tau_vertexDeltaR_0',
            'fj_tau_vertexEnergyRatio_0',
            'fj_tau_vertexEnergyRatio_1',
            'fj_tau_vertexMass_0',
            'fj_tau_vertexMass_1',
            'fj_trackSip2dSigAboveBottom_0',
            'fj_trackSip2dSigAboveBottom_1',
            'fj_trackSip2dSigAboveCharm_0',
            'fj_trackSipdSig_0',
            'fj_trackSipdSig_0_0',
            'fj_trackSipdSig_0_1',
            'fj_trackSipdSig_1',
            'fj_trackSipdSig_1_0',
            'fj_trackSipdSig_1_1',
            'fj_trackSipdSig_2',
            'fj_trackSipdSig_3',
            'fj_z_ratio']

# spectators to define mass/pT window
spectators = ['fj_sdmass',
              'fj_pt']

# 2 labels: QCD or Hbb (we'll reduce the following labels)
labels =  ['label_QCD_b',
           'label_QCD_bb',
           'label_QCD_c', 
           'label_QCD_cc', 
           'label_QCD_others',
           'sample_isQCD',
           'label_H_bb']

nfeatures = len(features)
nspectators = len(spectators)
nlabels = 2


def get_features_labels(file_name, remove_mass_pt_window=True, entrystop=None):
    # load file
    root_file = uproot.open(file_name)
    tree = root_file['deepntuplizer/tree']
    feature_array = tree.arrays(branches=features, 
                                entrystop=entrystop,
                                namedecode='utf-8')
    spec_array = tree.arrays(branches=spectators, 
                             entrystop=entrystop,
                             namedecode='utf-8')
    label_array_all = tree.arrays(branches=labels, 
                                  entrystop=entrystop,
                                  namedecode='utf-8')

    feature_array = np.stack([feature_array[feat] for feat in features],axis=1)
    spec_array = np.stack([spec_array[spec] for spec in spectators],axis=1)
    
    njets = feature_array.shape[0]
    
    label_array = np.zeros((njets,nlabels))
    label_array[:,0] = label_array_all['sample_isQCD'] * (label_array_all['label_QCD_b'] + \
                                                          label_array_all['label_QCD_bb'] + \
                                                          label_array_all['label_QCD_c'] + \
                                                          label_array_all['label_QCD_cc'] + \
                                                          label_array_all['label_QCD_others'])
    label_array[:,1] = label_array_all['label_H_bb']

    # remove samples outside mass/pT window
    if remove_mass_pt_window:
        feature_array = feature_array[(spec_array[:,0] > 40) & (spec_array[:,0] < 200) & (spec_array[:,1] > 300) & (spec_array[:,1] < 2000)]
        label_array = label_array[(spec_array[:,0] > 40) & (spec_array[:,0] < 200) & (spec_array[:,1] > 300) & (spec_array[:,1] < 2000)]
        spec_array = spec_array[(spec_array[:,0] > 40) & (spec_array[:,0] < 200) & (spec_array[:,1] > 300) & (spec_array[:,1] < 2000)]
    
    # remove unlabeled data
    feature_array = feature_array[np.sum(label_array,axis=1)==1]
    spec_array = spec_array[np.sum(label_array,axis=1)==1]
    label_array = label_array[np.sum(label_array,axis=1)==1]

    return feature_array, label_array, spec_array


# load training file
feature_array, label_array, spec_array = get_features_labels('/teams/DSC180A_FA20_A00/b06particlephysics/train/ntuple_merged_10.root', 
                                                             remove_mass_pt_window=False,
                                                             entrystop=20000)