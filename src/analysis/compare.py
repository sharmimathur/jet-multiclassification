import sys
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc


sys.path.insert(0, '../data')

from etl import get_features_labels, clean_array

def compare(config_path):
    with open(config_path) as fh:
        data_cfg = json.load(fh)

        data = get_features_labels(**data_cfg)
        features, labels, specs, tree = data

        label_QCD = labels[:,0]
        label_Hbb = labels[:,1]

        print(sum(label_QCD+label_Hbb)/len(label_QCD+label_Hbb))

    with open('config/compare.json') as f:

        compare_cfg = json.load(f)
        jet_feat = compare_cfg['jet_features']
        track_feat = compare_cfg['track_features']
        sv_feat = compare_cfg['sv_features']

        entrystop = compare_cfg['entrystop']
        namedecode = compare_cfg['namedecode']


        track_features = tree.arrays(branches=track_feat,
                              entrystop=entrystop,
                              namedecode=namedecode)

        jet_features = tree.arrays(branches=jet_feat,
                              entrystop=entrystop,
                              namedecode=namedecode)

        sv_features = tree.arrays(branches=sv_feat,
                              entrystop=entrystop,
                              namedecode=namedecode)

        #jet_features = np.stack([jet_features[feat] for feat in ['fj_pt','fj_sdmass']],axis=1)
        #jet_features = clean_array(jet_features, specs, data_cfg['remove_mass_pt_window'])




        vis_path = 'src/visualizations/'

        # TRACK FEATURES HISTOGRAMS DEPICTING DISCRIMINATORY EFFECT
        # number of tracks
        plt.figure()
        print(type(track_features['track_pt']))
        plt.hist(track_features['track_pt'].counts,weights=label_QCD,bins=np.linspace(0,80,81),density=True,alpha=0.7,label='QCD')
        plt.hist(track_features['track_pt'].counts,weights=label_Hbb,bins=np.linspace(0,80,81),density=True,alpha=0.7,label='H(bb)')
        plt.xlabel('Number of tracks')
        plt.ylabel('Fraction of jets')
        plt.legend()
        plt.savefig(vis_path + 'trackcounts_hist.png')

        # max. relative track pt
        plt.figure()
        plt.hist(track_features['track_pt'].max()/jet_features['fj_pt'],weights=label_QCD,bins=np.linspace(0,0.5,51),density=True,alpha=0.7,label='QCD')
        plt.hist(track_features['track_pt'].max()/jet_features['fj_pt'],weights=label_Hbb,bins=np.linspace(0,0.5,51),density=True,alpha=0.7,label='H(bb)')
        plt.xlabel(r'Maximum relative track $p_{T}$')
        plt.ylabel('Fraction of jets')
        plt.legend()
        plt.savefig(vis_path + 'trackmaxrelpt_hist.png')

        # maximum signed 3D impact paramter value
        plt.figure()
        plt.hist(track_features['trackBTag_Sip3dVal'].max(),weights=label_QCD,bins=np.linspace(-2,40,51),density=True,alpha=0.7,label='QCD')
        plt.hist(track_features['trackBTag_Sip3dVal'].max(),weights=label_Hbb,bins=np.linspace(-2,40,51),density=True,alpha=0.7,label='H(bb)')
        plt.xlabel('Maximum signed 3D impact parameter value')
        plt.ylabel('Fraction of jets')
        plt.legend()
        plt.savefig(vis_path + 'tracksip3val_hist.png')

        # maximum signed 3D impact paramter significance
        plt.figure()
        plt.hist(track_features['trackBTag_Sip3dSig'].max(),weights=label_QCD,bins=np.linspace(-2,200,51),density=True,alpha=0.7,label='QCD')
        plt.hist(track_features['trackBTag_Sip3dSig'].max(),weights=label_Hbb,bins=np.linspace(-2,200,51),density=True,alpha=0.7,label='H(bb)')
        plt.xlabel('Maximum signed 3D impact parameter significance')
        plt.ylabel('Fraction of jets')
        plt.legend()
        plt.savefig(vis_path + 'tracksip3sig_hist.png')

        plt.show()



        # JET FEATURES HISTOGRAMS DEPICTING DISCRIMINATORY EFFECT
        plt.hist(jet_features['fj_pt'],weights=label_QCD,bins=np.linspace(0,4000,101),density=True,alpha=0.7,label='QCD')
        plt.hist(jet_features['fj_pt'],weights=label_Hbb,bins=np.linspace(0,4000,101),density=True,alpha=0.7,label='H(bb)')
        plt.xlabel(r'Jet $p_{T}$ [GeV]')
        plt.ylabel('Fraction of jets')
        plt.legend()
        plt.savefig(vis_path + 'fj_pt_hist.png')

        plt.figure()

        plt.hist(jet_features['fj_sdmass'],weights=label_QCD,bins=np.linspace(0,300,101),density=True,alpha=0.7,label='QCD')
        plt.hist(jet_features['fj_sdmass'],weights=label_Hbb,bins=np.linspace(0,300,101),density=True,alpha=0.7,label='H(bb)')
        plt.xlabel(r'Jet $m_{SD}$ [GeV]')
        plt.ylabel('Fraction of jets')
        plt.legend()

        plt.show()
        plt.savefig(vis_path + 'fj_sdmass_hist.png')



        # SV FEATURES HISTOGRAMS DEPICTING DISCRIMINATORY EFFECT
        plt.figure()
        plt.hist(sv_features['sv_pt'].counts,weights=label_QCD,bins=np.linspace(-2,40,51),density=True,alpha=0.7,label='QCD')
        plt.hist(sv_features['sv_pt'].counts,weights=label_Hbb,bins=np.linspace(-2,40,51),density=True,alpha=0.7,label='H(bb)')
        plt.xlabel('SV pt Count')
        plt.ylabel('Fraction of jets')
        plt.legend()
        plt.savefig(vis_path + 'svptcounts_hist.png')

        plt.figure()
        plt.hist(sv_features['sv_mass'].counts,weights=label_QCD,bins=np.linspace(-2,40,51),density=True,alpha=0.7,label='QCD')
        plt.hist(sv_features['sv_mass'].counts,weights=label_Hbb,bins=np.linspace(-2,40,51),density=True,alpha=0.7,label='H(bb)')
        plt.xlabel('SV mass Count')
        plt.ylabel('Fraction of jets')
        plt.legend()
        plt.savefig(vis_path + 'svmasscounts_hist.png')



        # ROC CURVES

        disc = np.nan_to_num(sv_features['nsv'],nan=0)

        fpr, tpr, threshold = roc_curve(label_Hbb, disc)
        # plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, lw=2.5, label="AUC = {:.1f}%".format(auc(fpr,tpr)*100))
        plt.xlabel(r'False positive rate')
        plt.ylabel(r'True positive rate')
        #plt.semilogy()
        plt.ylim(0,1)
        plt.xlim(0,1)
        plt.plot([0, 1], [0, 1], lw=2.5, label='Random, AUC = 50.0%')
        plt.grid(True)
        plt.legend(loc='upper left')
        plt.show()
        plt.savefig(vis_path + 'svcount_roc.png')


        disc = np.nan_to_num(sv_features['sv_pt'].max()/jet_features['fj_pt'],nan=0)

        fpr, tpr, threshold = roc_curve(label_Hbb, disc)
        # plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, lw=2.5, label="AUC = {:.1f}%".format(auc(fpr,tpr)*100))
        plt.xlabel(r'False positive rate')
        plt.ylabel(r'True positive rate')
        #plt.semilogy()
        plt.ylim(0,1)
        plt.xlim(0,1)
        plt.plot([0, 1], [0, 1], lw=2.5, label='Random, AUC = 50.0%')
        plt.grid(True)
        plt.legend(loc='upper left')
        plt.show()
        plt.savefig(vis_path + 'maxsvpt-fjpt_roc.png')
    