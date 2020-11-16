import sys
import json

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '../data')

from etl import get_features_labels, clean_array

with open('../../config/data-params.json') as fh:
    data_cfg = json.load(fh)
    
    data = get_features_labels(**data_cfg)
    features, labels, specs, tree = data
    
    label_QCD = labels[:,0]
    label_Hbb = labels[:,1]
    
    print(sum(label_QCD+label_Hbb)/len(label_QCD+label_Hbb))
    
    jet_features = tree.arrays(branches=['fj_pt', 
                                     'fj_sdmass'],
                          entrystop=20000,
                          namedecode='utf-8')
    
    #print(jet_features)
    print(len(label_QCD))
    print(len(label_Hbb))
    
    jet_features = np.stack([jet_features[feat] for feat in ['fj_pt','fj_sdmass']],axis=1)
    #jet_features = clean_array(jet_features, specs, data_cfg['remove_mass_pt_window'])
    
    print(len(jet_features))
    
    plt.figure()

    plt.hist(jet_features[:,0],weights=label_QCD,bins=np.linspace(0,4000,101),density=True,alpha=0.7,label='QCD')
    plt.hist(jet_features[:,0],weights=label_Hbb,bins=np.linspace(0,4000,101),density=True,alpha=0.7,label='H(bb)')
    plt.xlabel(r'Jet $p_{T}$ [GeV]')
    plt.ylabel('Fraction of jets')
    plt.legend()
    plt.savefig('fig1.png')
    
    plt.figure()

    plt.hist(jet_features[:,1],weights=label_QCD,bins=np.linspace(0,300,101),density=True,alpha=0.7,label='QCD')
    plt.hist(jet_features[:,1],weights=label_Hbb,bins=np.linspace(0,300,101),density=True,alpha=0.7,label='H(bb)')
    plt.xlabel(r'Jet $m_{SD}$ [GeV]')
    plt.ylabel('Fraction of jets')
    plt.legend()

    plt.show()
    plt.savefig('fig2.png')