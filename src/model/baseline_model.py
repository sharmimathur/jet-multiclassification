import sys
import json

import tensorflow.keras as keras
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import uproot

import yaml

from DataGenerator import DataGenerator

sys.path.insert(0, '../data')

from generator import generator

def create_baseline_model(config_path, model_config_path):
    with open(config_path) as fh:
        try:
            # The FullLoader parameter handles the conversion from YAML
            # scalar values to Python the dictionary format
            definitions = yaml.load(fh, Loader=yaml.FullLoader)
        except:
            definitions = json.load(fh)


    features = definitions['features']
    spectators = definitions['spectators']
    labels = definitions['labels']

    nfeatures = definitions['nfeatures']
    nspectators = definitions['nspectators']
    nlabels = definitions['nlabels']
    ntracks = definitions['ntracks']

    with open(model_config_path) as fh:
        data_cfg = json.load(fh)
        train_files = data_cfg['train_file_name']
        test_files = data_cfg['test_file_name']
        val_files = data_cfg['val_file_name']

        batch_size = data_cfg["batch_size"]
        remove_mass_pt_window = data_cfg["remove_mass_pt_window"]
        remove_unlabeled = data_cfg["remove_unlabeled"]
        max_entry = data_cfg['max_entry']
        
    
    gen = DataGenerator([train_files], features, labels, spectators, batch_size=batch_size, n_dim=ntracks, 
                                remove_mass_pt_window=remove_mass_pt_window, 
                                remove_unlabeled=remove_unlabeled, max_entry=max_entry)
    
        

    

#     train_generator, test_generator, val_generator = generator(train_files, test_files, val_files, features, labels, spectators, batch_size, ntracks, remove_mass_pt_window, remove_unlabeled, max_entry)

    

    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, BatchNormalization, Conv1D, Flatten, Lambda, GlobalAveragePooling1D
    import tensorflow.keras.backend as K

    # define Deep Sets model with Conv1D Keras layer
    inputs = Input(shape=(ntracks,nfeatures,), name = 'input')  
    x = BatchNormalization(name='bn_1')(inputs)
    x = Conv1D(64, 1, strides=1, padding='same', name = 'conv1d_1', activation='relu')(x)
    x = Conv1D(32, 1, strides=1, padding='same', name = 'conv1d_2', activation='relu')(x)
    x = Conv1D(32, 1, strides=1, padding='same', name = 'conv1d_3', activation='relu')(x)
    # sum over tracks
    x = GlobalAveragePooling1D(name='pool_1')(x)
    x = Dense(100, name = 'dense_1', activation='relu')(x)
    outputs = Dense(nlabels, name = 'output', activation='softmax')(x)
    keras_model_conv1d = Model(inputs=inputs, outputs=outputs)
    keras_model_conv1d.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(keras_model_conv1d.summary())

    # define callbacks
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    reduce_lr = ReduceLROnPlateau(patience=5,factor=0.5)
    model_checkpoint = ModelCheckpoint('keras_model_conv1d_best.h5', monitor='val_loss', save_best_only=True)
    callbacks = [early_stopping, model_checkpoint, reduce_lr]

    # fit keras model
    history_conv1d = keras_model_conv1d.fit(gen, 
                                            validation_data = gen, 
                                            steps_per_epoch=len(gen), 
                                            validation_steps=len(gen),
                                            max_queue_size=5,
                                            epochs=20, 
                                            shuffle=False,
                                            callbacks = callbacks, 
                                            verbose=0)
    # reload best weights
    keras_model_conv1d.load_weights('keras_model_conv1d_best.h5')

    plt.figure()
    plt.plot(history_conv1d.history['loss'],label='Loss')
    plt.plot(history_conv1d.history['val_loss'],label='Val. loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    plt.savefig('src/visualizations/conv1d_loss.png')
    plt.savefig('test/conv1d_loss.png')


    # COMPARING MODELS
    predict_array_dnn = []
    predict_array_cnn = []
    label_array_test = []
    
    
    test_gen = DataGenerator(definitions['file_name'], features, labels, spectators, batch_size=batch_size, n_dim=ntracks, 
                                remove_mass_pt_window=remove_mass_pt_window, 
                                remove_unlabeled=remove_unlabeled, max_entry=max_entry)
    

    for t in gen:
        label_array_test.append(t[1])
        predict_array_cnn.append(keras_model_conv1d.predict(t[0]))


    predict_array_cnn = np.concatenate(predict_array_cnn,axis=0)
    label_array_test = np.concatenate(label_array_test,axis=0)


    # create ROC curves
    fpr_cnn, tpr_cnn, threshold_cnn = roc_curve(label_array_test[:,1], predict_array_cnn[:,1])

    # plot ROC curves
    plt.figure()
    plt.plot(tpr_cnn, fpr_cnn, lw=2.5, label="Conv1D, AUC = {:.1f}%".format(auc(fpr_cnn,tpr_cnn)*100))
    plt.xlabel(r'True positive rate')
    plt.ylabel(r'False positive rate')
    plt.semilogy()
    plt.ylim(0.001,1)
    plt.xlim(0,1)
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.show()

    plt.savefig('src/visualizations/conv1d.png')
    plt.savefig('test/conv1d.png')