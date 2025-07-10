"""Data preparation code 

Author - Ximi
License - MIT
"""
import utils
import config

import numpy as np
import pandas as pd

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import normalize
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
scaler = RobustScaler() 




n_segments = config.N_SEGMENTS

def data_loader_fusion(feature_type="fusion", val=True, base_dir="data"):
    # Step 1: Load fused features
    Xy = np.load(f"{base_dir}/Xy_{feature_type}.npy", allow_pickle=True)
    x1_list, x2_list, y_list = [], [], []

    for clip_id, marlin, openface, label in Xy:
        x1_list.append(marlin)
        x2_list.append(openface)
        y_list.append(label)

    x1 = np.array(x1_list)  # MARLIN (N, 768)
    x2 = np.array(x2_list)  # OpenFace (N, T, D)
    y = np.array(y_list)    # Labels (N,)

    # Reshape MARLIN to (N, 1, 768) if needed
    if len(x1.shape) == 2:
        x1 = x1[:, np.newaxis, :]  # (N, 1, 768)

    # Step 2: Stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, val_idx = next(sss.split(x1, y))

    x1_train, x1_val = x1[train_idx], x1[val_idx]
    x2_train, x2_val = x2[train_idx], x2[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Step 3: Oversample classes 0 and 1 in train set
    oversampler = RandomOverSampler(sampling_strategy='not majority', random_state=42)    
    x1_train_flat = x1_train.reshape((x1_train.shape[0], -1))  # Flatten for oversampler
    x2_train_flat = x2_train.reshape((x2_train.shape[0], -1))
    x_combined = np.concatenate([x1_train_flat, x2_train_flat], axis=1)

    x_resampled, y_resampled = oversampler.fit_resample(x_combined, y_train)

    # Recover x1 and x2 from flattened
    marlin_dim = x1_train.shape[2]
    openface_dim = x2_train.shape[1] * x2_train.shape[2]

    x1_final = x_resampled[:, :marlin_dim].reshape(-1, 1, marlin_dim)
    x2_final = x_resampled[:, marlin_dim:].reshape(-1, x2_train.shape[1], x2_train.shape[2])
    y_final = np.array(y_resampled)

    x1_temp, x1_test, x2_temp, x2_test, y_temp, y_test = train_test_split(
    x1, x2, y, test_size=0.2, stratify=y, random_state=42
    )

    # Step 2: Then split remaining into train/val (if val=True)
    if val:
        x1_train, x1_val, x2_train, x2_val, y_train, y_val = train_test_split(
            x1_temp, x2_temp, y_temp, test_size=0.2, stratify=y_temp, random_state=42
        )
    else:
        x1_train, x2_train, y_train = x1_temp, x2_temp, y_temp

    if val:
        return (
            (x1_final, x2_final, y_final),  # train
            (x1_val, x2_val, y_val),        # val
            (x1_test, x2_test, y_test)      # test
        )
    else:
        return (
            (x1_final, x2_final, y_final),  # train
            (x1_test, x2_test, y_test)      # test
        )

def data_loader_v1(feature_type, val=False, scale=True, base_dir='data'):
    
    """Data load without having separate npy files for splits
    """
    Xy = np.load(f'{base_dir}/Xy_{feature_type}.npy', allow_pickle=True)
    Xy = utils.cleanXy(Xy)
    
    
    features_label_map = {}
    for xy in Xy:  
        features_label_map[xy[0]] = (xy[1], xy[2])
        
    train_x = []
    train_y = []
    if val:
        val_x = []
        val_y = []
        
    test_x = []
    test_y = []
    
    trainXy = utils.read_file(f'{base_dir}/train.txt')
    testXy = utils.read_file(f'{base_dir}/test.txt')
    valXy = utils.read_file(f'{base_dir}/valid.txt')
    
    for e in trainXy:
        try:
            xy = features_label_map[e]
            if xy[1] != config.SNP:
                train_x.append(xy[0])
                train_y.append(config.LABEL_MAP[xy[1]])
        except KeyError as k:
            pass
#             print ('not found(train): ', k)
    if scale:    
        X = np.array(train_x)
        scaler.fit(X.reshape(-1, X.shape[-1]))
        for i in range(len(train_x)):
            train_x[i] = scaler.transform(train_x[i]) 
    
    for e in valXy:
        try:
            xy = features_label_map[e]
            if xy[1] != config.SNP:
#                     
                x = xy[0]
                if scale:
                    x = scaler.transform(xy[0])
                if val:
                    val_x.append(x)
                    val_y.append(config.LABEL_MAP[xy[1]])
                else:
                    train_x.append(x)
                    train_y.append(config.LABEL_MAP[xy[1]])
        except KeyError as k:
            pass
#                 print ('not found(val): ', k)
            
    for e in testXy:
        try:
            xy = features_label_map[e]
            if xy[1] != config.SNP:                
                x = xy[0]
                if scale:
                    x = scaler.transform(xy[0])
                test_x.append(x)
                test_y.append(config.LABEL_MAP[xy[1]])
        except KeyError as k:
            pass
#             print ('not found(test): ', k)
    if val:
        return ((np.array(train_x), np.array(train_y)), 
                (np.array(val_x), np.array(val_y)), 
                (np.array(test_x), np.array(test_y)))
    else:
        return ((np.array(train_x), np.array(train_y)), 
                (np.array(test_x), np.array(test_y)))
    



if __name__ == '__main__':
    print("✅ Testing data prep")
    feature_type = config.FUSION

    # Unpack correctly
    (train_x1, train_x2, train_y), (val_x1, val_x2, val_y), (test_x1, test_x2, test_y) = data_loader_fusion(feature_type, val=True)

    print("✅ Train X1 shape:", train_x1.shape)
    print("✅ Train X2 shape:", train_x2.shape)
    print("✅ Train y shape:", train_y.shape)
    print("✅ Val X1 shape:", val_x1.shape)
    print("✅ Val X2 shape:", val_x2.shape)
    print("✅ Val y shape:", val_y.shape)
    print("✅ Test X1 shape:", test_x1.shape)
    print("✅ Test X2 shape:", test_x2.shape)
    print("✅ Test y shape:", test_y.shape)

