import os

import numpy as np
from sklearn.preprocessing import StandardScaler

from ols import OLS_pytorch
from utils import load_dict


def get_activations(activations_dir, layer_name):
    train_file = os.path.join(activations_dir, "train_" + layer_name + ".npy")
    test_file = os.path.join(activations_dir, "test_" + layer_name + ".npy")

    train_activations = np.load(train_file)
    test_activations = np.load(test_file)

    scaler = StandardScaler()
    train_activations = scaler.fit_transform(train_activations)
    test_activations = scaler.fit_transform(test_activations)

    return train_activations, test_activations


def get_fmri(fmri_dir, ROI):
    ROI_file = os.path.join(fmri_dir, ROI + ".pkl")
    ROI_data = load_dict(ROI_file)

    ROI_data_train = np.mean(ROI_data["train"], axis=1)
    if ROI == "WB":
        voxel_mask = ROI_data["voxel_mask"]
        return ROI_data_train, voxel_mask

    return ROI_data_train


def predict_frmi_fast(train_activations, test_activations, train_fmri, use_gpu=False):
    reg = OLS_pytorch(use_gpu)
    reg.fit(train_activations, train_fmri.T)
    fmri_pred_test = reg.predict(test_activations)

    return fmri_pred_test
