import os

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from nilearn import plotting

from src.regression import OLS_pytorch, vectorized_correlation
from src.utils import saveasnii, get_fmri


def get_activations(activations_dir, layer_name):
    train_file = os.path.join(activations_dir, "train_" + layer_name + ".npy")
    test_file = os.path.join(activations_dir, "test_" + layer_name + ".npy")

    train_activations = np.load(train_file)
    test_activations = np.load(test_file)

    scaler = StandardScaler()
    train_activations = scaler.fit_transform(train_activations)
    test_activations = scaler.fit_transform(test_activations)

    return train_activations, test_activations


def predict_fmri(train_activations, test_activations, train_fmri, method="ols", alpha=1e4):
    if method == "ridge":
        reg = Ridge(alpha=alpha)
        reg.fit(train_activations, train_fmri)
        fmri_pred_test = reg.predict(test_activations)
    else:
        reg = OLS_pytorch()
        reg.fit(train_activations, train_fmri.T)
        fmri_pred_test = reg.predict(test_activations)
    return fmri_pred_test


def perform_encoding_val(
    activations_dir,
    fmri_dir,
    results_dir,
    sub,
    layer,
    ROI="WB",
    return_nii=True,
    batch_size=1000,
):
    # Load activations
    pca_dir = os.path.join(activations_dir, "pca_100")
    train_activations, test_activations = get_activations(pca_dir, layer)

    # Load fMRI data
    if ROI == "WB":
        track = "full_track"
    else:
        track = "mini_track"
    fmri_dir = os.path.join(fmri_dir, track)
    sub_fmri_dir = os.path.join(fmri_dir, sub)
    if track == "full_track":
        fmri_train_all, voxel_mask = get_fmri(sub_fmri_dir, ROI)
    else:
        fmri_train_all = get_fmri(sub_fmri_dir, ROI)
    num_voxels = fmri_train_all.shape[1]

    # Using first 900 videos as training and rest of the videos as validation
    test_activations = train_activations[900:, :]
    train_activations = train_activations[:900, :]
    fmri_train = fmri_train_all[:900, :]
    fmri_test = fmri_train_all[900:, :]
    pred_fmri = np.zeros_like(fmri_test)
    pred_fmri_save_path = os.path.join(results_dir, ROI + "_val.npy")

    # Performing regression
    iter = 0

    while iter < num_voxels - batch_size:
        pred_fmri[:, iter : iter + batch_size] = predict_fmri(
            train_activations,
            test_activations,
            fmri_train[:, iter : iter + batch_size],
            method="ridge",
        )
        iter = iter + batch_size
    pred_fmri[:, iter:] = predict_fmri(
        train_activations,
        test_activations,
        fmri_train[:, iter : iter + batch_size],
        method="ridge",
    )

    score = vectorized_correlation(fmri_test, pred_fmri)
    print("----------------------------------------------------------------------------")
    print(
        "Mean correlation for ROI : ",
        ROI,
        "in ",
        sub,
        " using ",
        layer,
        " is :",
        round(score.mean(), 6),
    )
    np.save(pred_fmri_save_path, pred_fmri)

    # Result visualization
    if track == "full_track" and return_nii:
        visual_mask_3D = np.zeros((78, 93, 71))
        visual_mask_3D[voxel_mask == 1] = score
        brain_mask = "./example.nii"
        nii_save_path = os.path.join(results_dir, ROI + "_val.nii")
        nii_img = saveasnii(brain_mask, nii_save_path, visual_mask_3D)
        return nii_img
    else:
        return


def perform_encoding_test(
    activations_dir,
    fmri_dir,
    results_dir,
    sub,
    layer,
    ROI="WB",
    batch_size=1000,
):
    # Load activations
    pca_dir = os.path.join(activations_dir, "pca_100")
    train_activations, test_activations = get_activations(pca_dir, layer)

    # Load fMRI data
    if ROI == "WB":
        track = "full_track"
    else:
        track = "mini_track"
    fmri_dir = os.path.join(fmri_dir, track)
    sub_fmri_dir = os.path.join(fmri_dir, sub)
    if track == "full_track":
        fmri_train_all, voxel_mask = get_fmri(sub_fmri_dir, ROI)
    else:
        fmri_train_all = get_fmri(sub_fmri_dir, ROI)
    num_voxels = fmri_train_all.shape[1]

    fmri_train = fmri_train_all
    num_test_videos = 102
    pred_fmri = np.zeros((num_test_videos, num_voxels))
    pred_fmri_save_path = os.path.join(results_dir, ROI + "_test.npy")

    # Performing regression
    iter = 0

    while iter < num_voxels - batch_size:
        pred_fmri[:, iter : iter + batch_size] = predict_fmri(
            train_activations,
            test_activations,
            fmri_train[:, iter : iter + batch_size],
            method="ridge",
        )
        iter = iter + batch_size
    pred_fmri[:, iter:] = predict_fmri(
        train_activations,
        test_activations,
        fmri_train[:, iter : iter + batch_size],
        method="ridge",
    )

    np.save(pred_fmri_save_path, pred_fmri)
