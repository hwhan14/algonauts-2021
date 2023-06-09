import os

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from nilearn import plotting

from src.ols import OLS_pytorch, vectorized_correlation
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


def predict_fmri_fast(train_activations, test_activations, train_fmri, use_gpu=False):
    reg = OLS_pytorch(use_gpu)
    reg.fit(train_activations, train_fmri.T)
    fmri_pred_test = reg.predict(test_activations)

    return fmri_pred_test


def perform_encoding(
    activations_dir,
    fmri_dir,
    results_dir,
    sub,
    layer,
    ROI="WB",
    mode="val",
    visualize_results=True,
    batch_size=1000,
):
    if torch.cuda.is_available():
        use_gpu = True
    else:
        use_gpu = False

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

    # Creating data splits
    if mode == "val":
        # Using first 900 videos as training and rest of the videos as validation
        test_activations = train_activations[900:, :]
        train_activations = train_activations[:900, :]
        fmri_train = fmri_train_all[:900, :]
        fmri_test = fmri_train_all[900:, :]
        pred_fmri = np.zeros_like(fmri_test)
        pred_fmri_save_path = os.path.join(results_dir, ROI + "_val.npy")
    else:
        fmri_train = fmri_train_all
        num_test_videos = 102
        pred_fmri = np.zeros((num_test_videos, num_voxels))
        pred_fmri_save_path = os.path.join(results_dir, ROI + "_test.npy")

    # Performing regression
    iter = 0

    while iter < num_voxels - batch_size:
        pred_fmri[:, iter : iter + batch_size] = predict_fmri_fast(
            train_activations,
            test_activations,
            fmri_train[:, iter : iter + batch_size],
            use_gpu=use_gpu,
        )
        iter = iter + batch_size
    pred_fmri[:, iter:] = predict_fmri_fast(
        train_activations,
        test_activations,
        fmri_train[:, iter : iter + batch_size],
        use_gpu=use_gpu,
    )
    if mode == "val":
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

        nii_save_path = os.path.join(results_dir, ROI + "_val.nii")

        # Result visualization
        if track == "full_track" and visualize_results:
            visual_mask_3D = np.zeros((78, 93, 71))
            visual_mask_3D[voxel_mask == 1] = score
            brain_mask = "/content/example.nii"
            saveasnii(brain_mask, nii_save_path, visual_mask_3D)
            plotting.plot_glass_brain(
                nii_save_path,
                plot_abs=False,
                title="Correlation for " + sub + " and " + layer,
                display_mode="lyr",
                colorbar=True,
                vmin=-1,
                vmax=1,
            )

    np.save(pred_fmri_save_path, pred_fmri)
