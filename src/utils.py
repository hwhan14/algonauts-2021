import os
import pickle

import numpy as np
import nibabel as nib
from nilearn import plotting


def save_dict(dict, filename):
    with open(filename, "wb") as f:
        pickle.dump(dict, f)


def load_dict(filename):
    with open(filename, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        ret_di = u.load()
    return ret_di


def saveasnii(brain_mask, nii_save_path, nii_data):
    img = nib.load(brain_mask)
    print(img.shape)
    nii_img = nib.Nifti1Image(nii_data, img.affine, img.header)
    nib.save(nii_img, nii_save_path)


def get_fmri(fmri_dir, ROI):
    ROI_file = os.path.join(fmri_dir, ROI + ".pkl")
    ROI_data = load_dict(ROI_file)

    ROI_data_train = np.mean(ROI_data["train"], axis=1)
    if ROI == "WB":
        voxel_mask = ROI_data["voxel_mask"]
        return ROI_data_train, voxel_mask

    return ROI_data_train


def visualize_activity(vid_id, sub):
    fmri_dir = "./participants_data_v2021"
    track = "full_track"
    results_dir = "/content/"
    track_dir = os.path.join(fmri_dir, track)
    sub_fmri_dir = os.path.join(track_dir, sub)
    fmri_train_all, voxel_mask = get_fmri(sub_fmri_dir, "WB")
    visual_mask_3D = np.zeros((78, 93, 71))
    visual_mask_3D[voxel_mask == 1] = fmri_train_all[vid_id, :]
    brain_mask = "./example.nii"
    nii_save_path = os.path.join(results_dir, "vid_activity.nii")
    saveasnii(brain_mask, nii_save_path, visual_mask_3D)
    plotting.plot_glass_brain(
        nii_save_path, title="fMRI response", plot_abs=False, display_mode="lyr", colorbar=True
    )
