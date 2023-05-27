import nibabel as nib
import pickle


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
