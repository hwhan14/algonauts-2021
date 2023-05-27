import os
import glob
import torch
import time

import numpy as np
from tqdm import tqdm
from PIL import Image

from torchvision import transforms
from torch.autograd import Variable

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from decord import VideoReader
from decord import cpu

from src.model.alexnet import alexnet


def load_alexnet(model_checkpoints):
    model = alexnet()
    model_file = model_checkpoints

    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    model_dict = [
        "conv1.0.weight",
        "conv1.0.bias",
        "conv2.0.weight",
        "conv2.0.bias",
        "conv3.0.weight",
        "conv3.0.bias",
        "conv4.0.weight",
        "conv4.0.bias",
        "conv5.0.weight",
        "conv5.0.bias",
        "fc6.1.weight",
        "fc6.1.bias",
        "fc7.1.weight",
        "fc7.1.bias",
        "fc8.1.weight",
        "fc8.1.bias",
    ]
    state_dict = {}

    for i, (k, v) in enumerate(checkpoint.items()):
        state_dict[model_dict[i]] = v

    model.load_state_dict(state_dict)

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    return model


def sample_video_from_mp4(file, num_frames=16):
    vr = VideoReader(file, ctx=cpu(0))

    images, total_frames = [], len(vr)
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=np.int32)

    for seg_ind in indices:
        images.append(Image.fromarray(vr[seg_ind].asnumpy()))

    return images, num_frames


def get_activations_and_save(model, video_list, activations_dir):
    resize_normalize = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    for video_file in tqdm(video_list):
        vid, num_frames = sample_video_from_mp4(video_file)
        video_file_name = os.path.split(video_file)[-1].split(".")[0]
        activations = []

        for frame, img in enumerate(vid):
            input_img = Variable(resize_normalize(img).unsqueeze(0))

            if torch.cuda.is_available():
                input_img = input_img.cuda()

            x = model.forward(input_img)

            for i, feat in enumerate(x):
                if frame == 0:
                    activations.append(feat.data.cpu().numpy().ravel())
                else:
                    activations[i] = activations[i] + feat.data.cpu().numpy().ravel()

            for layer in range(len(activations)):
                save_path = os.path.join(
                    activations_dir, "_".join([video_file_name, "layer", str(layer + 1)]) + ".npy"
                )
                avg_layer_activation = activations[layer] / float(num_frames)
                np.save(save_path, avg_layer_activation)


def PCA_and_save(num_layers, activations_dir, save_dir):
    layers = [f"layer_{i}" for i in range(1, num_layers + 1)]

    n_components = 100
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for layer in tqdm(layers):
        activations_file_list = sorted(glob.glob(activations_dir + "/*" + str(layer) + ".npy"))
        feature_dim = np.load(activations_file_list[0])
        X = np.zeros((len(activations_file_list), feature_dim.shape[0]))

        for i, activation_file in enumerate(activations_file_list):
            temp = np.load(activation_file)
            X[i, :] = temp

        X_train = X[:1000, :]
        X_test = X[1000:, :]

        start_time = time.time()

        X_train = StandardScaler().fit_transform(X_train)
        X_test = StandardScaler().fit_transform(X_test)

        ipca = PCA(n_components=n_components, random_state=SEED)
        ipca.fit(X_train)

        X_train = ipca.transform(X_train)
        X_test = ipca.transform(X_test)

        train_save_path = os.path.join(save_dir, "train_" + str(layer))
        test_save_path = os.path.join(save_dir, "test_" + str(layer))

        np.save(train_save_path, X_train)
        np.save(test_save_path, X_test)
