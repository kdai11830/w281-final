from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import cv2
from skimage import feature
import colorsys
import os
import random
from tqdm import tqdm
from PIL import Image

from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

import torch
from torch import nn
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights

"""
Utility functions to be called from a Jupyter notebook...
"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # #
## # #   Load Data & Apply Feature Extraction    # # ##
# # # # # # # # # # # # # # # # # # # # # # # # # # # #


def load_dataset(data_dir, cl_limit=30, img_limit=220):
    """ """
    X = []
    Y = []
    idx_to_cl = {}

    for i, cl in enumerate(os.listdir(data_dir)):
        if i >= cl_limit:
            break

        print(cl)
        idx_to_cl[i] = cl
        for j, fname in tqdm(enumerate(os.listdir(os.path.join(data_dir, cl)))):
            if j >= img_limit:
                break

            img = cv2.imread(os.path.join(data_dir, cl, fname))
            X.append(img)
            Y.append(i)

    return np.array(X), np.array(Y), idx_to_cl


def apply_features(X, feature_functions={}):
    """ """
    features = []
    features_idxs = {}
    for img in tqdm(X):
        feature = np.array([])
        for function_name in feature_functions.keys():
            start = len(feature)
            feature = np.append(feature, feature_functions[function_name](img))
            features_idxs[function_name] = (start, len(feature) - 1)
        features.append(feature)
    return np.array(features), features_idxs


def load_dataset_apply_features(
    data_dir, feature_functions={}, cl_limit=30, img_limit=220
):
    """ """
    X = []
    Y = []
    idx_to_cl = {}
    features = []
    features_idxs = {}

    for i, cl in enumerate(os.listdir(data_dir)):
        if i >= cl_limit:
            break

        print(cl)
        idx_to_cl[i] = cl
        for j, fname in tqdm(enumerate(os.listdir(os.path.join(data_dir, cl)))):
            if j >= img_limit:
                break

            img = cv2.imread(os.path.join(data_dir, cl, fname))
            X.append(img)
            Y.append(i)

            feature = np.array([])
            for function_name in feature_functions.keys():
                start = len(feature)
                feature = np.append(feature, feature_functions[function_name](img))
                # start of feature is length before appending, end of feature is length -1 after appending
                features_idxs[function_name] = (start, len(feature) - 1)
            features.append(feature)

    return np.array(X), np.array(Y), idx_to_cl, np.array(features), features_idxs


# # # # # # # # # # # # # # # # # # # # # # #
## # #   Feature Extraction Methods    # # ##
# # # # # # # # # # # # # # # # # # # # # # #


# # #   Color   # # #
def extract_rgb_features(img):
    """
    takes dataset and feature dictionaries as input
    gets the mean and variance of each RGB channel
    """
    channel_mean = img.mean(axis=(0, 1))
    channel_var = img.var(axis=(0, 1))
    return np.append(channel_mean, channel_var)


def extract_hsv_features(img):
    """
    takes dataset and feature dictionaries as input
    gets the mean and variance of each hsv channel
    """
    img_convert = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    channel_mean = img_convert.mean(axis=(0, 1))
    channel_var = img_convert.var(axis=(0, 1))
    return np.append(channel_mean, channel_var)


# # #   Histogram of Oriented Gradients (HOG)   # # #
def extract_RGB_or_HSV_hog_features(
    img,
    orientations=9,
    pixels_per_cell=(60, 60),
    cells_per_block=(1, 1),
    RGB_or_HSV_or_gray="gray",
):
    """
    HOG on RGB, HSV or grayscale channels.

    from skimage import data, exposure
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    """
    if RGB_or_HSV_or_gray == "R":
        img = img[:, :, 2]
    elif RGB_or_HSV_or_gray == "G":
        img = img[:, :, 1]
    elif RGB_or_HSV_or_gray == "B":
        img = img[:, :, 0]
    elif RGB_or_HSV_or_gray == "H":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = img[:, :, 0]
    elif RGB_or_HSV_or_gray == "S":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = img[:, :, 1]
    elif RGB_or_HSV_or_gray == "V":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = img[:, :, 2]
    elif RGB_or_HSV_or_gray == "gray":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        print("specify a channel or grayscale...")

    hog = feature.hog(
        img,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
    )
    return hog


# # #   DAISY   # # #
def extract_daisy_features(
    img,
    step=220,
    radius=70,
    rings=3,
    histograms=8,
    orientations=8,
    RGB_or_HSV_or_gray="gray",
):
    """
    DAISY on RGB, HSV or grayscale channels.
    """
    if RGB_or_HSV_or_gray == "R":
        img = img[:, :, 2]
    elif RGB_or_HSV_or_gray == "G":
        img = img[:, :, 1]
    elif RGB_or_HSV_or_gray == "B":
        img = img[:, :, 0]
    elif RGB_or_HSV_or_gray == "H":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = img[:, :, 0]
    elif RGB_or_HSV_or_gray == "S":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = img[:, :, 1]
    elif RGB_or_HSV_or_gray == "V":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = img[:, :, 2]
    elif RGB_or_HSV_or_gray == "gray":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        print("specify a channel or grayscale...")

    daisy = feature.daisy(
        img,
        step=step,
        radius=radius,
        rings=rings,
        histograms=histograms,
        orientations=orientations,
    )
    return daisy.flatten()


# # #   Deep Learning (Convolutional Based)   # # #
def retype_image(in_img):
    if np.max(in_img) > 1:
        in_img = in_img.astype(np.uint8)
    else:
        in_img = (in_img * 255.0).astype(np.uint8)
    return in_img


class ResNet:
    """ """

    def __init__(self) -> None:
        self.model_weights = ResNet101_Weights.IMAGENET1K_V2
        model = resnet101(weights=self.model_weights)
        # remove classification layer
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.model.eval()
        self.preprocess = self.model_weights.transforms()

    def preprocess_im(self, im):
        """ """
        im = retype_image(im)
        return self.preprocess(Image.fromarray(im))

    def process_im(self, im):
        """ """
        im = self.preprocess_im(im=im)
        im = im.unsqueeze(0).to("cpu")
        output = self.model(im).squeeze()
        return output.detach().numpy()


def apply_resnet(im):
    model = ResNet()
    return model.process_im(im=im)


class EffNet:
    """ """

    def __init__(self) -> None:
        self.model_weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1
        model = efficientnet_v2_m(weights=self.model_weights)
        # remove classification layer
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.model.eval()
        self.preprocess = self.model_weights.transforms()

    def preprocess_im(self, im):
        """ """
        im = retype_image(im)
        return self.preprocess(Image.fromarray(im))

    def process_im(self, im):
        """ """
        im = self.preprocess_im(im=im)
        im = im.unsqueeze(0).to("cpu")
        output = self.model(im).squeeze()
        return output.detach().numpy()


def apply_effnet(im):
    model = EffNet()
    return model.process_im(im=im)


# # # # # # # # # # # # # # # # #
## # #   Visualizations    # # ##
# # # # # # # # # # # # # # # # #
def get_random_pics_by_class(data_dir):
    """
    Get a random example image per class and display given a path.
    """
    dict_imgs = {}
    for cl in os.listdir(data_dir):
        f = random.choice(os.listdir(os.path.join(data_dir, cl)))
        # print(os.path.join(data_dir, cl, f))
        dict_imgs[cl] = plt.imread(os.path.join(data_dir, cl, f))

    num_classes = len(dict_imgs.keys())
    fig, axes = plt.subplots(nrows=num_classes // 3, ncols=3, figsize=(8, num_classes))
    for i, cl in enumerate(dict_imgs.keys()):
        axes[i // 3, i % 3].imshow(dict_imgs[cl])
        axes[i // 3, i % 3].axis("off")
        axes[i // 3, i % 3].set_title(cl)
    plt.title("One Random Example Per Class")
    plt.show()
