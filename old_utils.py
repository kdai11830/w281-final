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
from skimage.feature import hog
from PIL import Image

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

import torch
from torch import nn
import torchvision.models as models
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from torchvision import transforms
from pytorch_model_summary import summary


class Imag:
    def __init__(self) -> None:
        pass


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


def load_dataset(data_dir, limit_to=25):
    """
    Load image data set given a path.
    limit_to: limits the number of examples per class to this value.
    """
    data = {}
    for cl in os.listdir(data_dir):
        print(cl)
        data[cl] = []
        counter = 0
        for fname in tqdm(os.listdir(os.path.join(data_dir, cl))):
            if counter > limit_to - 1:
                break
            data[cl].append(plt.imread(os.path.join(data_dir, cl, fname)))
            counter += 1
    return data


def plot_distribution_of_images_per_class(data):
    tmp = {cl: len(data[cl]) for cl in data.keys()}
    vals = sorted(list(tmp.values()))
    labels = [x for _, x in sorted(zip(list(tmp.values()), list(tmp.keys())))]
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.bar(range(len(tmp)), vals, tick_label=labels)
    fig.savefig("plots/class_distribution.png")
    plt.xticks(rotation=90)


def get_rgb_features(data):
    rgb = {
        cl: pd.melt(
            pd.DataFrame(
                np.array([x.mean(axis=(0, 1)) for x in data[cl]]),
                columns=["R", "G", "B"],
            ),
            var_name="RGB",
            value_name="Average Value",
        )
        for cl in data.keys()
    }
    return rgb


def get_hog_features(data):
    """"""
    hog_dict = {}
    for name in data.keys():
        features_per_im = []
        for im in data[name]:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            f, h = hog(im, orientations=4, pixels_per_cell=(20, 20), visualize=True)
            features_per_im.append(f)
        hog_dict[name] = features_per_im

    return hog_dict


def plot_hist_per_class(data, rgb_features):
    num_classes = len(rgb_features.keys())
    fig, axes = plt.subplots(
        nrows=num_classes // 3, ncols=3, figsize=(18, num_classes * 2)
    )
    for i, cl in enumerate(rgb_features.keys()):
        sns.histplot(
            x="Average Value",
            hue="RGB",
            multiple="layer",
            data=rgb_features[cl],
            ax=axes[i // 3, i % 3],
        )
        axes[i // 3, i % 3].set_title(cl)
    fig.savefig("plots/class_rgb_hist.png")
    plt.show()


def visualize_rgb_features(data):
    rgb_features_3d = pd.DataFrame()
    for cl in data.keys():
        tmp = pd.DataFrame(
            np.array([x.mean(axis=(0, 1)) for x in data[cl]]), columns=["R", "G", "B"]
        )
        tmp["class"] = cl
        rgb_features_3d = pd.concat([rgb_features_3d, tmp], axis=0)

    fig = px.scatter_3d(
        rgb_features_3d, x="R", y="G", z="B", color="class", width=800, height=800
    )
    fig.update_traces(marker_size=6)
    fig.write_html("plots/class_scatter.html")
    fig.show()


def retype_image(in_img):
    if np.max(in_img) > 1:
        in_img = in_img.astype(np.uint8)
    else:
        in_img = (in_img * 255.0).astype(np.uint8)
    return in_img


def slice_model(original_model, from_layer=None, to_layer=None):
    return nn.Sequential(*list(original_model.children())[from_layer:to_layer])


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


class VGG:
    """Currently not working"""

    def __init__(self, feature_version=True) -> None:
        if feature_version:
            self.model_weights = VGG16_Weights.IMAGENET1K_FEATURES
        else:
            self.model_weights = VGG16_Weights.IMAGENET1K_V1

        model = vgg16(weights=self.model_weights)
        # # model_parts_list = list(model.children())[:-1]
        # # # this removes the ReLU and Dropout layers as well as the final linear layer
        # # model_parts_list.append(list(model.children())[-1][:-3])
        # model_parts_list = []
        # for i, child in enumerate(model.children()):
        #     if i == 0 or i == 1:
        #         model_parts_list.append(child)
        #     elif i == 2:
        #         model_parts_list.append(child[:-3])
        # self.model = nn.Sequential(*model_parts_list)
        # print(self.model)
        self.model = model
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
