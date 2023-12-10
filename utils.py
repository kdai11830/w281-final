import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import cv2
import os
import random
from tqdm import tqdm
from skimage.feature import hog

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
            if counter > limit_to:
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
