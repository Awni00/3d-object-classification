import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.keras.preprocessing import dataset_utils
from tensorflow.python.ops import image_ops, io_ops


def create_rgbd_dataset(rgbd_data_path, label_mode='categorical', azure=True):
    """creates a tensorflow dataset from the washington rgbd-dataset

    Args:
        rgbd_data_path (str): path to rgbd_dataset
        label_mode (str, optional): 'int' or 'categorical'. Defaults to 'categorical'.

    Returns:
        ZipDataset of form ((rgb_img, depth_img), label)
    """

    image_paths = [str(file) for file in get_files_in_dir(
        pathlib.Path(rgbd_data_path)) if is_rgb_im(file)]
    labels = [label_from_path(path, azure=azure) for path in image_paths]

    class_names = list(np.unique(labels))
    label_int_dict = {class_name: i for i,
                      class_name in enumerate(class_names)}
    int_labels = [label_int_dict[class_name] for class_name in labels]

    rgbd_dataset = paths_and_labels_to_dataset(
        image_paths, int_labels, len(class_names), label_mode)

    rgbd_dataset.class_names = class_names
    rgbd_dataset.file_paths = image_paths
    rgbd_dataset.label_int_dict = label_int_dict

    return rgbd_dataset

def create_rgb_dataset(rgbd_data_path, label_mode='categorical'):
    image_paths = [str(file) for file in get_files_in_dir(pathlib.Path(rgbd_data_path)) if is_rgb_im(file)]
    labels = [label_from_path(path) for path in image_paths]

    class_names = list(np.unique(labels))
    label_int_dict = {class_name: i for i, class_name in enumerate(class_names)}
    int_labels = [label_int_dict[class_name] for class_name in labels]

    rgb_dataset = paths_and_labels_to_rgb_dataset(image_paths, int_labels, len(class_names), label_mode)

    rgb_dataset.class_names = class_names
    rgb_dataset.file_paths = image_paths
    rgb_dataset.label_int_dict = label_int_dict

    return rgb_dataset

def label_from_path(path, azure=True):
    '''gets label from path directory structure'''
    if azure: return path.split('/')[-2]
    else: return path.split('\\')[-2]

def paths_and_labels_to_dataset(image_paths, labels, num_classes, label_mode):
    """Constructs a dataset of images and labels."""
    path_ds = dataset_ops.Dataset.from_tensor_slices(image_paths)
    img_ds = path_ds.map(lambda path: load_rgb_depth_img_from_path(path))
    label_ds = dataset_utils.labels_to_dataset(labels, label_mode, num_classes)
    img_ds = dataset_ops.Dataset.zip((img_ds, label_ds))

    return img_ds


def paths_and_labels_to_rgb_dataset(image_paths, labels, num_classes, label_mode):
    """Constructs a dataset of images and labels."""
    path_ds = dataset_ops.Dataset.from_tensor_slices(image_paths)
    img_ds = path_ds.map(lambda path: load_rgb_img_from_path(path))
    label_ds = dataset_utils.labels_to_dataset(labels, label_mode, num_classes)
    img_ds = dataset_ops.Dataset.zip((img_ds, label_ds))

    return img_ds


def load_rgb_depth_img_from_path(path):
    '''loads rgb and depth images from a given path'''
    rgb_path = path

    str_len = tf.strings.length(path)
    depth_path = tf.strings.substr(path, 0, str_len-4) + '_depth.png'

    rgb_img = io_ops.read_file(rgb_path)
    rgb_img = image_ops.decode_image(
        rgb_img, channels=3, expand_animations=False)

    depth_img = io_ops.read_file(depth_path)
    depth_img = image_ops.decode_image(
        depth_img, channels=1, expand_animations=False)
    return rgb_img, depth_img


def load_rgb_img_from_path(rgb_path):
    rgb_img = io_ops.read_file(rgb_path)
    rgb_img = image_ops.decode_image(rgb_img, channels=3, expand_animations=False)

    return rgb_img

non_rgb = ['mask', 'depth']


def is_rgb_im(file):
    '''determines whether a file is an rgb image from its name'''
    is_img = file.suffix == '.png'
    is_rgb = not any([x in file.name for x in non_rgb])

    return is_img and is_rgb


def get_files_in_dir(path):
    '''gets list of all files in a directory given by a pathlib Path. (recursive)'''
    files = []
    for entry in path.iterdir():
        if entry.is_file():
            files.append(entry)
        elif entry.is_dir():
            files += get_files_in_dir(entry)

    return files


def plot_rgb_depth(rgb_im, depth_im, mask_image=None, fig_size=(10, 5)):
    '''plots rgb, depth, and mask images next to each other'''

    if mask_image is None:
        fig, (rgb_ax, depth_ax) = plt.subplots(
            nrows=1, ncols=2, figsize=fig_size)

        rgb_ax.imshow(rgb_im)
        rgb_ax.set_title('RGB Image')

        depth_ax.imshow(depth_im, cmap='binary')
        depth_ax.set_title('Depth Image')
    else:
        fig, (rgb_ax, depth_ax, mask_ax) = plt.subplots(
            nrows=1, ncols=3, figsize=fig_size)

        rgb_ax.imshow(rgb_im)
        rgb_ax.set_title('RGB Image')

        depth_ax.imshow(depth_im, cmap='binary')
        depth_ax.set_title('Depth Image')

        mask_ax.imshow(mask_image, cmap='binary')
        mask_ax.set_title('Binary mask')

    return fig
