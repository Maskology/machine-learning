import os
import humanize
import shutil
import re
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D


def remove_existing_dir(dir):
    if os.path.exists(dir) and os.path.isdir(dir):
        shutil.rmtree(dir)


def remove_existing_file(path):
    if os.path.exists(path) and os.path.isfile(path):
        os.remove(path)


def create_dir_if_not_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)


def get_file_size(path):
    return humanize.naturalsize(os.path.getsize(path))


def copy(src, dst):
    filename = os.path.basename(dst)
    remove_existing_dir(dst)
    remove_existing_file(dst)
    create_dir_if_not_exists(re.sub(filename, "", dst))

    if os.path.isdir(src):
        shutil.copytree(src, dst)
    elif os.path.isfile(src):
        shutil.copy(src, dst)
    else:
        print(f"{src} is neither directory nor file")


def get_nested_file_path(dir):
    paths = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            path = os.path.join(root, file)
            paths.append(path)
    return paths


def get_img_array(img_path, size):
    img = keras.preprocessing.image.load_img(path=img_path, target_size=size)
    array = keras.preprocessing.image.img_to_array(img)
    return array


def get_cnn_last_conv_layer_name(model: tf.keras.models.Model):
    conv_layers = get_cnn_conv_layers(model)
    assert len(conv_layers) > 0, "model doesn't have conv2d layer"
    return conv_layers[-1].name


def get_cnn_conv_layers(model: tf.keras.models.Model):
    return list(filter(lambda x: isinstance(x, Conv2D), model.layers))