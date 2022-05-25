import tensorflow as tf
import numpy as np
import matplotlib.cm as cm
from tensorflow import keras
from util import get_cnn_last_conv_layer_name
import matplotlib.pyplot as plt
from modelWrapper import ModelWrapper
from util import get_img_array, get_cnn_conv_layers, get_nested_file_path
import math
import random


def get_gradcam_superimposed_image_from_path(img_path, img_size, model: tf.keras.models.Model):
    img_array = get_img_array(img_path, img_size)
    heatmap = get_gradcam_heatmap(img_array/255.0, model)
    superimposed_img = get_gradcam_superimosed_img(img_array, heatmap)
    return superimposed_img

def get_gradcam_heatmap(img_array, model: tf.keras.models.Model, pred_index=None):
    last_conv_layer_name = get_cnn_last_conv_layer_name(model)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(np.expand_dims(img_array, axis=0))
        if pred_index is None:
            pred_index = np.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def get_gradcam_superimosed_img(img_array, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img_array
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    return superimposed_img

def show_img_freature_map(img_path: str, wrapper: ModelWrapper, ncols:int=2**4, cmap='viridis'):
    img_array = get_img_array(img_path=img_path, size=wrapper.input_shape[:-1])
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array/255.0

    conv_layers = get_cnn_conv_layers(wrapper.model)
    outputs = [layer.output for layer in conv_layers]
    layer_names = [layer.name for layer in conv_layers]

    model = tf.keras.models.Model(inputs=wrapper.model.inputs, outputs=outputs)
    feature_maps = model.predict(img_array)

    for i, (layer_name, fmap) in enumerate(zip(layer_names, feature_maps)):
        print(f'\nlayer {layer_name} with {fmap.shape[-1]} filters, layer {i+1} of {len(feature_maps)}')

        nrows = math.ceil(fmap.shape[-1]/ncols)
        plt.figure(figsize=(ncols, nrows))
        for i in range(fmap.shape[-1]):
            plt.subplot(nrows, ncols, i+1)
            plt.axis('off')
            plt.imshow(fmap[0, :, :, i], aspect='auto', cmap=cmap) # cmap=gray|viridis
        plt.show()

def show_gradcam_images_from_model_wrapper(
    dataset_dir,
    wrapper: ModelWrapper,
    single_label=False,
    custom_label=None,
    figsize=(16,16),
    n=25,
    ncols=5,
):
    paths = get_nested_file_path(dataset_dir)
    if single_label:
        labels = wrapper.labels
        label = custom_label if custom_label is not None and custom_label in labels else random.choice(labels)
        paths = [x for x in paths if label in x]
        print(f'Grad-CAM {label}')
    else:
        print('Grad-CAM from random images')

    n = n if n < len(paths) else len(paths)
    sample = random.sample(paths, n)

    plt.figure(figsize=figsize)
    nrows = math.ceil(len(sample)/ncols)
    for i, path in enumerate(sample):
        img = get_gradcam_superimposed_image_from_path(
            img_path=path,
            img_size=wrapper.input_shape[:-1],
            model=wrapper.model
        )
        plt.subplot(nrows, ncols, i+1)
        plt.imshow(img)
        plt.axis('off')
    plt.show()