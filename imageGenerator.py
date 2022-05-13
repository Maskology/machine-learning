from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_IMG_SIZE = (64, 64)
DEFAULT_RESCALE = 1 / 255.0
DEFAULT_HORIZONTAL_SPLIT = True
DEFAULT_ROTATION_RANGE = 20
DEFAULT_CLASS_MODE = 'categorical'
DEFAULT_SHUFFLE_TRAINING = True
DEFAULT_BATCH_SIZE = 32

DEFAULT_FIGSIZE = None
DEFAULT_SHOW_SIZE = 3

def get_image_dataset_generator_from_path(
    train_path,
    validation_path,
    test_path,
    img_size=DEFAULT_IMG_SIZE,
    class_mode=DEFAULT_CLASS_MODE,
    batch_size=DEFAULT_BATCH_SIZE,
    train_shuffle=DEFAULT_SHUFFLE_TRAINING
):
    train_gen = ImageDataGenerator(
        rescale=DEFAULT_RESCALE,
        horizontal_flip=DEFAULT_HORIZONTAL_SPLIT,
        rotation_range=DEFAULT_ROTATION_RANGE,
    )
    train_ds = train_gen.flow_from_directory(
        directory=train_path,
        batch_size=batch_size,
        target_size=img_size,
        class_mode=class_mode,
        shuffle=train_shuffle
    )

    validation_gen = ImageDataGenerator(rescale=DEFAULT_RESCALE)
    validation_ds = validation_gen.flow_from_directory(
        directory=validation_path,
        batch_size=batch_size,
        target_size=img_size,
        class_mode=class_mode,
    )

    test_gen = ImageDataGenerator(rescale=DEFAULT_RESCALE)
    test_ds = test_gen.flow_from_directory(
        directory=test_path,
        batch_size=batch_size,
        target_size=img_size,
        class_mode=class_mode,
        shuffle=False
    )

    return train_ds, validation_ds, test_ds

def show_dataset_images(dataset, nrows=3, ncols=3, figsize=None):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    labels = list(dataset.class_indices.keys())
    for i in range(nrows):
        for j in range(ncols):
            img, label = dataset.next()
            ax = axes[i][j]
            ax.imshow(img[0])
            ax.set_title(labels[np.argmax(label)])
            ax.axis('off')