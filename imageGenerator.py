from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import math

DEFAULT_IMG_SIZE = (64, 64)
DEFAULT_RESCALE = 1 / 255.0
DEFAULT_CLASS_MODE = 'categorical'
DEFAULT_BATCH_SIZE = 32

DEFAULT_FIGSIZE = None
DEFAULT_SHOW_SIZE = 3

DEFAULT_TRAIN_GEN_PARAMS = {
    'rescale': DEFAULT_RESCALE,
    'horizontal_flip': True,
    'rotation_range': 45,
    'width_shift_range': .3,
    'height_shift_range': .3,
    'shear_range': .2,
    'zoom_range': .3,
}
DEFAULT_VAL_GEN_PARAMS = {
    'rescale': DEFAULT_RESCALE,
}
DEFAULT_TEST_GEN_PARAMS = {
    'rescale': DEFAULT_RESCALE,
}


def get_image_dataset_generator_from_path(
        train_path,
        validation_path,
        test_path,

        img_size=DEFAULT_IMG_SIZE,
        class_mode=DEFAULT_CLASS_MODE,
        batch_size=DEFAULT_BATCH_SIZE,

        train_gen_params=DEFAULT_TRAIN_GEN_PARAMS,
        val_gen_params=DEFAULT_VAL_GEN_PARAMS,
        test_gen_params=DEFAULT_TEST_GEN_PARAMS
):
    ds_params = {
        'batch_size': batch_size,
        'target_size': img_size,
        'class_mode': class_mode,
    }

    train_gen = ImageDataGenerator(**train_gen_params)
    train_ds = train_gen.flow_from_directory(train_path, **ds_params, shuffle=True)

    validation_gen = ImageDataGenerator(**val_gen_params)
    validation_ds = validation_gen.flow_from_directory(validation_path, **ds_params, shuffle=False)

    test_gen = ImageDataGenerator(**test_gen_params)
    test_ds = test_gen.flow_from_directory(test_path, **ds_params, shuffle=False)

    return train_ds, validation_ds, test_ds


def show_dataset_images(dataset, ncols=8, n=8 * 2, figsize=(16, 4)):
    labels = list(dataset.class_indices.keys())
    nrows = math.ceil(n / ncols)
    plt.figure(figsize=figsize)
    for i in range(n):
        img, label = dataset.next()
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(img[0])
        plt.title(labels[np.argmax(label)])
        plt.axis('off')
    plt.show()