import random
import os
import matplotlib.pyplot as plt
from modelWrapper import ModelWrapper
from modelManual import ModelManual
from modelTransfer import ModelTransferVGG16
from imageGenerator import get_image_dataset_generator_from_path
from util import get_nested_file_path
from gradCam import show_img_freature_map, show_gradcam_images_from_model_wrapper


model_options = {"manual": ModelManual, "transfer_learning_vgg16": ModelTransferVGG16}


# Image data generator settings
IMG_SIZE = (280, 280)
BATCH_SIZE = 64  # @param {type:"slider", min:5, max:64, step:1}
DATASET_TYPE = "compressed"  # @param ["original", "compressed"]

EPOCHS = 50  # @param {type:"slider", min:5, max:50, step:5}
MODEL_TYPE = "transfer_learning_vgg16"  # @param ["manual", "transfer_learning_vgg16"]


# This value get from the end of result prepare.py
dataset_distribution_path = ""
dataset_distribution_compressed_path = {'train': '/home/dipadana/Learn/capstone_bangkit/model/structured_dataset/train', 'validation': '/home/dipadana/Learn/capstone_bangkit/model/structured_dataset/validation', 'test': '/home/dipadana/Learn/capstone_bangkit/model/structured_dataset/test'}

dataset_dist = (
    dataset_distribution_compressed_path
    if DATASET_TYPE == "compressed"
    else dataset_distribution_path
)

train_ds, validation_ds, test_ds = get_image_dataset_generator_from_path(
    train_path=dataset_dist['train'],
    validation_path=dataset_dist['validation'],
    test_path=dataset_dist['test'],
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)


# print("Train images")
# show_dataset_images(train_ds, figsize=(8, 8))
# plt.show()
# print("Validation images")
# show_dataset_images(validation_ds, figsize=(8, 8))
# plt.show()


cnn = ModelWrapper(
    train_ds=train_ds,
    validation_ds=validation_ds,
    test_ds=test_ds,
    model=model_options[MODEL_TYPE],
    epochs=EPOCHS,
)
cnn.compile()
cnn.train()

cnn.show_metrics_per_epochs(
    metric_title='Accuracy per epochs',
    metric_name='accuracy',
    metrics=['accuracy', 'val_accuracy']
)

cnn.show_metrics_per_epochs(
    metric_title='Loss per epochs',
    metric_name='loss',
    metrics=['loss', 'val_loss']
)


# cnn.export_to_h5 ('./saved_model/model.h5')


cnn.evaluate()
cnn.show_confusion_matrix(dataset=cnn.test_ds, figsize=(8, 6.6))
plt.show()


print(cnn.get_classification_report(dataset=cnn.test_ds))


SHOW_FEATURE_MAP = False
STRUCTURED_COMPRESSED_DATASET_PATH = os.path.join(os.getcwd(), 'structured_compressed_dataset')

if SHOW_FEATURE_MAP:
    paths = get_nested_file_path(STRUCTURED_COMPRESSED_DATASET_PATH)
    img_path = random.choice(paths)
    show_img_freature_map(img_path=img_path, wrapper=cnn)

dataset_dir = dataset_dist['train']
wrapper = cnn
ncols = 8
n = ncols*3
figsize = (16, 6)
labels = sorted(os.listdir(dataset_dir))

for label in labels:
    show_gradcam_images_from_model_wrapper(
        dataset_dir=dataset_dir,
        wrapper=wrapper,
        ncols=ncols,
        n=n,
        figsize=figsize,
        single_label=True,
        custom_label=label,
    )
    print()

# cnn.export_to_tfjs("./tfjs-model")
# print("Model converted to tfjs")
