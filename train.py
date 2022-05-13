import matplotlib.pyplot as plt
from modelWrapper import ModelWrapper
from modelManual import ModelManual
from modelTransfer import ModelTransferVGG16
from imageGenerator import get_image_dataset_generator_from_path, show_dataset_images


# CELL 52
model_options = {"manual": ModelManual, "transfer_learning_vgg16": ModelTransferVGG16}


# CELL 7
# Image data generator settings
IMG_SIZE = (128, 128)
BATCH_SIZE = 20  # @param {type:"slider", min:5, max:64, step:1}
DATASET_TYPE = "compressed"  # @param ["original", "compressed"]
EPOCHS = 30  # @param {type:"slider", min:5, max:50, step:5}
MODEL_TYPE = "manual"  # @param ["manual", "transfer_learning_vgg16"]


# CELL 44
# This value get from the end of result prepare.py
dataset_distribution_path = ""
dataset_distribution_compressed_path = {
    "train": "/home/dipadana/Learn/capstone/structured_compressed_dataset/train",
    "validation": "/home/dipadana/Learn/capstone/structured_compressed_dataset/validation",
    "test": "/home/dipadana/Learn/capstone/structured_compressed_dataset/test",
}

dataset_dist = (
    dataset_distribution_compressed_path
    if DATASET_TYPE == "compressed"
    else dataset_distribution_path
)

train_ds, validation_ds, test_ds = get_image_dataset_generator_from_path(
    train_path=dataset_dist["train"],
    validation_path=dataset_dist["validation"],
    test_path=dataset_dist["test"],
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)


# CELL 45
print("Train images")
show_dataset_images(train_ds, figsize=(8, 8))
# plt.show()
print("Validation images")
show_dataset_images(validation_ds, figsize=(8, 8))
# plt.show()


# CELL 57
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
    metric_title="accuracy", metrics=["accuracy", "val_accuracy"]
)
cnn.show_metrics_per_epochs(metric_title="loss", metrics=["loss", "val_loss"])

cnn.evaluate()
cnn.show_confusion_matrix(dataset=cnn.test_ds, figsize=(8, 6.6))
plt.show()
