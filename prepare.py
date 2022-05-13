import os
import math
import random
import matplotlib.pyplot as plt
from PIL import Image
from compress import CompressImgDataset
from split import DatasetDirectoryLabelSplit


# CELL 5
# Data Path
DATASET_DIR = "balinese mask"
DATASET_PATH = os.path.join(os.getcwd(), DATASET_DIR)

# Compressed
COMPRESSED_DATASET_DIR = "compressed_dataset"
COMPRESSED_DATASET_PATH = os.path.join(os.getcwd(), COMPRESSED_DATASET_DIR)

# Split Distribution
STRUCTURED_COMPRESSED_DATASET_PATH = os.path.join(
    os.getcwd(), "structured_compressed_dataset"
)
STRUCTURED_DATASET_PATH = os.path.join(os.getcwd(), "structured_dataset")


# CELL 7
# Compress Image Settings
COMPRESS_QUALITY_PERCENT = 20  # @param {type:"slider", min:1, max:100, step:1}
COMPRESS_SAMPLE = 2  # @param {type:"slider", min:1, max:4, step:1}
COMPRESS_OVERWRITE_EXISTS = False  # @param {type:"boolean"}
COMPRESS_IMG_SIZE = (500, 500)

# Split dataset settings
TRAINING_SPLIT_PERCENT = 70  # @param {type:"slider", min:60, max:90, step:5}
TRAINING_SPLIT = TRAINING_SPLIT_PERCENT / 100
VALIDATION_SPLIT_PERCENT = 20  # @param {type:"slider", min:10, max:40, step:5}
VALIDATION_SPLIT = VALIDATION_SPLIT_PERCENT / 100


# CELL 9
# Assert Split Dataset Settings
train_val_split = TRAINING_SPLIT_PERCENT + VALIDATION_SPLIT_PERCENT
test_split_percent = 100 - train_val_split
assert test_split_percent >= 0, "Should provide for test split"
assert train_val_split <= 100, "Invalid split"
print(
    f"Dataset Split: Train {TRAINING_SPLIT_PERCENT}%, Validation {VALIDATION_SPLIT_PERCENT}%, Test {test_split_percent}%"
)


# START PREPARE THE DATA


# COMPRESS DATA


# CELL 27
cid = CompressImgDataset(
    src_dir=DATASET_PATH,
    dst_dir=COMPRESSED_DATASET_PATH,
    img_size=COMPRESS_IMG_SIZE,
    quality=COMPRESS_QUALITY_PERCENT,
    overwrite_exists=COMPRESS_OVERWRITE_EXISTS,
)
cid.compress()


# CELL 28
cid.show_result_samples(n=COMPRESS_SAMPLE, figsize=(8, 8))


# CELL 30
def show_image_from_each_dir_label(dir, ncols=4, figsize=None, shuffle=True):
    # Getting first file of each label
    paths = []
    labels = sorted(os.listdir(dir))
    for label in labels:
        label_dir = os.path.join(dir, label)
        files = os.listdir(label_dir)
        if shuffle:
            random.shuffle(files)
        for f in files:
            file_path = os.path.join(label_dir, f)
            if os.path.isfile(file_path):
                paths.append((file_path, label))
                break

    # Plott images
    nrows = math.ceil(len(paths) / ncols)
    plt.figure(figsize=figsize)
    for i, data in enumerate(paths):
        path, label = data
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(Image.open(path))
        plt.title(label)
        plt.axis("off")
    plt.show()


show_image_from_each_dir_label(COMPRESSED_DATASET_PATH, figsize=(16, 8))


# SPLIT DATA


# CELL 39
compressed_split = DatasetDirectoryLabelSplit(
    src_dir=COMPRESSED_DATASET_PATH,
    dst_dir=STRUCTURED_COMPRESSED_DATASET_PATH,
    train_split=TRAINING_SPLIT,
    validation_split=VALIDATION_SPLIT,
)
compressed_split.create()
dataset_distribution_compressed_path = compressed_split.distribution_path
print(compressed_split.distribution_path)
