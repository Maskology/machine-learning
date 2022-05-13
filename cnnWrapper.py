import os
import re
import shutil
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

from util import remove_existing_dir, create_dir_if_not_exists
from modelInterface import ModelInterface

DEFAULT_EPOCHS = 15
DEFAULT_TRAIN_VERBOSE = 1
DEFAULT_FIGSIZE = None


class ModelWrapper:
    def __init__(
        self,
        train_ds: tf.keras.preprocessing.image.DirectoryIterator,
        validation_ds: tf.keras.preprocessing.image.DirectoryIterator,
        test_ds: tf.keras.preprocessing.image.DirectoryIterator,
        model: ModelInterface,
        epochs=DEFAULT_EPOCHS,
        saved_model_dir=os.path.join(os.getcwd(), "saved_model"),
    ):
        # Data
        self.train_ds = train_ds
        self.validation_ds = (validation_ds,)
        self.test_ds = test_ds
        self.labels = self.get_labels_from_dataset()

        # Saved Model
        self.saved_model_dir = saved_model_dir

        # Model Configuration
        self.input_shape = self.get_input_shape_from_dataset()
        self.output_units = len(self.labels)
        model = model(self.input_shape, self.output_units)
        self.model = model.model
        self.compile_params = model.compile_params

        # Training
        self.training = None
        self.epochs = epochs
        self.callbacks = (
            model.callbacks if model.callbacks else self.get_default_callbacks()
        )

        self.model.summary()

    def compile(self):
        self.model.compile(**self.compile_params)

    def train(self, verbose=DEFAULT_TRAIN_VERBOSE):
        self.training = self.model.fit(
            x=self.train_ds,
            validation_data=self.validation_ds,
            epochs=self.epochs,
            verbose=verbose,
            callbacks=self.callbacks,
        )

    def evaluate(self):
        return self.model.evaluate(self.test_ds)

    # Just work in google colab
    # def test_upload_images(self, show_image=False):
    #     uploaded = files.upload()
    #     paths = uploaded.keys()
    #     print()

    #     for i, path in enumerate(paths):
    #         img = image.load_img(path, target_size=self.input_shape[:-1])
    #         x = image.img_to_array(img)
    #         x = np.expand_dims(x, axis=0)
    #         images = np.vstack([x])
    #         label = self.labels[np.argmax(self.model.predict(images))]
    #         msg = f"{path} Predicted as {label}"

    #         if show_image:
    #             plt.subplot(len(paths), 1, i + 1)
    #             plt.imshow(img)
    #             plt.title(msg)
    #             plt.axis("off")
    #         else:
    #             print(msg)

    #         os.remove(path)

    def show_metrics_per_epochs(
        self,
        metric_title="accuracy",
        metrics=["accuracy", "val_accuracy"],
        figsize=DEFAULT_FIGSIZE,
    ):
        plt.figure(figsize=figsize)
        for metric in metrics:
            plt.plot(self.training.history[metric], label=metric)
        plt.xlabel("epochs")
        plt.ylabel(metric_title)
        plt.legend()
        plt.show()

    def show_confusion_matrix(
        self, dataset, scaled=True, figsize=None, hline=True, hline_color="#f00"
    ):
        labels = list(dataset.class_indices.keys())
        y_pred = self.model.predict(dataset)
        y_pred = np.argmax(y_pred, axis=1)

        cf_matrix = tf.math.confusion_matrix(dataset.classes, y_pred)
        cf_matrix = cf_matrix / np.sum(cf_matrix, axis=1) if scaled else cf_matrix

        plt.figure(figsize=figsize)
        if hline:
            for i in range(1, len(labels)):
                plt.axhline(y=i, linewidth=1, color=hline_color)

        heatmap = sns.heatmap(cf_matrix, annot=True, cmap="Blues")
        heatmap.set_xlabel("\nPredicted")
        heatmap.set_ylabel("Actual")
        heatmap.xaxis.set_ticklabels(labels)
        heatmap.yaxis.set_ticklabels(labels)

    def export_to_tfjs(self, dir="tfjs"):
        def convert():
            remove_existing_dir(dir)
            create_dir_if_not_exists(dir)
            cmd = f"tensorflowjs_converter --input_format=tf_saved_model {self.saved_model_dir} {dir}"
            os.system(cmd)

        self.saved_model_wrapper(convert)

    def export_to_tflite(self, path="model.tflite"):
        def convert():
            filename = os.path.basename(path)
            dir = re.sub(filename, "", path)
            create_dir_if_not_exists(dir)
            converter = tf.lite.TFLiteConverter.from_saved_model(self.saved_model_dir)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            tflite_model = converter.convert()
            with open(path, "wb") as f:
                f.write(tflite_model)

        self.saved_model_wrapper(convert)

    def saved_model_wrapper(self, export_func):
        tf.saved_model.save(self.model, self.saved_model_dir)
        export_func()
        shutil.rmtree(self.saved_model_dir)

    def get_default_callbacks(self):
        return [
            EarlyStopping(monitor="val_loss", patience=3),
        ]

    def get_input_shape_from_dataset(self):
        first_batch = train_ds.next()[0]
        first_img = first_batch[0]
        return tuple(tf.shape(first_img).numpy())

    def get_labels_from_dataset(self):
        return list(self.train_ds.class_indices.keys())
