import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications.vgg16 import VGG16
from modelInterface import ModelInterface


# CELL 51
class ModelTransferVGG16(ModelInterface):
    def get_model(self):
        base = VGG16(
            weights="imagenet", include_top=False, input_shape=self.input_shape
        )
        base.trainable = False
        return Sequential(
            [
                base,
                Flatten(),
                # Dense(units=128, activation=tf.nn.relu),
                # Dense(units=256, activation=tf.nn.relu),
                Dense(units=self.output_units, activation=tf.nn.softmax),
            ]
        )

    def get_compile_params(self):
        return {
            "loss": tf.losses.CategoricalCrossentropy(),
            # 'optimizer': tf.optimizers.SGD(learning_rate=0.0001, momentum=0.9),
            "optimizer": tf.optimizers.Adam(),
            "metrics": ["accuracy"],
        }

    def get_callbacks(self):
        return [
            # EarlyStopping(monitor='val_loss', patience=3),
            EarlyStopping(
                monitor="val_accuracy",
                mode="max",
                patience=5,
                restore_best_weights=True,
            )
        ]
