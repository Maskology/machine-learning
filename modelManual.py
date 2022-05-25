import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from modelInterface import ModelInterface


class ModelManual(ModelInterface):

    def get_model(self):
        return Sequential([
            Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.relu, input_shape=self.input_shape),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.relu),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu),
            MaxPooling2D(pool_size=(2, 2)),

            Dropout(rate=.2),

            Flatten(),
            Dense(units=512, activation=tf.nn.relu),
            Dense(units=self.output_units, activation=tf.nn.softmax),
        ])

    def get_compile_params(self):
        return {
            'loss': tf.losses.CategoricalCrossentropy(),
            'optimizer': tf.optimizers.Adam(),
            'metrics': ["accuracy"]
        }

    def get_callbacks(self):
        return [
            EarlyStopping(monitor='val_loss', patience=5)
        ]
