import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.applications.vgg16 import VGG16
from modelInterface import ModelInterface


FINE_TUNING = 2

class ModelTransferVGG16(ModelInterface):

    fine_tune: int = FINE_TUNING

    def get_model(self):
        base = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)

        if self.fine_tune is not None:
            for layer in base.layers[:-self.fine_tune]:
                layer.trainable = False
        else:
            base.trainable = False

        # Functional
        x = base.output
        x = Dropout(rate=.2)(x)
        x = Flatten()(x)
        x = Dense(units=1024, activation=tf.nn.relu)(x)
        x = Dense(units=512, activation=tf.nn.relu)(x)
        x = Dense(units=256, activation=tf.nn.relu)(x)
        x = Dense(units=128, activation=tf.nn.relu)(x)
        x = Dense(units=64, activation=tf.nn.relu)(x) #0.95 #0.93 #0.94
        # x = Dense(units=32, activation=tf.nn.relu)(x) #0.95
        # x = Dense(units=16, activation=tf.nn.relu)(x) #0.93
        x = Dropout(rate=.2)(x)
        output_layer = Dense(units=self.output_units, activation=tf.nn.softmax)(x)
        return Model(inputs=base.input, outputs=output_layer)

    def get_compile_params(self):
        return {
            'loss': tf.losses.CategoricalCrossentropy(),
            # 'optimizer': tf.optimizers.SGD(learning_rate=0.0001, momentum=0.9),
            'optimizer': tf.optimizers.Adam(),
            'metrics':["accuracy"]
        }

    def get_callbacks(self):
        return [
            # EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True),
            EarlyStopping(monitor='val_accuracy', mode='max', patience=5,  restore_best_weights=True)
        ]
