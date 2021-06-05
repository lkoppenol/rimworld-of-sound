import abc
import random
from tensorflow import keras
from tensorflow.python.keras.backend import argmax, equal, cast, floatx
from tensorflow.python.keras.optimizer_v2.adam import Adam


class Discriminator(abc.ABC):
    def __init__(self, input_shape: (int,), num_classes: int, name: str):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.name = name
        self.model = self._compile_model()

    @abc.abstractmethod
    def _compile_model(self):
        raise NotImplementedError

    def fit(self, train_data, epochs, validation_data=None, **kwargs):
        self.model.fit(train_data, epochs=epochs, validation_data=validation_data, **kwargs)
        return self

    def save(self, path):
        self.model.save(path)
        return self

    def load(self, path):
        self.model = keras.models.load_model(path)


class MultiLabelDiscriminator(Discriminator):
    def __init__(self, input_shape: (int,), num_classes: int, conv_activation: str = 'relu'):
        name = f'convolutional_discriminator_{random.randint(0, 999999):06}'
        self.conv_activation = conv_activation
        super().__init__(input_shape, num_classes, name)

    def _compile_model(self):
        discriminator = keras.Sequential(
            [
                keras.layers.Input(shape=self.input_shape),
                keras.layers.Conv2D(8, kernel_size=(5, 10), activation='relu'),
                keras.layers.MaxPooling2D(pool_size=(3, 10)),
                keras.layers.Conv2D(16, kernel_size=(5, 10), activation='relu'),
                keras.layers.MaxPooling2D(pool_size=(3, 10)),
                keras.layers.Flatten(),
                keras.layers.Dense(512, activation='relu'),
                keras.layers.Dense(self.num_classes, activation='sigmoid'),
            ],
            name=self.name
        )
        discriminator.compile(loss="binary_crossentropy", optimizer="adam", metrics=["binary_accuracy"])
        return discriminator


class LabelDiscriminator(Discriminator):
    def __init__(self, input_shape: (int,), num_classes: int):
        name = f'convolutional_discriminator_{random.randint(0, 999999):06}'
        super().__init__(input_shape, num_classes, name)

    @staticmethod
    def custom_loss(real_loss, noise_class=0, noise_weight=0.001):
        """
        Usage:
        loss = self.custom_loss(keras.losses.categorical_crossentropy)
        discriminator.compile(loss=loss, optimizer="adam", metrics=["categorical_accuracy"])

        :param real_loss: loss function
        :param noise_class: class id of noise
        :param noise_weight: multiplier of losses of noise class
        """
        def weighed_loss(true, pred):
            batch_classes = argmax(true)

            # noise is 1's, others = 0
            noise = cast(equal(batch_classes, noise_class), floatx())

            # set noise to noise weight, set others to 1
            weight = (noise * (-1 + noise_weight)) + 1
            return real_loss(true, pred) * weight
        return weighed_loss

    def _compile_model(self):
        discriminator = keras.Sequential(
            [
                keras.layers.Input(shape=self.input_shape),
                keras.layers.Conv2D(8, kernel_size=(5, 10), activation='relu'),
                keras.layers.MaxPooling2D(pool_size=(3, 10)),
                keras.layers.Conv2D(16, kernel_size=(5, 10), activation='relu'),
                keras.layers.MaxPooling2D(pool_size=(3, 10)),
                keras.layers.Flatten(),
                keras.layers.Dense(512, activation='relu'),
                keras.layers.Dense(self.num_classes, activation='sigmoid'),
            ],
            name=self.name
        )
        loss = self.custom_loss(keras.losses.categorical_crossentropy)
        # loss = keras.losses.categorical_crossentropy
        opt = Adam()
        discriminator.compile(loss=loss, optimizer=opt, metrics=["categorical_accuracy"])
        return discriminator


class FlatDiscriminator(Discriminator):
    def __init__(self, input_shape: (int,), num_classes: int):
        name = f'flat_discriminator_{random.randint(0, 999999):06}'
        super().__init__(input_shape, num_classes, name)

    @staticmethod
    def custom_loss(real_loss, noise_class=0, noise_weight=0.001):
        """
        Usage:
        loss = self.custom_loss(keras.losses.categorical_crossentropy)
        discriminator.compile(loss=loss, optimizer="adam", metrics=["categorical_accuracy"])

        :param real_loss: loss function
        :param noise_class: class id of noise
        :param noise_weight: multiplier of losses of noise class
        """
        def weighed_loss(true, pred):
            batch_classes = argmax(true)

            # noise is 1's, others = 0
            noise = cast(equal(batch_classes, noise_class), floatx())

            # set noise to noise weight, set others to 1
            weight = (noise * (-1 + noise_weight)) + 1
            return real_loss(true, pred) * weight
        return weighed_loss

    def _compile_model(self):
        discriminator = keras.Sequential(
            [
                keras.layers.Input(shape=self.input_shape),
                keras.layers.Dense(512, activation='relu'),
                keras.layers.Dense(self.num_classes, activation='softmax'),
            ],
            name=self.name
        )
        # loss = self.custom_loss(keras.losses.categorical_crossentropy)
        # loss = keras.losses.categorical_crossentropy
        opt = Adam()
        discriminator.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=["categorical_accuracy"])
        return discriminator
