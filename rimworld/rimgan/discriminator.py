import abc
import random

from tensorflow import keras
from tensorflow.keras import layers


class Discriminator(abc.ABC):
    def __init__(self, input_shape: (int,), num_classes: int, name: str):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.name = name
        self.model = self._compile_model()

    @abc.abstractmethod
    def _compile_model(self):
        raise NotImplementedError

    def fit(self, train_data, epochs, validation_data=None):
        self.model.fit(train_data, epochs=epochs, validation_data=validation_data)
        return self

    def save(self, path):
        self.model.save(path)
        return self


class ConvolutionalDiscriminator(Discriminator):
    def __init__(self, input_shape: (int,), num_classes: int, conv_activation: str = 'relu'):
        name = f'convolutional_discriminator_{random.randint(0, 999999):06}'
        self.conv_activation = conv_activation
        super().__init__(input_shape, num_classes, name)

    def _compile_model(self):
        discriminator = keras.Sequential(
            [
                keras.Input(shape=self.input_shape),
                layers.Conv2D(32, kernel_size=(5, 5), activation=self.conv_activation),
                layers.Dropout(0.3),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.3),
                layers.Dense(128, activation='sigmoid'),
                layers.Dropout(0.3),
                layers.Dense(self.num_classes, activation="softmax"),
            ],
            name=self.name
        )
        discriminator.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return discriminator


class StraightDiscriminator(Discriminator):
    def __init__(self, input_shape: (int,), num_classes: int):
        name = f'straight_discriminator_{random.randint(0, 999999):06}'
        super().__init__(input_shape, num_classes, name)

    def _compile_model(self):
        discriminator = keras.Sequential(
            [
                keras.Input(shape=self.input_shape),
                layers.Flatten(),
                layers.Dense(128, activation='sigmoid'),
                layers.Dropout(0.3),
                layers.Dense(128),
                layers.LeakyReLU(),
                layers.Dropout(0.3),
                layers.Dense(self.num_classes, activation="softmax")
            ],
            name=self.name
        )
        discriminator.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return discriminator


class BranchedDiscriminator(Discriminator):
    def __init__(self, input_shape: (int,), num_classes: int):
        name = f'branched_discriminator_{random.randint(0, 999999):06}'
        super().__init__(input_shape, num_classes, name)

    def _compile_model(self):
        input_image = layers.Input(shape=self.input_shape)

        # branch one: dense layers
        b1 = layers.Flatten()(input_image)
        b1 = layers.Dense(64, activation='sigmoid')(b1)
        b1_out = layers.Dense(32, activation='relu')(b1)

        # branch two: conv + pooling layers
        b2 = layers.Conv2D(32, (3, 3), activation='sigmoid')(input_image)
        b2 = layers.MaxPooling2D((2, 2))(b2)
        b2 = layers.Conv2D(64, (3, 3), activation='relu')(b2)
        b2_out = layers.MaxPooling2D((2, 2))(b2)

        # merge two branches
        flattened_b2 = layers.Flatten()(b2_out)
        merged = layers.concatenate([b1_out, flattened_b2])

        output = layers.Dense(self.num_classes, activation="softmax")(merged)

        discriminator = keras.models.Model(input_image, output, name=self.name)
        discriminator.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return discriminator
