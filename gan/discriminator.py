import abc
import random
from tensorflow import keras


class Discriminator(abc.ABC):
    def __init__(self, input_shape: (int,), num_classes: int, name: str):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.name = name
        self.model = self._compile_model()

    @abc.abstractmethod
    def _compile_model(self):
        raise NotImplementedError

    def fit(self, train_data, epochs, validation_data=None, steps_per_epoch=None):
        self.model.fit(train_data, epochs=epochs, validation_data=validation_data, steps_per_epoch=steps_per_epoch)
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
        discriminator.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["categorical_accuracy"])
        return discriminator