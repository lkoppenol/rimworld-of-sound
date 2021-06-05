import abc
import operator
import random
from functools import reduce

from tensorflow import keras
from tensorflow.keras import layers


class Generator(abc.ABC):
    def __init__(self, input_length: int, output_shape: (int, ), name: str):
        self.input_length = input_length
        self.output_shape = output_shape
        self.name = name
        self.model = self._compile_model()

    @abc.abstractmethod
    def _compile_model(self):
        raise NotImplementedError

    def fit(self, train_data, epochs, test_data=None):
        self.model.fit(train_data, epochs=epochs, validation_data=test_data)
        return self

    def save(self, path):
        self.model.save(path)
        return self


class BaseGenerator(Generator):
    def __init__(self, input_length: int, output_shape: (int,)):
        name = f'flat_generator_{random.randint(0, 999999):06}'
        super().__init__(input_length, output_shape, name)

    def _compile_model(self):
        output_values = reduce(operator.mul, self.output_shape, 1)
        generator = keras.Sequential([
            keras.layers.Reshape((1,11,1)),
            keras.layers.Conv2DTranspose(8, (2,6), strides=1, padding="VALID"),
            #keras.layers.BatchNormalization(),
            # keras.layers.Dense(output_values/64, activation="tanh"),
            # layers.Reshape((16,128,1)),
            layers.Conv2DTranspose(128, kernel_size=3, strides=4, padding="SAME"),
            keras.layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.01),
            layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding="SAME"),
            keras.layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.01),
            layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding="SAME"),
            keras.layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.01),
            layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding="SAME"),
            keras.layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.01),
            layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding="SAME", activation="tanh"),
            # layers.Reshape(self.output_shape),
            #
            # keras.layers.Dense(output_values, activation="relu", kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)),
            # keras.layers.BatchNormalization(),
            #layers.Reshape(self.output_shape),
        ], name='generator')

        print("&&&&&&&&&&& GENERATOR &&&&&&&&&&&&&&")
        generator.summary()
        return generator


class FlatGenerator(Generator):
    def __init__(self, input_length: int, output_shape: (int,)):
        name = f'base_generator_{random.randint(0, 999999):06}'
        super().__init__(input_length, output_shape, name)

    def _compile_model(self):
        generator = keras.Sequential([
            keras.Input(shape=self.input_length),
            layers.Dense(512, activation='relu'),
            layers.Dense(self.output_shape, activation='sigmoid'),
        ], name='generator')
        return generator