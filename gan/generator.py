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
        name = f'base_generator_{random.randint(0, 999999):06}'
        super().__init__(input_length, output_shape, name)

    def _compile_model(self):
        output_values = reduce(operator.mul, self.output_shape, 1)
        generator = keras.Sequential([
            keras.Input(shape=self.input_length),
            layers.Dense(output_values, activation='relu'),
            layers.Reshape(self.output_shape),
            layers.Conv2D(8, (5, 5), padding='same'),
            layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')
        ], name='generator')
        return generator
