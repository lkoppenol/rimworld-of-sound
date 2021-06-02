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
        print(self.input_length)
        generator = keras.Sequential([
            keras.Input(shape=self.input_length, name='g_i'),
            layers.Reshape((14,40,1)),
            layers.UpSampling2D(size=(3,2)),
            layers.Conv2D(16, kernel_size=(5,10), activation='relu', padding="SAME"),
            layers.UpSampling2D(size=(3, 13)),
            layers.Conv2D(8, kernel_size=(5, 10), activation='relu', padding="SAME"),
            layers.Conv2D(1, kernel_size=(1, 16), activation='relu', padding="VALID"),
            #layers.Dense(output_values, activation='sigmoid'),
            #layers.Dropout(0.5),
            layers.Reshape(self.output_shape, name='g_r')
        ], name='generator')
        generator.summary()
        return generator