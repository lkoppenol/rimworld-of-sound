from tensorflow import keras
import numpy as np
import tensorflow as tf


class Mnist:
    def __init__(self, train_noise_ratio=0.2, test_noise_ratio=0.1):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        self.shape = (28, 28, 1)
        self.num_classes = 11  # 0 t/m 9 digits, 10 = noise
        self.name = 'mnist'
        self.train_dataset = self.prepare_data(x_train, y_train, train_noise_ratio)
        self.test_dataset = self.prepare_data(x_test, y_test, test_noise_ratio)

    def get_train_test(self) -> (tf.data.Dataset, tf.data.Dataset):
        return self.train_dataset, self.test_dataset

    @staticmethod
    def generate_noise(n, label=10):
        x = np.random.randint(0, 256, (n, 28, 28))
        y = np.ones((n,)) * label
        return x, y

    def prepare_data(self, x, y, noise_ratio, num_classes=11):
        n_noise_samples = int(len(y) * noise_ratio)
        x_noise, y_noise = self.generate_noise(n_noise_samples)
        total_x = np \
            .concatenate((x, x_noise)) \
            .astype('float32') \
            / 255
        shaped_x = np.expand_dims(total_x, -1)

        total_y = np.concatenate((y, y_noise))
        vectored_y = keras.utils.to_categorical(total_y, num_classes)

        dataset = tf.data \
            .Dataset.from_tensor_slices((shaped_x, vectored_y)) \
            .shuffle(1_024) \
            .batch(256)

        return dataset
