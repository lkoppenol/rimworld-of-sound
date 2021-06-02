from tensorflow import keras
import numpy as np


class Gan:
    def __init__(self, input_classes, output_classes):
        self.input_classes = input_classes
        self.output_classes = output_classes
        self.model = None
        self.discriminator = None

    @classmethod
    def from_couple(cls, generator, discriminator):
        input_classes = generator.input_length
        output_classes = discriminator.num_classes
        gan = cls(input_classes, output_classes)
        gan.model = keras.Sequential([
            generator.model,
            discriminator.model
        ])
        gan.discriminator = discriminator
        gan.compile_model()
        return gan

    @classmethod
    def from_file(cls, path, input_classes, output_classes):
        return cls(input_classes, output_classes) \
            .load(path) \
            .compile_model()

    def compile_model(self):
        optimizer = keras.optimizers.Adam()
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        return self

    def _generate_samples(self, sample_size):
        y = np.random.randint(0, self.output_classes, (sample_size, 1))
        y = keras.utils.to_categorical(y, self.output_classes)
        columns_to_add = self.input_classes - self.output_classes
        added_cols = np.random.randn(sample_size, columns_to_add)
        x = np.concatenate((y, added_cols), axis=1)
        return x, y

    def fit(self, sample_size, batch_size, epochs, validation_split):
        self.discriminator.model.trainable = False
        x, y = self._generate_samples(sample_size)
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        self.discriminator.model.trainable = True
        return self

    def save(self, path):
        self.model.save(path)
        return self

    def load(self, path):
        self.model = keras.models.load_model(path)
        return self

    def _get_vector(self, class_id, non_random, randomized=True):
        if randomized:
            bonus_vector = np.random.randn(1, self.input_classes - self.output_classes)
        else:
            bonus_vector = np.ones((1, self.input_classes - self.output_classes)) * non_random
        vector = np.concatenate((np.zeros((1, self.output_classes)), bonus_vector), axis=1)
        if class_id < self.output_classes:  # Assume last class is noise class
            vector[0, class_id] = 1
        return vector

    def _get_vectors(self, class_ids, randomized, non_random=0):
        vector_list = [self._get_vector(class_id, non_random, randomized) for class_id in class_ids]
        return np.concatenate(vector_list)

    def generate(self, class_ids, randomized=True, plot_ready=False):
        vectors = self._get_vectors(class_ids, randomized)
        generated = self.get_generator().predict(vectors)
        if plot_ready:
            generated *= 255
        return generated

    def get_generator(self):
        return self.model.layers[0]

    def get_discriminator(self):
        return self.model.layers[1]
