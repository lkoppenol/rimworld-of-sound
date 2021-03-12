# Tensorflow log level
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from loguru import logger
import tensorflow as tf

from rimworld.rimgan.gan import Gan
from rimworld.rimgan.generator import BaseGenerator
from rimworld.rimgan.rimgan import Mnist
from rimworld.rimgan.discriminator import StraightDiscriminator


if len(tf.config.list_physical_devices('GPU')) > 0:
    logger.warning("WARNING: GPU AVAILABLE. BRACE YOURSELF FOR 1337SPEED TRAINING.")

# CONFIG
discriminator_train_epochs = 15
generator_input_classes = 50

# INPUT
datasource = Mnist()
train_data, test_data = datasource.get_train_test()
data_shape = datasource.shape
num_classes = datasource.num_classes  # 0 t/m 9 digits, 10 = noise
logger.info(f"Loaded '{datasource.name}' data with data shape {data_shape} containing {num_classes} classes")

# MODELS
discriminator = StraightDiscriminator(data_shape, num_classes) \
    .fit(train_data, discriminator_train_epochs, test_data) \
    .save('models/discriminator.h5')

generator = BaseGenerator(input_length=generator_input_classes, output_shape=data_shape)

# GAN MET DIE BANAN
gan = Gan.from_couple(generator, discriminator)

epoch_step = 5
epoch_max = 200
for epoch_count in range(0, epoch_max, epoch_step):
    gan.fit(
        sample_size=int(2e3),
        batch_size=int(1e2),
        epochs=epoch_step,
        validation_split=0.1
    )
    generator.save(f'models/generator_{epoch_count}.h5')
