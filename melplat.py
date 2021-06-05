import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from rimworld.utils import get_label



from gan.discriminator import LabelDiscriminator, FlatDiscriminator
from gan.gan import Gan
from gan.generator import BaseGenerator, FlatGenerator
from loguru import logger
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from rimworld import utils
import soundfile as sf
from time import time
import tensorflow as tf

load_dotenv()


# create .env in root folder, for example:
# ROOT_FOLDER=D:\Projects\nsynth-data\data\stft
ROOT_FOLDER = os.getenv('ROOT_FOLDER')

# LABEL = 'instrument_subtype'
# LABEL_ID = 'instrument_and_pitch_single_label'
# LABEL = 'no_organ'
# LABEL = 'organ_pitch'
LABEL = "pitch"
# LABEL = 'electronic_bass_pitch'
BATCH_SIZE = 32
EPOCHS = 200
SAMPLE_RATE = 16000

DEBUG_MODE = False

if DEBUG_MODE:
    logger.warning("WARNING WARNING DEBUG MODE IS ON WARNING WARNING")


def save_img(g, name, folder='img'):
    g2 = np.tile(g, (100, 1)).T
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    path = folder / Path(name).with_suffix('.png')
    plt.imsave(str(path), g2, format='png', cmap='gray')


def save_wav(g, name, folder='wav'):
    try:
        g2 = np.tile(g, (100, 1)).T
        s = librosa.feature.inverse.mel_to_audio(g2, sr=16000, hop_length=2048)
        # s = librosa.core.spectrum.griffinlim(g2)
        folder = Path(folder)
        folder.mkdir(exist_ok=True, parents=True)
        path = folder / Path(name).with_suffix('.wav')
        sf.write(str(path), s, SAMPLE_RATE)
    except Exception as e:
        logger.error(str(e))


def main(label):
    label_size = utils.label_shapes[label]
    id = time()
    logger.info(f"Running experiment with id = '{id:.0f}'")
    class_weight = {i: 1 for i in range(1, label_size)}
    class_weight[0] = 0.01

    train_df = pd.read_csv(os.path.join(ROOT_FOLDER, 'train.csv'))
    train_df = train_df[train_df.filename.str.contains('organ')]
    train_labels = train_df.filename.apply(lambda x: get_label(x, label, label_size, one_hot=False))
    train_labels = tf.one_hot(train_labels.values, depth=label_size)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_df.drop('filename', axis=1), train_labels)).batch(1024)

    valid_df = pd.read_csv(os.path.join(ROOT_FOLDER, 'test.csv'))
    valid_df = valid_df[valid_df.filename.str.contains('organ')]
    valid_labels = valid_df.filename.apply(lambda x: get_label(x, label, label_size, one_hot=False))
    valid_labels = tf.one_hot(valid_labels.values, depth=label_size)
    valid_dataset = tf.data.Dataset.from_tensor_slices((valid_df.drop('filename', axis=1), valid_labels)).batch(1024)

    if DEBUG_MODE:
        subsample = 10
        logger.warning(f"WARNING DEBUG MODE TAKING TRAINING SAMPLE OF {subsample} B*TCHES")
        train_dataset = train_dataset.take(subsample)
        valid_dataset = valid_dataset.take(subsample)

    discriminator = FlatDiscriminator(
        input_shape=train_df.shape[1] - 1,
        num_classes=label_size
    )

    generator = FlatGenerator(
        input_length=label_size,
        output_shape=train_df.shape[1] - 1
    )

    # GAN MET DIE BANAN
    gan = Gan.from_couple(generator, discriminator)

    epoch_step = 1
    epoch_max = 200
    for epoch_count in range(0, epoch_max, epoch_step):
        logger.info(f"{epoch_count:03} DISCRIMINATE")
        discriminator.fit(
            train_dataset,
            epochs=1,
            validation_data=valid_dataset,
            # steps_per_epoch=2048
        )
        logger.info(f"{epoch_count:03} GAN-ORREA")
        gan.fit(
            sample_size=int(2e6),
            batch_size=int(1e2),
            epochs=epoch_step,
            validation_split=0.1
        )
        logger.info(f"{epoch_count:03} GENERATE")
        generated = gan.generate(range(label_size))

        for i, g in enumerate(generated[12:24]):
            # if i % 10 == 0:
            name = f"{id:.0f}-{epoch_count:04}-{i:04}"
            save_img(g, name)
            save_wav(g, name)


if __name__ == "__main__":
    main(LABEL)
