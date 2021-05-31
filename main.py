from time import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from gan.discriminator import MultiLabelDiscriminator
from gan.gan import Gan
from gan.generator import BaseGenerator

from loguru import logger
import tensorflow as tf
import numpy as np


ROOT_FOLDER = r"C:\Users\david.isaacspaternos\broncode\data\stft"
# LABEL = 'instrument_subtype'
LABEL = 'instrument_and_pitch_single_label'
BATCH_SIZE = 32
EPOCHS = 200

DEBUG_MODE = False

if DEBUG_MODE:
    logger.warning("WARNING WARNING DEBUG MODE IS ON WARNING WARNING")


def get_label(filename, label_type, label_size):
    switch = {
        "instrument": get_instrument_label,
        "instrument_subtype": get_instrument_subtype_label,
        "pitch": get_pitch_label,
        "multi": get_multi_label,
        "instrument_and_pitch_single_label": get_instrument_and_pitch,
    }
    sparse_label = switch[label_type](filename)
    #these arent really sparse, just funky code (should hire cleaner)
    if label_type == "multi" or label_type == "instrument_and_pitch_single_label":  
        return sparse_label
    label = np.zeros((label_size, 1))
    label[sparse_label] = 1
    return label


def get_multi_label(filename):
    """ multi label with best recognizable instruments
        bass_electronic, vocal acoustic, organ electronic, string acoustic, other_instruments, noise(when added to dataset), pitch """
    pitch_label = get_pitch_label(filename)
    instrument_label = get_instrument_subtype_label(filename)
    instrument_mapping = {
        1: 0,   # bass_electronic
        19: 1,  # organ_electronic
        24: 2,  # string_acoustic
        30: 3,  # vocal_acoustic
        "other": 4
    }
    try:
        instrument_part_label = instrument_mapping[instrument_label]
    except KeyError:
        instrument_part_label = 4  # Other
    n_instruments = len(instrument_mapping)
    n_pitches = 112  # 112 for pitches, lowest = 9, highest is 120 (check vocal synthetic, it has them both)
    label = np.zeros(n_instruments + n_pitches)
    label[instrument_part_label] = 1
    label[pitch_label + n_instruments - 9] = 1    # +5 because first 5 are instruments, -9 because 009 is the lowest pitch in the nsynth dataset
    return label


def get_instrument_label(filename):
    instrument = "_".join(filename.split('_')[:-2])
    switch = {
        'bass': 0,
        'brass': 1,
        'flute': 2,
        'guitar': 3,
        'keyboard': 4,
        'mallet': 5,
        'organ': 6,
        'reed': 7,
        'string': 8,
        'synth_lead': 9,
        'vocal': 10,
    }
    return switch[instrument]


def get_instrument_subtype_label(filename):
    instrument_label = get_instrument_label(filename)
    subtype = filename.split('_')[-2]
    switch = {
        "acoustic": 0,
        "electronic": 1,
        "synthetic": 2
    }
    label = instrument_label * len(switch) + switch[subtype]
    return label


def get_pitch_label(filename):
    label = int(filename.split('-')[1])
    return label


def get_instrument_and_pitch(filename):
    pitch_label = get_pitch_label(filename)
    instrument_label = get_instrument_subtype_label(filename)

    if instrument_label == 1:
        instrument_label = 0      #"bass_electronic"
    elif instrument_label == 19:
        instrument_label = 1      #"organ_electronic"
    elif instrument_label == 24:
        instrument_label = 2      #"string_acoustic"
    elif instrument_label == 30:
        instrument_label = 3      #"vocal_acoustic"
    else:
        instrument_label = 4      #"other"
        
    label = np.zeros(5*112)
    label[instrument_label*112 + pitch_label-9] = 1
#     print(filename)
#     print(np.argmax(label))
#     time.sleep(10)
    
    return label

def reset(batch_size, label_size):
    imgs = np.zeros((batch_size, 126, 1025, 1))
    labels = np.zeros((batch_size, label_size))
    return imgs, labels


def get_dataset(path, label_type, label_size, batch_size):
    filenames = [f for r, d, fs in os.walk(path) for f in fs]  # tf uses os.walk to determine file order
    labels = [get_label(filename, label_type, label_size) for filename in filenames]
    dataset = tf.keras.preprocessing \
        .image_dataset_from_directory(
            directory=path,
            labels=labels,
            color_mode='grayscale',
            batch_size=batch_size,
            image_size=(126, 1025)
        )
    return dataset


def main():
    # Required folder structure:
    # ROOT_FOLDER\train\anything\all_your_imgs.png
    # ROOT_FOLDER\valid\anything\all_your_imgs.png

    label_shapes = dict(
        instrument=1,
        instrument_subtype=33,
        pitch=128,
        multi=117,
        instrument_and_pitch_single_label=5*112,
    )

    train_folder = os.path.join(ROOT_FOLDER, 'train')
    train_dataset = get_dataset(train_folder, LABEL, label_shapes[LABEL], BATCH_SIZE)
    valid_folder = os.path.join(ROOT_FOLDER, 'valid')
    valid_dataset = get_dataset(valid_folder, LABEL, label_shapes[LABEL], BATCH_SIZE)

    if DEBUG_MODE:
        subsample = 10
        logger.warning(f"WARNING DEBUG MODE TAKING TRAINING SAMPLE OF {subsample} B*TCHES")
        train_dataset = train_dataset.take(subsample)
        valid_dataset = valid_dataset.take(subsample)

    discriminator = MultiLabelDiscriminator(
        input_shape=(126, 1025, 1),
        num_classes=label_shapes[LABEL]
    )
    discriminator.load(r"notebooks/instrument_and_pitch_single_label_model_1621681850")
    #discriminator.fit(train_dataset, epochs=EPOCHS, validation_data=valid_dataset)
    #discriminator.save(f'models/model_{int(time())}')

    generator = BaseGenerator(
        input_length=label_shapes[LABEL],
        output_shape=(126, 1025, 1)
    )

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
        gan.save(f'models/gan_{epoch_count:05}.h5')


if __name__ == "__main__":
    main()
