import os
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from gan.discriminator import LabelDiscriminator
from gan.gan import Gan
from gan.generator import BaseGenerator
from loguru import logger
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from rimworld import utils
import soundfile as sf
from time import time

load_dotenv()


# create .env in root folder, for example:
# ROOT_FOLDER=D:\Projects\nsynth-data\data\stft
ROOT_FOLDER = os.getenv('ROOT_FOLDER')

# LABEL = 'instrument_subtype'
# LABEL_ID = 'instrument_and_pitch_single_label'
# LABEL = 'no_organ'
LABEL = 'organ_pitch'
BATCH_SIZE = 32
EPOCHS = 200
SAMPLE_RATE = 16000

DEBUG_MODE = False

if DEBUG_MODE:
    logger.warning("WARNING WARNING DEBUG MODE IS ON WARNING WARNING")


def save_img(g, name, folder='img'):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    path = folder / Path(name).with_suffix('.png')
    plt.imsave(str(path), g[:, :, 0], format='png', cmap='gray', vmin=0, vmax=1)


def save_wav(g, name, folder='wav'):
    s = utils.reconstruct_from_sliding_spectrum(g[:, :, 0])
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    path = folder / Path(name).with_suffix('.wav')
    sf.write(str(path), s, SAMPLE_RATE)


def main(label):
    label_size = utils.label_shapes[label]
    img_size = (126, 1025, 1)
    id = time()
    logger.info(f"Running experiment with id = '{id:.0f}'")
    class_weight = {i: 1 for i in range(1, label_size)}
    class_weight[0] = 0.01

    train_folder = os.path.join(ROOT_FOLDER, 'train')
    train_dataset = utils.get_image_dataset(train_folder, label, label_size, BATCH_SIZE, n_random_samples=30_000)
    valid_folder = os.path.join(ROOT_FOLDER, 'valid')
    valid_dataset = utils.get_image_dataset(valid_folder, label, label_size, BATCH_SIZE, n_random_samples=1_000)

    if DEBUG_MODE:
        subsample = 10
        logger.warning(f"WARNING DEBUG MODE TAKING TRAINING SAMPLE OF {subsample} B*TCHES")
        train_dataset = train_dataset.take(subsample)
        valid_dataset = valid_dataset.take(subsample)

    discriminator = LabelDiscriminator(
        input_shape=img_size,
        num_classes=label_size
    )

    generator = BaseGenerator(
        input_length=label_size,
        output_shape=img_size
    )

    # GAN MET DIE BANAN
    gan = Gan.from_couple(generator, discriminator)

    epoch_step = 1
    epoch_max = 200
    for epoch_count in range(0, epoch_max, epoch_step):
        logger.info("DISCRIMINATE")
        discriminator.fit(
            train_dataset,
            epochs=1,
            validation_data=valid_dataset,
            steps_per_epoch=2048
        )
        logger.info("GAN-ORREA")
        gan.fit(
            sample_size=int(2e4),
            batch_size=int(1e2),
            epochs=epoch_step,
            validation_split=0.1
        )
        logger.info("GENERATE")
        generated = gan.generate(range(label_size))

        for i, g in enumerate(generated):
            if i % 10 == 0:
                name = f"{id:.0f}-{epoch_count:04}-{i:04}"
                save_img(g, name)
                save_wav(g, name)
        logger.info("REPEAT")


if __name__ == "__main__":
    main(LABEL)
