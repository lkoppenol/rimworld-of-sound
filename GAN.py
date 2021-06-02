from gan.discriminator import MultiLabelDiscriminator
from gan.gan import Gan
from gan.generator import BaseGenerator
from loguru import logger
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from rimworld.utils import *
import soundfile as sf

load_dotenv()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DEBUG_MODE = False
# create .env in root folder, for example:
# ROOT_FOLDER=D:\Projects\nsynth-data\data\stft
# TEST_SOUNDS_FOLDER = D:\Projects\nsynth-data\data\nsynth-test\audio
ROOT_FOLDER = os.getenv('ROOT_FOLDER')
TEST_SOUNDS_FOLDER = os.getenv('TEST_SOUNDS_FOLDER')


if DEBUG_MODE:
    print(ROOT_FOLDER)
# LABEL = 'instrument_subtype'
LABEL_ID = 'instrument_and_pitch_single_label'
BATCH_SIZE =32
EPOCHS = 200
SAMPLE_RATE = 16000

generator_description = "last_day_of_may"

DEBUG_MODE = True

if DEBUG_MODE:
    logger.warning("WARNING WARNING DEBUG MODE IS ON WARNING WARNING")


def main():
    # Required folder structure:
    # ROOT_FOLDER\train\anything\all_your_imgs.png
    # ROOT_FOLDER\valid\anything\all_your_imgs.png

    # Get some sound samples for checking how well the generator performs, choose manually based on classifier
    if DEBUG_MODE:
        logger.warning("DON'T FORGET TO MANUALLY LOAD SENSIBLE SAMPLES FOR INTER-EPOCHIAL-SOUNDCHECKS")
    waves = ["bass_electronic_018-030-050.wav",
             "organ_electronic_007-037-050.wav",
             "string_acoustic_071-039-127.wav",
             "vocal_acoustic_000-065-100.wav"]
    test_samples = []
    for wave in waves:
        test_samples.append(os.path.join(TEST_SOUNDS_FOLDER, wave))

    test_labels = []
    for sample_name in waves:
        test_labels.append(np.argmax(np.array(get_label(sample_name, LABEL_ID, label_shapes[LABEL_ID]))))



    if DEBUG_MODE:
        import time
        print("sample: {}".format(test_samples[0]))
        print("corresponding label: {}".format(test_labels[0]))

    train_folder = os.path.join(ROOT_FOLDER, 'train')
    train_dataset = get_image_dataset(train_folder, LABEL_ID, label_shapes[LABEL_ID], BATCH_SIZE)
    valid_folder = os.path.join(ROOT_FOLDER, 'valid')
    valid_dataset = get_image_dataset(valid_folder, LABEL_ID, label_shapes[LABEL_ID], BATCH_SIZE)

    if DEBUG_MODE:
        subsample = 10
        logger.warning(f"WARNING DEBUG MODE TAKING TRAINING SAMPLE OF {subsample} B*TCHES")
        train_dataset = train_dataset.take(subsample)
        valid_dataset = valid_dataset.take(subsample)

    discriminator = MultiLabelDiscriminator(
        input_shape=(126, 1025, 1),
        num_classes=label_shapes[LABEL_ID]
    )
    discriminator.load(r"notebooks/instrument_and_pitch_single_label_model_1621681850")
    #discriminator.fit(train_dataset, epochs=EPOCHS, validation_data=valid_dataset)
    #discriminator.save(f'models/model_{int(time())}')

    generator = BaseGenerator(
        input_length=label_shapes[LABEL_ID],
        output_shape=(126, 1025, 1)
    )

    # GAN MET DIE BANAN
    gan = Gan.from_couple(generator, discriminator)

    epoch_step = 1
    epoch_max = 200
    for epoch_count in range(0, epoch_max, epoch_step):
        gan.fit(
            sample_size=int(2e3),
            batch_size=int(1e1),
            epochs=epoch_step,
            validation_split=0.1
        )
        gan.get_generator().save(f'models/{generator_description}/epoch_{epoch_count:05}/generator.h5')
        gan.save(f'models/{generator_description}/epoch_{epoch_count:05}/gan.h5')
        generated = gan.generate(test_labels, False, False)
        spectrograms = gan.generate(test_labels, False, True)

        images_dir = f'models/{generator_description}/epoch_{epoch_count:05}/images'
        os.mkdir(images_dir)
        for idx, s in enumerate(spectrograms):
            plt.imshow(s, cmap='gray')
            plt.savefig(f'{images_dir}/{waves[idx][:-8]}.png')

        sounds_dir = f'models/{generator_description}/epoch_{epoch_count:05}/sounds'
        os.mkdir(sounds_dir)
        for idx in range(len(waves)):
            s = reconstruct_from_sliding_spectrum(generated[idx, :, :, 0])
            sf.write(f'{sounds_dir}/{waves[idx][:-8]}.wav', s,
                     SAMPLE_RATE)


if __name__ == "__main__":
    main()
