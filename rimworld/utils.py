import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf
import os
import librosa


def read_metadata(data_folder: Path, instrument_filter: str=None, filename: str="examples.json") -> pd.DataFrame:
    """
    Read an NSynth metadatafile from disk as pandas DataFrame.

    :param data_folder: root folder of dataset, for example `Path('./data/nsynth-test')`
    :param instrument_filter: exact name of instrument_str, Falsy reads all
    :param filename: default = "examples.json"
    :return: pandas DataFrame with sound-id as index
    """
    if type(data_folder) is str:
        data_folder = Path(data_folder)

    metadata_file = data_folder / filename
    metadata = pd \
        .read_json(metadata_file, orient='index')

    if instrument_filter:
        metadata = metadata.query('instrument_str == @INSTRUMENT')

    return metadata


label_shapes = dict(
    instrument=1,
    instrument_subtype=33,
    pitch=128,
    instrument_subtype_and_pitch=5+112,  # 4 instruments, 1 other, 112 pitches
    instrument_and_pitch_single_label=5*112,  # same but then single label so more labels
    no_organ=2,
    organ_pitch=129,
)


def get_label(filename, label_type, label_size):
    switch = {
        "instrument": get_instrument_label,
        "instrument_subtype": get_instrument_subtype_label,
        "pitch": get_pitch_label,
        "instrument_subtype_and_pitch": get_multi_label,
        "instrument_and_pitch_single_label": get_instrument_and_pitch,
        "no_organ": get_no_organ_label,
        "organ_pitch": get_organ_pitch_label,
    }
    # just a hacky solution to cope with the fact that we havent taken the effort yet to make this code clean
    # but still be able to add more label methods
    sparse_labels = ["instrument_and_pitch_single_label", "instrument_subtype_and_pitch"]
    if label_type not in sparse_labels:
        sparse_label = switch[label_type](filename)
        label = np.zeros((label_size, 1))
        label[sparse_label] = 1
    else:
        label = switch[label_type](filename)
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


def get_no_organ_label(filename):
    instrument = "_".join(filename.split('_')[:-2])
    if instrument == 'organ':
        return 1
    else:
        return 0


def get_organ_pitch_label(filename):
    return get_no_organ_label(filename) * get_pitch_label(filename)


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


def get_multi_label(filename):
    """
    multi label with best recognizable instruments bass_electronic, vocal acoustic, organ electronic, string acoustic,
    other_instruments, noise(when added to dataset), pitch
    """

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
    
    return label


def reset(batch_size, label_size):
    imgs = np.zeros((batch_size, 126, 1025, 1))
    labels = np.zeros((batch_size, label_size))
    return imgs, labels


def get_image_dataset(path, label_type, label_size, batch_size):
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


def reconstruct_from_sliding_spectrum(S_abs):
    return librosa.core.spectrum.griffinlim(S_abs)

# def get_n_samples_for_storing_generator_results(path, n=5, random=True):
#     funky_extra_folder_for_tensorflow_image_dataset_function = os.walk(path).next()[1]
#     print(funky_extra_folder_for_tensorflow_image_dataset_function)
#
#     files = np.zeros(n)
#     if random:
#         for i in range(n):
#             files[i] = random.choice(os.listdir(path))
#     else:
#
#
#     assert len(files) is not 0
#
#     return files



