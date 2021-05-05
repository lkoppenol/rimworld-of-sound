import os
from glob import glob
from pathlib import Path

import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from keras.layers import Dense
from keras.models import Sequential

from rimworld.utils import read_metadata
from rimworld.rimsound import RimSound

from loguru import logger


def create_spectrum_dataset(data_folder: Path, n_fft, metadata_columns: [str] = None) -> pd.DataFrame:
    """
    Create a dataset with fourier spectra for each sound-id, based on a metadata file

    :param data_folder: root folder of dataset, for example `Path('./data/nsynth-test')`
    :param n_fft: see [stft](https://librosa.org/doc/0.8.0/generated/librosa.stft.html)
    :param metadata_columns: list of metadata columns to keep
    :return: pandas DataFrame, a row for each sound-id, a column for each frequency. value is relative amplitude
    """
    if type(data_folder) is str:
        data_folder = Path(data_folder)

    metadata = read_metadata(data_folder)

    def _get_row_spectrum(row):
        wave_file = (data_folder / 'audio' / row.name).with_suffix(".wav")
        spectrum = RimSound \
            .from_wav(wave_file, row.sample_rate, n_fft=n_fft) \
            .get_spectrum()
        return spectrum

    spectra = metadata.apply(_get_row_spectrum, axis=1)

    if metadata_columns:
        spectra = metadata \
            [metadata_columns] \
            .join(spectra)

    return spectra


def train_classifier_instrumentfamily_pitch(data_folder: Path):
    # Multi-task learning
    n = 12_000
    split = int(n * 0.8)
    phase = 'valid'

    df = pd.read_csv(data_folder / f'dataset_{phase}.csv', index_col=0, nrows=n)

    target_cols = ['pitch', 'instrument_family']
    target = df[target_cols]
    target_vector = pd.get_dummies(target, columns=target_cols)
    df.drop(['velocity', 'instrument_family', 'pitch'], axis=1, inplace=True)

    dataset_train = tf.data \
        .Dataset \
        .from_tensor_slices(
            (
                df.iloc[:split],
                target_vector[:split]
            )) \
        .shuffle(1_000) \
        .batch(128)

    dataset_test = tf.data \
        .Dataset \
        .from_tensor_slices(
            (
                df.iloc[split:],
                target_vector[split:]
            )) \
        .shuffle(1_000) \
        .batch(128)

    model = Sequential()
    model.add(Dense(64, input_dim=df.shape[-1], activation='relu'))
    model.add(Dense(target_vector.shape[-1], activation='sigmoid'))

    loss = 'binary_crossentropy'
    model.compile(
        loss=loss,
        optimizer=tf.optimizers.Adam(),
        metrics=['accuracy']
    )

    model.fit(dataset_train, batch_size=15, epochs=250, validation_data=dataset_test, verbose=0)

    return model


def create_stft_dataset(folder_in: str, folder_out: str, sample_rate=16_000):
    wav_names = glob(os.path.join(folder_in, '*.wav'))
    for wav_name in wav_names:
        spectrum = RimSound \
            .from_wav(wav_name, sample_rate) \
            .get_short_time_spectrum()
        path_out = os.path.join(folder_out, os.path.split(wav_name)[-1] + '.png')
        logger.info(f"writing {path_out}")
        plt.imsave(path_out, spectrum.T, cmap='gray')
