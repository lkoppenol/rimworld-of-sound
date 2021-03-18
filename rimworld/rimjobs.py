from pathlib import Path

import pandas as pd
<<<<<<< HEAD
import librosa
import numpy as np
=======
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from keras.layers import Dense
from keras.models import Sequential
>>>>>>> 9060ca2d9c958431371a1425a8bdf8b796af2b87

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

<<<<<<< HEAD
def create_moving_spectrum_dataset(data_folder: Path, n_fft, metadata_columns: [str] = "note_str") -> pd.DataFrame:
    
    if type(data_folder) is str:
        data_folder = Path(data_folder)

    metadata = read_metadata(data_folder)    

    def _get_row_stft(row):
        wave_file = (data_folder / 'audio' / row.name).with_suffix(".wav")

        y, sr = librosa.load(wave_file, sr=22050)

        s = np.abs(librosa.stft(y, n_fft=n_fft)).flatten()

        s = np.around(s, decimals=0, out=None)
        
        return pd.Series(s)

    stft = metadata.apply(_get_row_stft, axis=1)
    
    return stft
=======

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
>>>>>>> 9060ca2d9c958431371a1425a8bdf8b796af2b87
