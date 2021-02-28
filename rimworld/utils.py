import pandas as pd
import numpy as np
from pathlib import Path
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


def raw_audio_to_spectrum(raw_audio: [float], sample_rate: int, n_fft: int) -> pd.Series:
    """
    Converts raw audio (audiobuffer?) to a single spectrum. TODO: function for FFT instead of STFT

    :param raw_audio: for example returned by `librosa.load(path)`
    :param sample_rate: for example returned by `librosa.load(path)`
    :param n_fft: see [stft](https://librosa.org/doc/0.8.0/generated/librosa.stft.html)
    :return: pandas Series, index is frequency, value is relative strength
    """
    short_time_spectrum_i = librosa.stft(raw_audio, n_fft=n_fft)
    short_time_spectrum = np.abs(short_time_spectrum_i)
    raw_spectrum = short_time_spectrum.sum(axis=1)
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    spectrum = pd.Series(raw_spectrum, frequencies)
    return spectrum


def wavefile_to_spectrum(path: Path, sample_rate: int, n_fft: int) -> pd.Series:
    """
    Convert a wavefile to a fourier spectrum

    :param path: path object to wavefile
    :param sample_rate: sample rate of audio file
    :param n_fft: see [stft](https://librosa.org/doc/0.8.0/generated/librosa.stft.html)
    :return: pandas Series, index is frequency, value is relative strength
    """
    raw_audio, sample_rate = librosa.load(path, sr=sample_rate)
    spectrum = raw_audio_to_spectrum(raw_audio, sample_rate=sample_rate, n_fft=n_fft)
    return spectrum


def create_spectrum_dataset(data_folder: Path, n_fft) -> pd.DataFrame:
    """
    Create a dataset with fourier spectra for each sound-id, based on a metadata file

    :param data_folder: root folder of dataset, for example `Path('./data/nsynth-test')`
    :param n_fft: see [stft](https://librosa.org/doc/0.8.0/generated/librosa.stft.html)
    :return: pandas DataFrame, a row for each sound-id, a column for each frequency. value is relative amplitude
    """
    if type(data_folder) is str:
        data_folder = Path(data_folder)

    metadata = read_metadata(data_folder)

    def _get_row_spectrum(row):
        wave_file = (data_folder / 'audio' / row.name).with_suffix(".wav")
        return wavefile_to_spectrum(wave_file, row.sample_rate, n_fft)

    spectra = metadata.apply(_get_row_spectrum, axis=1)
    return spectra
