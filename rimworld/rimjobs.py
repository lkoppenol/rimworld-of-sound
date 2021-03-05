from pathlib import Path

import pandas as pd

from rimworld.utils import read_metadata
from rimworld.rimsound import RimSound


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
        spectra = spectra.join(metadata[metadata_columns])

    return spectra
