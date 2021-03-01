from rimworld import rimjobs
from pathlib import Path


def create_spectrum_dataset(data_type):
    assert data_type in ['test', 'train', 'valid']
    data_folder = Path(f"./data/nsynth-{data_type}/")

    # TODO: what is best n_fft and what is n_fft uberhaupt?
    lekkere_columns = ['instrument', 'pitch', 'velocity']
    spectra = rimjobs.create_spectrum_dataset(
        data_folder,
        n_fft=2048,
        metadata_columns=lekkere_columns
    )

    spectra.to_csv(f'data/generated/dataset_{data_type}.csv')


if __name__ == "__main__":
    create_spectrum_dataset('test')
    # create_spectrum_dataset('valid')
