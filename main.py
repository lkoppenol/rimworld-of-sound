from rimworld import utils as rimworld_utils
from pathlib import Path


def create_spectrum_dataset(data_type):
    assert data_type in ['test', 'train', 'valid']
    data_folder = Path(f"./data/nsynth-{data_type}/")

    lekkere_columns = ['instrument', 'pitch', 'velocity']
    metadata = rimworld_utils \
        .read_metadata(data_folder) \
        [lekkere_columns]

    # TODO: what is best n_fft and what is n_fft uberhaupt?
    spectra = rimworld_utils.create_spectrum_dataset(data_folder, n_fft=2048)

    metadata \
        .join(spectra) \
        .to_csv(f'data/generated/dataset_{data_type}.csv')


if __name__ == "__main__":
    create_spectrum_dataset('test')
