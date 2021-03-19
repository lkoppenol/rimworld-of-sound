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


def create_moving_spectrum_dataset(data_set):
    assert data_set in ['test', 'train', 'valid']
    data_folder = Path(f"./data/nsynth-{data_set}/")

    # TODO: what is best n_fft and what is n_fft uberhaupt?
    lekkere_columns = ['note_str']
    spectra = rimjobs.create_moving_spectrum_dataset(
        data_folder,
        n_fft=2048,
        metadata_columns=lekkere_columns
    )

    spectra.to_csv(f'data/generated/dataset_stft_{data_set}.csv')

def create_raw_audio_dataset(data_set):
    assert data_set in ['test', 'train', 'valid']
    data_folder = Path(f"./data/nsynth-{data_set}/")

    # TODO: what is best n_fft and what is n_fft uberhaupt?
    lekkere_columns = ['note_str']
    spectra = rimjobs.create_raw_audio_dataset(
        data_folder,
        metadata_columns=lekkere_columns
    )

    spectra.to_csv(f'data/generated/dataset_raw_audio_{data_set}.csv')
    
if __name__ == "__main__":
    # create_spectrum_dataset('test')
    # print("spectrum af")
    rimjobs.create_spectrum_dataset('test')
    #create_raw_audio_dataset('test')
    print("raw")
