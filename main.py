from rimworld import utils as rimworld_utils
from pathlib import Path


def create_test_dataset():
    test_path = Path("./data/nsynth-test/")
    rimworld_utils.create_spectrum_dataset(test_path, n_fft=2048)


if __name__ == "__main__":
    create_test_dataset()
