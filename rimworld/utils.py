import pandas as pd
import numpy as np
from pathlib import Path


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

