from glob import glob
import os

import librosa
import numpy as np
import pandas as pd


def get_fft(path):
    a, _ = librosa.load(path, 16000)
    s = librosa.feature.melspectrogram(a, sr=16000, n_mels=64, fmin=20, fmax=8000)
    s_abs = np.abs(s)
    s_s = s_abs.sum(axis=1)
    s_rel = s_s / s_s.sum()
    f = librosa.mel_frequencies(n_mels=64, fmin=20, fmax=8000)
    _, filename = os.path.split(path)
    return {v: k for k, v in zip([filename] + list(s_rel), ['filename'] + list(f))}


ds = 'train'
pattern = f'./nsynth-{ds}.jsonwav/nsynth-{ds}/audio/*'
paths = glob(pattern)

specs = [get_fft(path) for path in paths]
pd.DataFrame.from_records(specs).to_csv(f'{ds}.csv', index=False)