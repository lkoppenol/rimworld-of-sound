from glob import glob
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import librosa
import skimage.io
import numpy


load_dotenv()

sr = 16000

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def create_mel_dataset(folder_in: str, folder_out: str, sample_rate=16_000):
    wav_names = glob(os.path.join(folder_in, '*.wav'))
    print(len(wav_names))
    for wav_name in wav_names:
        a, _ = librosa.load(wav_name, sr)
        s = librosa.feature.melspectrogram(a, sr=sr, n_mels=64, fmin=20, fmax=8000)
        mels = numpy.log(s + 1e-9)  # add small number to avoid log(0)

        # min-max scale to fit inside 8-bit range
        img = scale_minmax(mels, 0, 255).astype(numpy.uint8)
        img = numpy.flip(img, axis=0)  # put low frequencies at the bottom in image
        img = 255 - img  # invert. make black==more energy
        path_out = os.path.join(folder_out, os.path.split(wav_name)[-1] + '.png')
        skimage.io.imsave(path_out, img)

ds = 'train'


folder_in = os.getenv("TRAIN_SOUNDS_FOLDER")
folder_out = os.getenv("MEL_FREQS_ROOT") + f"/{ds}/{ds}_fortf"

create_mel_dataset(folder_in, folder_out)

