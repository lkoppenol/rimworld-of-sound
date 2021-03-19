from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import librosa
import pandas as pd


@dataclass
class RimSound:
    raw_audio: np.ndarray
    sample_rate: int
    n_fft: int = 2048

    @classmethod
    def from_wav(cls, path, sample_rate, **kwargs):
        raw_audio, sample_rate = librosa.load(path, sr=sample_rate)
        rim_sound = cls(
            raw_audio=raw_audio,
            sample_rate=sample_rate,
            **kwargs
        )
        return rim_sound

    def get_spectrum(self) -> pd.Series:
        """
        Converts raw audio (audiobuffer?) to a single spectrum by taking the sum for every bin over all units of time.
        TODO: function for FFT instead of STFT

        :return: pandas Series, index is frequency, value is relative strength
        """
        short_time_spectrum_i = librosa.stft(
            self.raw_audio,
            n_fft=self.n_fft
        )
        short_time_spectrum = np.abs(short_time_spectrum_i)
        raw_spectrum = short_time_spectrum.sum(axis=1)
        frequencies = librosa.fft_frequencies(
            sr=self.sample_rate,
            n_fft=self.n_fft
        )
        spectrum = pd.Series(raw_spectrum, frequencies)
        return spectrum
