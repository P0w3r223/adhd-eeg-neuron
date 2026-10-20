"""Cechy z okien EEG: moc w pasmach (Welch) i stosunek theta/beta."""
import numpy as np
from scipy.signal import welch

from src.data import FS

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}  # [Hz]


def window_psd(X, fs=FS):
    """Widmo mocy (Welch) dla każdego okna i kanału.
    X: (okna, próbki, kanały) -> f: (częstotliwości,), psd: (okna, częstotliwości, kanały)."""
    f, psd = welch(X, fs=fs, nperseg=2 * fs, axis=1)  # segmenty 2 s -> rozdzielczość 0.5 Hz
    return f, psd


def band_powers(f, psd):
    """Moc w każdym paśmie = suma widma w zakresie częstotliwości pasma.
    Wynik: (okna, pasma, kanały) – pasma w kolejności z BANDS."""
    powers = []
    for low, high in BANDS.values():
        in_band = (f >= low) & (f < high)              # maska: które częstotliwości należą do pasma
        powers.append(psd[:, in_band, :].sum(axis=1))  # suma po częstotliwościach -> (okna, kanały)
    return np.stack(powers, axis=1)


def extract_features(X, relative=True):
    """Macierz cech (okna, 95): 4 pasma x 19 kanałów + theta/beta x 19 kanałów, w skali log10.
    relative=True  -> moc pasma / moc 1-30 Hz (odporna na różną „głośność” osób),
    relative=False -> moc bezwzględna."""
    f, psd = window_psd(X)
    bp = band_powers(f, psd)                      # (okna, 4, 19)

    theta_beta = bp[:, 1, :] / bp[:, 3, :]        # pasmo 1 = theta, 3 = beta -> (okna, 19)

    if relative:
        total = bp.sum(axis=1, keepdims=True)     # 4 pasma razem = 1-30 Hz -> (okna, 1, 19)
        bp = bp / total

    n_windows = len(X)
    features = np.concatenate([bp.reshape(n_windows, -1), theta_beta], axis=1)
    return np.log10(features)


def feature_names(channels):
    """Nazwy 95 cech w tej samej kolejności co kolumny extract_features."""
    names = [f"{band}_{ch}" for band in BANDS for ch in channels]
    names += [f"theta_beta_{ch}" for ch in channels]
    return names
