"""Wczytywanie surowych danych EEG (Kaggle: danizo/eeg-dataset-for-adhd)."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from sklearn.model_selection import StratifiedGroupKFold

RANDOM_STATE = 42  # stałe ziarno losowości: ten sam podział przy każdym uruchomieniu
FS = 128  # częstotliwość próbkowania [Hz] wg opisu zbioru
RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "adhdata.csv"
META_COLS = ["Class", "ID"]


def load_raw(path=RAW_PATH):
    """Wczytuje CSV: jeden wiersz = jedna próbka czasu ze wszystkich kanałów."""
    return pd.read_csv(path)


def channel_names(df):
    """Nazwy kanałów EEG = wszystkie kolumny poza Class i ID (bierzemy z pliku, nie z opisu)."""
    return [c for c in df.columns if c not in META_COLS]


def bandpass(x, fs=FS, low=1.0, high=40.0, order=4):
    """Filtr pasmowy Butterwortha bez przesunięcia fazowego.
    x: tablica (próbki, kanały) jednej osoby."""
    sos = butter(order, [low, high], btype="bandpass", output="sos", fs=fs)
    return sosfiltfilt(sos, x, axis=0)  # axis=0: filtrujemy wzdłuż czasu, każdy kanał osobno


def make_windows(x, fs=FS, win_sec=4):
    """Tnie sygnał jednej osoby na rozłączne okna o długości win_sec sekund.
    x: tablica (próbki, kanały) -> wynik: tablica (okna, próbki_w_oknie, kanały).
    Końcówka krótsza niż jedno okno jest odrzucana."""
    win_len = int(win_sec * fs)       # 4 s * 128 Hz = 512 próbek
    n_windows = len(x) // win_len     # ile pełnych okien się mieści
    windows = []
    for i in range(n_windows):
        start = i * win_len
        windows.append(x[start:start + win_len])
    return np.array(windows)


def artifact_mask(windows, k=3):
    """Zwraca maskę True/False: które okna jednej osoby zostawić.
    Okno odrzucamy, gdy jego największa amplituda przekracza k razy
    medianę największych amplitud wszystkich okien tej osoby.
    windows: tablica (okna, próbki, kanały) -> wynik: tablica (okna,) typu bool."""
    peak = np.abs(windows).max(axis=(1, 2))  # max |x| w każdym oknie (po czasie i kanałach)
    threshold = k * np.median(peak)          # próg względny, osobny dla każdej osoby
    return peak <= threshold


def build_windows(df, win_sec=4, k=3):
    """Cały pipeline dla każdej osoby: filtr -> okna -> odrzucenie artefaktów.
    Zwraca:
      X      (okna, próbki, kanały) – sygnał,
      y      (okna,) – etykieta: 1 = ADHD, 0 = Control,
      groups (okna,) – ID osoby, do podziału po osobach (GroupKFold)."""
    ch = channel_names(df)
    X_list, y_list, groups_list = [], [], []

    for subject_id, g in df.groupby("ID", sort=False):
        x = bandpass(g[ch].to_numpy())
        windows = make_windows(x, win_sec=win_sec)
        windows = windows[artifact_mask(windows, k=k)]

        label = 1 if g["Class"].iloc[0] == "ADHD" else 0
        X_list.append(windows)
        y_list.append(np.full(len(windows), label))
        groups_list.append(np.full(len(windows), subject_id))

    X = np.concatenate(X_list).astype(np.float32)  # float32: o połowę mniej pamięci
    y = np.concatenate(y_list)
    groups = np.concatenate(groups_list)
    return X, y, groups


def subject_split(X, y, groups, n_splits=5, random_state=RANDOM_STATE):
    """Podział train/test PO OSOBACH: wszystkie okna danej osoby trafiają
    w całości albo do train, albo do test. Test = 1 z n_splits części osób
    (n_splits=5 -> ok. 20% osób), z zachowaniem proporcji klas.
    Zwraca indeksy okien: train_idx, test_idx."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))  # bierzemy pierwszą z 5 części jako test

    # zabezpieczenie: żadna osoba nie może być jednocześnie w train i test
    assert set(groups[train_idx]).isdisjoint(groups[test_idx]), "Ta sama osoba w train i test!"
    return train_idx, test_idx
