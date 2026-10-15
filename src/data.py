"""Wczytywanie surowych danych EEG (Kaggle: danizo/eeg-dataset-for-adhd)."""
from pathlib import Path

import pandas as pd

FS = 128  # częstotliwość próbkowania [Hz] wg opisu zbioru
RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "adhdata.csv"
META_COLS = ["Class", "ID"]


def load_raw(path=RAW_PATH):
    """Wczytuje CSV: jeden wiersz = jedna próbka czasu ze wszystkich kanałów."""
    return pd.read_csv(path)


def channel_names(df):
    """Nazwy kanałów EEG = wszystkie kolumny poza Class i ID (bierzemy z pliku, nie z opisu)."""
    return [c for c in df.columns if c not in META_COLS]
