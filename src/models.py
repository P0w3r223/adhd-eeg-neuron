"""Definicje modeli. Każda funkcja zwraca NOWY, nienauczony model."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import RANDOM_STATE


def make_logreg():
    """Baseline: standaryzacja + regresja logistyczna.
    Scaler w Pipeline -> średnia i odchylenie liczone tylko na danych treningowych."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000),
    )


def make_random_forest():
    """Las losowy: wiele drzew decyzyjnych, wynik = głosowanie drzew.
    Nie potrzebuje standaryzacji (drzewa porównują progi, skala nie ma znaczenia)."""
    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


MODELS = {"logreg": make_logreg, "random_forest": make_random_forest}
