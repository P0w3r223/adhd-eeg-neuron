"""Eksperyment: porównanie modeli i wariantów cech walidacją krzyżową po osobach.
Używa TYLKO zbioru treningowego – zbiór testowy zostaje nietknięty do końcowej ewaluacji.
Uruchomienie z katalogu głównego repo:  python -m src.train"""
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src.data import RANDOM_STATE, build_windows, load_raw, subject_split
from src.evaluate import subject_auc
from src.features import extract_features
from src.models import MODELS

RESULTS_DIR = "results"


def cross_validate(make_model, F, y, groups, n_splits=5):
    """Walidacja krzyżowa po osobach: w każdym foldzie uczymy NOWY model
    na 4/5 osób i oceniamy na pozostałej 1/5. Zwraca wynik każdego foldu."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for fold, (tr, va) in enumerate(cv.split(F, y, groups)):
        model = make_model()
        model.fit(F[tr], y[tr])
        proba = model.predict_proba(F[va])[:, 1]  # kolumna 1 = prawdopodobieństwo ADHD
        rows.append({
            "fold": fold,
            "auc_window": roc_auc_score(y[va], proba),
            "auc_subject": subject_auc(y[va], proba, groups[va]),
        })
    return pd.DataFrame(rows)


def main():
    X, y, groups = build_windows(load_raw())
    train_idx, _ = subject_split(X, y, groups)  # test odkładamy – tu go nie używamy
    X_tr, y_tr, g_tr = X[train_idx], y[train_idx], groups[train_idx]

    all_results = []
    for variant, relative in [("relative", True), ("absolute", False)]:
        F_tr = extract_features(X_tr, relative=relative)
        for model_name, make_model in MODELS.items():
            res = cross_validate(make_model, F_tr, y_tr, g_tr)
            res["features"] = variant
            res["model"] = model_name
            all_results.append(res)

    results = pd.concat(all_results)
    results.to_csv(f"{RESULTS_DIR}/cv_results.csv", index=False)

    summary = results.groupby(["model", "features"])[["auc_window", "auc_subject"]].agg(["mean", "std"])
    print(summary.round(3))


if __name__ == "__main__":
    main()
