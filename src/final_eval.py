"""Końcowa ewaluacja: model wybrany w CV (Random Forest na cechach względnych)
uczymy na CAŁYM zbiorze treningowym i oceniamy JEDEN raz na zbiorze testowym.
Uruchomienie z katalogu głównego repo:  python -m src.final_eval"""
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

from src.data import build_windows, load_raw, subject_split
from src.evaluate import subject_scores
from src.features import extract_features
from src.models import MODELS
from src.train import RESULTS_DIR

FINAL_MODEL = "random_forest"  # wybrany na podstawie CV, przed spojrzeniem na test
RELATIVE = True
THRESHOLD = 0.5  # osoba = ADHD, gdy średnie prawdopodobieństwo z jej okien >= 0,5


def main():
    X, y, groups = build_windows(load_raw())
    train_idx, test_idx = subject_split(X, y, groups)  # ten sam podział co w train.py

    model = MODELS[FINAL_MODEL]()
    model.fit(extract_features(X[train_idx], relative=RELATIVE), y[train_idx])
    proba = model.predict_proba(extract_features(X[test_idx], relative=RELATIVE))[:, 1]

    subjects = subject_scores(y[test_idx], proba, groups[test_idx])
    subjects["pred"] = (subjects["proba"] >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(subjects["y"], subjects["pred"]).ravel()

    metrics = pd.DataFrame([{
        "auc_window": roc_auc_score(y[test_idx], proba),
        "auc_subject": roc_auc_score(subjects["y"], subjects["proba"]),
        "accuracy_subject": (tp + tn) / len(subjects),
        "sensitivity_subject": tp / (tp + fn),  # ile osób z ADHD wykryto
        "specificity_subject": tn / (tn + fp),  # ile osób z grupy kontrolnej rozpoznano poprawnie
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }])
    metrics.to_csv(f"{RESULTS_DIR}/test_results.csv", index=False)
    subjects.to_csv(f"{RESULTS_DIR}/test_subjects.csv")

    print(metrics.round(3).T)


if __name__ == "__main__":
    main()
