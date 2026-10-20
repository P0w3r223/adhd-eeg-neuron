"""Ewaluacja: metryki na poziomie okien i osób."""
import pandas as pd
from sklearn.metrics import roc_auc_score


def subject_scores(y, proba, groups):
    """Łączy okna w osoby: wynik osoby = średnie prawdopodobieństwo ADHD z jej okien.
    Zwraca DataFrame: jeden wiersz na osobę, kolumny y (prawdziwa klasa) i proba."""
    df = pd.DataFrame({"subject": groups, "y": y, "proba": proba})
    return df.groupby("subject").agg(y=("y", "first"), proba=("proba", "mean"))


def subject_auc(y, proba, groups):
    """ROC-AUC liczony na osobach zamiast na oknach."""
    s = subject_scores(y, proba, groups)
    return roc_auc_score(s["y"], s["proba"])
