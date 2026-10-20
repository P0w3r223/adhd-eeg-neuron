"""Definicje modeli. Każda funkcja zwraca NOWY, nienauczony model."""
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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


class TorchMLP:
    """Mała sieć w PyTorch z interfejsem jak w sklearn (fit / predict_proba),
    żeby działała w tej samej walidacji krzyżowej co pozostałe modele.
    Architektura: 95 cech -> 32 neurony (ReLU, Dropout) -> 1 wyjście (logit ADHD)."""

    def __init__(self, hidden=32, dropout=0.3, epochs=30, lr=1e-3,
                 weight_decay=1e-3, batch_size=64, random_state=RANDOM_STATE):
        self.hidden = hidden
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state)  # powtarzalne wagi startowe i tasowanie batchy

        # standaryzacja: średnia/odchylenie tylko z danych treningowych
        self.scaler = StandardScaler().fit(X)
        X_t = torch.tensor(self.scaler.transform(X), dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        self.net = nn.Sequential(
            nn.Linear(X.shape[1], self.hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, 1),
        )

        # pos_weight = odpowiednik class_weight="balanced": błędy na rzadszej klasie ważą więcej
        pos_weight = (y_t == 0).sum() / (y_t == 1).sum()
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)

        self.net.train()  # tryb treningu: Dropout włączony
        for epoch in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()                     # 1. wyzeruj gradienty z poprzedniego kroku
                logits = self.net(xb).squeeze(1)          # 2. przejście w przód: (batch, 1) -> (batch,)
                loss = loss_fn(logits, yb)                # 3. policz błąd
                loss.backward()                           # 4. policz gradienty
                optimizer.step()                          # 5. popraw wagi
        return self

    def predict_proba(self, X):
        self.net.eval()  # tryb oceny: Dropout wyłączony
        X_t = torch.tensor(self.scaler.transform(X), dtype=torch.float32)
        with torch.no_grad():  # bez liczenia gradientów – szybciej, nie uczymy
            p = torch.sigmoid(self.net(X_t).squeeze(1)).numpy()
        return np.column_stack([1 - p, p])  # kolumny jak w sklearn: [P(Control), P(ADHD)]


def make_mlp():
    return TorchMLP()


MODELS = {"logreg": make_logreg, "random_forest": make_random_forest, "mlp": make_mlp}
