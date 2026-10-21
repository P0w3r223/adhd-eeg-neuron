# Status projektu

Date: 2026-10-21
Status: accepted
Author: P0w3r223 + Claude

---

## Etap

Projekt zamknięty merytorycznie. Wszystkie kroki pipeline'u są gotowe:
EDA → preprocessing → podział po osobach → cechy → CV modeli → ewaluacja na teście.

## Wyniki

- CV (5 foldów po osobach, tylko train): najlepsze AUC osób ~0,62–0,64, rozrzut między foldami 0,1–0,2.
- Test (25 osób, Random Forest na cechach względnych, użyty raz): AUC osób 0,744, accuracy 0,64.
- Szczegóły i interpretacja: `README.md`.

## Następny krok

Omówienie kodu plik po pliku i quiz przed rozmową rekrutacyjną. Bez zmian w kodzie.
