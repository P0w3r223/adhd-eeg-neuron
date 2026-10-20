# Klasyfikacja ADHD z EEG

Zadanie rekrutacyjne do sekcji AI koła naukowego **KN Neuron** (Politechnika Wrocławska).
Binarny klasyfikator ADHD vs grupa kontrolna na podstawie sygnałów EEG, z **podziałem po osobach (subject split)**.

> Status: EDA, preprocessing, podział, cechy i porównanie modeli walidacją krzyżową są gotowe.
> W trakcie: końcowa ewaluacja na zbiorze testowym.

## Dane

[EEG Dataset for ADHD](https://www.kaggle.com/datasets/danizo/eeg-dataset-for-adhd) (Kaggle, oryginalnie IEEE DataPort, DOI 10.21227/rzfh-zn36).

- 121 dzieci w wieku 7–12 lat: **61 ADHD i 60 kontroli**
- 19 kanałów w układzie 10-20, **128 Hz**, referencja na uszach
- nagrania wykonane podczas **zadania uwagi wzrokowej** (liczenie postaci na obrazkach), a nie w spoczynku
- jeden plik `adhdata.csv` (~267 MB): wiersz to jedna próbka czasu, kolumny to kanały oraz `Class` i `ID`

## Uruchomienie

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

1. Pobierz `adhdata.csv` z Kaggle i umieść w `data/raw/adhdata.csv`. Plik nie jest w repo.
2. EDA: otwórz `notebooks/eda.ipynb` z kernelem z `.venv`. Wykresy zapisują się do `results/`.
3. Porównanie modeli walidacją krzyżową (z katalogu głównego repo):

```bash
python -m src.train
```

Wynik trafia do `results/cv_results.csv`. Wszystkie losowe kroki używają `RANDOM_STATE = 42`.

## Struktura

```
src/data.py       wczytanie CSV, filtr, okienkowanie, odrzucanie artefaktów, podział po osobach
src/features.py   moc w pasmach (Welch) + theta/beta -> 95 cech na okno
src/models.py     regresja logistyczna, Random Forest, MLP (PyTorch)
src/train.py      eksperyment: CV po osobach na zbiorze treningowym
src/evaluate.py   metryki, agregacja okien do poziomu osoby
notebooks/eda.ipynb   tylko analiza eksploracyjna
results/          wykresy i wyniki
```

## Pipeline i uzasadnienie decyzji

| Krok | Decyzja | Dlaczego |
|---|---|---|
| Filtr | Butterworth 1–40 Hz, 4. rząd, `sosfiltfilt`, osobno dla każdej osoby | usuwa offset (~+140) i sieć 50 Hz, której było **więcej u Control** (ryzyko, że model nauczy się szumu sieciowego zamiast ADHD); zero-phase; postać SOS jest numerycznie stabilna |
| Okna | 4 s (512 próbek), bez nakładania | pojedyncza próbka nic nie znaczy; 4 s wystarcza na kilka okresów delty; bez nakładania nie powielamy danych |
| Artefakty | okno odrzucone, gdy max\|x\| > 3 × mediana max\|x\| **tej osoby** | jednostki w pliku to nie µV, więc progi z literatury nie pasują (próg 500 odrzucał 90% okien); skala amplitudy różni się między osobami ~6×. Odrzucono 2,4% okien ADHD i 3,0% okien Control |
| Podział | `StratifiedGroupKFold` po `ID`: test = 25 osób (13 ADHD / 12 Control), train = 96 | dane jednej osoby nigdy nie trafiają jednocześnie do train i test (sprawdza to `assert`); stratyfikacja chroni przed losowym niebalansem małego testu |
| Cechy | log10 mocy względnej w pasmach delta/theta/alpha/beta + theta/beta, dla 19 kanałów (95 cech) | moc względna jest odporna na różną „głośność” osób; theta/beta to klasyczny, choć sporny marker ADHD |
| Modele | regresja logistyczna (baseline), Random Forest, MLP w PyTorch | od najprostszego do nieliniowych; wszystkie z wyważeniem klas (okien ADHD jest ~56%) |
| Walidacja | 5-krotna CV po osobach **tylko na train**; test użyty raz, na końcu | wybór modelu na podstawie testu zawyżyłby wynik |

## Wyniki walidacji krzyżowej (ROC-AUC, średnia ± odchylenie z 5 foldów)

| Model | Cechy | AUC okien | AUC osób |
|---|---|---|---|
| Regresja logistyczna | względne | 0,502 ± 0,092 | 0,479 ± 0,124 |
| Regresja logistyczna | bezwzględne | 0,559 ± 0,116 | 0,574 ± 0,195 |
| Random Forest | względne | 0,573 ± 0,127 | 0,619 ± 0,196 |
| Random Forest | bezwzględne | 0,588 ± 0,091 | 0,600 ± 0,146 |
| MLP | względne | 0,575 ± 0,087 | 0,636 ± 0,108 |
| MLP | bezwzględne | 0,604 ± 0,105 | 0,633 ± 0,162 |

Wnioski:

- Przy uczciwym podziale po osobach wyniki są **niewiele powyżej losowych**. Prace raportujące na tym zbiorze accuracy powyżej 95% zwykle dzielą okna, a nie osoby.
- Odchylenie między foldami (0,1–0,2) jest większe niż różnice między modelami. Na tych danych nie da się wiarygodnie wskazać zwycięzcy.
- Jako model główny wybrano **Random Forest na cechach względnych**, przed spojrzeniem na test.

## Obserwacje z EDA i ograniczenia

- Nagrania ADHD są dłuższe (średnio 155 s wobec 125 s), bo długość zależała od tempa dziecka. Daje to więcej okien ADHD.
- Theta/beta wyszło **niższe** u ADHD, odwrotnie niż w klasycznej hipotezie. Możliwe przyczyny: zadanie zamiast spoczynku, Ritalin w grupie ADHD, artefakty.
- Dzieci z ADHD przyjmowały Ritalin, co może zakłócać wnioski (efekt leku zamiast efektu ADHD).
- Wartości są skwantowane (skok ~36) i nie są w µV. Część kanałów osiąga nasycenie.
- Tylko 121 osób, więc wszystkie metryki mają duży rozrzut.
