# Riassunto Analisi Dati Cinematici — VRula BodyTracking Study

**Script di riferimento:** `separated_scripts/analisi_dati.py`
**Dataset:** `output_analysis/analysis_dataset.csv`
**Data aggiornamento:** 2026-05-05

---

## 1. Dataset

| Parametro | Valore |
|---|---|
| N soggetti | 18 |
| Condizioni | baseline, workload |
| Righe totali | 36 (18 × 2) |
| Design | Within-subject controbilanciato A/B |
| Gruppo A→B (baseline prima) | n = 8 |
| Gruppo B→A (workload prima) | n = 10 |

### Variabili analizzate (21 totali)

**Movimento del polso dominante (8):**
`SPARC_Dom`, `Jerk_Dom`, `VelInv_Dom`, `Dwell_Dom`, `Vel_Mean_Dom`, `Vel_Peak_Dom`, `Vel_SD_Dom`, `PathLen_Dom`

**Posturali / angolari (13):**
`RULA_Mean`, `Wr_Flex_Mean_Dom`, `Wr_Flex_SD_Dom`, `Wr_Dev_Mean_Dom`, `Wr_Dev_SD_Dom`, `UA_Mean_Dom`, `SD_UA_Dom`, `LA_Mean_Dom`, `SD_LA_Dom`, `Neck_Sag_Mean`, `Neck_Sag_SD`, `Trunk_Sag_Mean`, `Trunk_Sag_SD`

---

## 2. Statistiche descrittive

Per ogni variabile e per ogni condizione separatamente sono stati calcolati: n, media, deviazione standard, minimo, Q1, mediana, Q3, massimo.

**Output:** stampa a console, non salvato in CSV.

---

## 3. Rilevamento e sostituzione outlier

**Criterio:** ±2 deviazioni standard, calcolate **per condizione separatamente** (non sul dataset pooled). Due distribuzioni di riferimento indipendenti: una per baseline, una per workload.

**Procedura per ogni variabile e condizione:**
1. Calcolo di media µ e SD σ sui valori non-NaN della condizione
2. Soglie: lower = µ − 2σ, upper = µ + 2σ
3. Valori fuori range → sostituiti con la **media dei non-outlier** della stessa condizione
4. Colonne int64 convertite a float64 prima della sostituzione per evitare errori di tipo

**Dataset pulito salvato in:** `output_analysis/analysis_dataset_clean.csv`

---

## 4. Visualizzazioni

### `boxplots_cinematiche.png`
Boxplot per tutte e 21 le variabili sul **dataset originale** (prima della pulizia outlier).
- Punti individuali sovrapposti con jitter (colore = condizione)
- Linee rosse tratteggiate = soglie ±2 SD per condizione
- Layout a griglia, baseline e workload affiancati per variabile

### `sparc_jerk_overview.png`
Approfondimento su SPARC e Jerk con 3 tipi di grafici in subplot (dataset pulito):
- Boxplot baseline vs workload
- Paired lines (ogni linea = un soggetto)
- Violin plot

### `paired_lines_cinematiche.png`
Paired lines per tutte e 21 le variabili (dataset pulito): ogni linea connette baseline e workload dello stesso soggetto, permettendo di vedere la direzione e la consistenza dell'effetto individuale.

### `qqplot_delta.png`
Q-Q plot dei delta scores (workload − baseline) per ogni variabile, con la retta teorica normale di riferimento.

---

## 5. Test di normalità (Shapiro-Wilk)

Il test di Shapiro-Wilk è stato applicato su due livelli:

**A) Delta scores** (workload − baseline, N=18):
Verifica dell'assunzione per il paired t-test. I delta sono la quantità rilevante per i test inferenziali accoppiati.

**B) Per condizione** (N=18 per condizione):
Analisi descrittiva aggiuntiva della distribuzione dei valori grezzi in baseline e workload separatamente.

**Output:** stampa a console con W, p e interpretazione. Q-Q plot in `qqplot_delta.png`.

---

## 6. Effetto condizione (baseline vs workload)

### Scelta del test
Per ogni variabile, la scelta tra **paired t-test** e **Wilcoxon signed-rank** è basata sul risultato dello Shapiro-Wilk sui delta:
- Delta normali (p > 0.05) → paired t-test (Cohen's d_z come effect size)
- Delta non normali (p ≤ 0.05) → Wilcoxon signed-rank (rank-biserial r come effect size)

### Correzione per comparazioni multiple
**Benjamini-Hochberg FDR** (via `statsmodels.multipletests`) applicata sui 21 p-value grezzi.
Soglia di significatività: q_BH < 0.05.

### Risultati

| Variabile | p grezzo | p FDR | Sig? |
|---|---|---|---|
| **SPARC_Dom** | 0.0047 | 0.0231 | ✓ |
| **Jerk_Dom** | 0.0070 | 0.0231 | ✓ |
| VelInv_Dom | 0.5763 | 0.6370 | |
| **Dwell_Dom** | 0.0001 | 0.0007 | ✓ |
| **Vel_Mean_Dom** | <0.0001 | <0.0001 | ✓ |
| Vel_Peak_Dom | 0.1330 | 0.2539 | |
| **Vel_SD_Dom** | 0.0170 | 0.0446 | ✓ |
| **PathLen_Dom** | 0.0001 | 0.0007 | ✓ |
| RULA_Mean | 0.7697 | 0.7697 | |
| Wr_Flex_Mean_Dom | 0.3724 | 0.5214 | |
| Wr_Flex_SD_Dom | 0.6384 | 0.6703 | |
| Wr_Dev_Mean_Dom | 0.1994 | 0.3221 | |
| Wr_Dev_SD_Dom | 0.0518 | 0.1209 | |
| UA_Mean_Dom | 0.4956 | 0.6122 | |
| SD_UA_Dom | 0.0960 | 0.2016 | |
| LA_Mean_Dom | 0.3445 | 0.5168 | |
| **SD_LA_Dom** | 0.0056 | 0.0231 | ✓ |
| Neck_Sag_Mean | 0.4302 | 0.5646 | |
| **Neck_Sag_SD** | 0.0077 | 0.0231 | ✓ |
| Trunk_Sag_Mean | 0.5524 | 0.6370 | |
| Trunk_Sag_SD | 0.1935 | 0.3221 | |

**Significative dopo FDR: 8/21**

Le variabili significative riguardano la fluidità e la velocità del movimento (SPARC, Jerk, Dwell, Vel_Mean, Vel_SD, PathLen) e la variabilità posturale distale (SD_LA, Neck_Sag_SD). Le variabili angolari medie (posizione media degli arti, RULA) non mostrano effetti significativi.

**Grafici:** `condition_effect.png` — barplot degli effect size + volcano plot (rosso = FDR sig, arancione = grezzo sig, grigio = n.s.)

---

## 7. Order effect check

**Test:** Mann-Whitney U sui delta scores tra gruppo A→B (n=8) e gruppo B→A (n=10).
**Obiettivo:** verificare che i delta non differiscano sistematicamente in funzione dell'ordine di somministrazione (potenziale effetto apprendimento o fatica).

**Risultato:** 1/21 variabili con p < 0.05:
- `Vel_SD_Dom`: p = 0.034 (non corretto per FDR)

Dopo correzione per comparazioni multiple questo risultato non sarebbe più significativo. L'ordine non sembra influenzare sistematicamente i delta cinematici.

**Grafico:** `order_effect.png` — dot plot dei delta mediani per gruppo d'ordine, per ogni variabile.

---

## 8. PCA sui delta scores

### Metodo
PCA implementata manualmente via **SVD** (numpy, senza sklearn):
- Matrice di input: delta scores 18 soggetti × 21 variabili
- Standardizzazione: z-score per variabile (media=0, SD=1) prima della SVD
- Equazioni: X = U·S·Vt → scores = U·S, loadings = Vt.T, var_exp = S²/Σ(S²)
- Numero massimo di componenti = min(N, p) = min(18, 21) = 18

### Varianza spiegata

| Componente | Varianza | Cumulata |
|---|---|---|
| PC1 | 22.5% | 22.5% |
| PC2 | 16.2% | 38.7% |
| PC3 | 15.3% | 54.1% |
| PC4 | 10.0% | 64.1% |
| PC5 | 8.6% | 72.6% |
| PC6 | 7.1% | 79.7% |

La varianza è distribuita su molte componenti (PC1 spiega solo il 22.5%), indicando che la risposta al workload è multidimensionale: non c'è una singola direzione dominante di variazione interindividuale.

### Grafici prodotti
- `pca_scree.png` — Scree plot + varianza cumulata
- `pca_biplot.png` — Soggetti in spazio PC1×PC2 con frecce delle variabili (rosse = significative FDR)
- `pca_loadings.png` — Heatmap loadings PC1–PC4 (asterisco = variabili significative FDR)

---

## 9. K-Means k=2 sulle condizioni (separabilità baseline/workload)

### Obiettivo
Valutare se le due condizioni formano cluster naturali nel spazio delle 21 variabili cinematiche, indipendentemente dalle etichette. Questo risponde alla domanda: "il profilo cinematico di baseline e workload è sufficientemente diverso da essere separabile in modo non supervisionato?"

### Metodo
- Matrice: 36 righe × 21 variabili (dataset pulito, tutte le condizioni)
- Standardizzazione z-score
- K-Means k=2 via `scipy.cluster.vq.kmeans2`, 100 run con seed diversi, selezione per best inertia
- Allineamento etichette cluster → condizione per majority vote

### Risultati

**Accuracy: 77.8%** (28/36 righe classificate correttamente)

| | Pred baseline | Pred workload |
|---|---|---|
| **True baseline** | 15 | 3 |
| **True workload** | 5 | 13 |

**Soggetti mal classificati (8):**
- Soggetti 5, 6, 13 classificati come workload pur essendo baseline
- Soggetti 2, 4, 12, 14, 17 classificati come baseline pur essendo workload

L'accuracy del 77.8% superiore al 50% del caso indica che le due condizioni sono parzialmente separabili nello spazio cinematico. La sovrapposizione residua riflette la variabilità inter-soggetto: alcuni individui mostrano profili cinematici simili nelle due condizioni (effetto workload attenuato), altri mostrano alta variabilità indipendentemente dalla condizione.

**Grafico:** `kmeans_conditions.png` — scatter PCA 36×21 con doppio subplot: sinistra = condizione vera, destra = corretto/errato (verde/rosso) con cerchio=baseline e quadrato=workload.

---

## 10. File prodotti

### Dataset
| File | Contenuto |
|---|---|
| `analysis_dataset.csv` | Dataset originale 36 righe × 43 colonne |
| `analysis_dataset_clean.csv` | Dataset dopo sostituzione outlier ±2SD per condizione |

### Figure
| File | Descrizione |
|---|---|
| `boxplots_cinematiche.png` | Boxplot tutte 21 variabili, dati originali, soglie ±2SD |
| `sparc_jerk_overview.png` | SPARC e Jerk: boxplot + paired lines + violin (dati puliti) |
| `paired_lines_cinematiche.png` | Paired lines tutte 21 variabili (dati puliti) |
| `qqplot_delta.png` | Q-Q plot normalità dei delta scores |
| `condition_effect.png` | Effect size + volcano plot effetto condizione |
| `order_effect.png` | Dot plot order effect check |
| `pca_scree.png` | Scree plot e varianza cumulata PCA |
| `pca_biplot.png` | Biplot soggetti e variabili in spazio PC1×PC2 |
| `pca_loadings.png` | Heatmap loadings PCA (PC1–PC4) |
| `kmeans_conditions.png` | K-Means k=2: separazione baseline/workload in spazio PCA |

---

## 11. Librerie utilizzate

| Libreria | Uso |
|---|---|
| pandas | Caricamento e manipolazione dataset |
| numpy | Calcoli numerici, SVD per PCA |
| scipy | Shapiro-Wilk, Wilcoxon, t-test, Mann-Whitney U, K-Means |
| statsmodels | Correzione FDR Benjamini-Hochberg |
| matplotlib | Tutte le visualizzazioni |
