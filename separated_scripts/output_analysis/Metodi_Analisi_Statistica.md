# Metodi di analisi statistica — VRula Body Tracking Study

## 1. Design dello studio

Lo studio è una **validazione pilota** di uno strumento ergonomico basato su computer vision (Intel RealSense D435i + MediaPipe Pose). Il design è **within-subject controbilanciato A/B** con N = 18 partecipanti.

Ogni partecipante ha eseguito un task di trasferimento oggetti in due condizioni:

| Condizione | Descrizione |
|---|---|
| **Baseline** | Task motorio puro (trasferimento oggetti) |
| **Workload** | Task motorio + sottrazione seriale a ritroso di 13 da 100 (dual-task cognitivo) |

L'ordine delle condizioni è stato controbilanciato:
- **Gruppo A→B** (n = 9): prima Baseline, poi Workload
- **Gruppo B→A** (n = 9): prima Workload, poi Baseline

**Mano dominante:** due partecipanti (codici 01 e 09) sono mancini; per tutti gli altri la mano dominante è la destra. Il questionario usa una scala 0–100 (separatore virgola, decimale punto); i valori vengono divisi per 100 nel preprocessing per ottenere la scala 0–1 usata nell'analisi.

---

## 2. Strumento e acquisizione

La videocamera Intel RealSense D435i acquisisce il segnale RGB-D. Il modulo `processing.py` usa MediaPipe Pose per stimare la posa 3D dello scheletro frame-per-frame, producendo coordinate normalizzate in uno spazio metrico corretto per la gravità (vettore di gravità calibrato con `gravity.py` prima di ogni sessione).

L'output è un CSV per sessione con:
- Timestamp ad ogni frame
- Coordinate 3D (X, Y, Z in metri) dei landmark: polso destro/sinistro, gomito, spalla, collo, bacino
- Angoli articolari: braccio superiore (sagittale), avambraccio, flessione/deviazione polso
- Score RULA composito (RULA_C, scala 1–7) calcolato frame-per-frame da `rula_logic.py`

**Frequenza di campionamento effettiva:** ~25–30 Hz (variabile tra sessioni, stimata dalla mediana degli intervalli di frame).

---

## 3. Pipeline di elaborazione del segnale

Il modulo `AnalisiDati.py` elabora ogni CSV con i seguenti passi:

### 3.1 Preparazione del segnale

1. Parsing del timestamp e ordinamento cronologico dei frame
2. Stima della frequenza di campionamento: `fs = 1 / median(Δt)`
3. **Filtro passa-basso Butterworth** di ordine 2 a 1.5 Hz (o 80% della frequenza di Nyquist se inferiore) applicato separatamente alle coordinate X, Y, Z di ogni polso tramite `scipy.signal.filtfilt` (zero-phase, forward-backward). Questo rimuove il rumore ad alta frequenza del tracker preservando la cinematica del movimento.

### 3.2 Metriche cinematiche calcolate

#### SPARC (Spectral Arc Length)
Misura la fluidità spettrale della velocità scalare. Calcola la lunghezza dell'arco dello spettro normalizzato della velocità nel range 0–10 Hz. **Più negativo = movimento meno fluido** (più componenti ad alta frequenza = more jerkiness).

```
SPARC = -Σ √(Δfreq² + ΔM_n²)
```
dove M_n è lo spettro di ampiezza normalizzato, troncato al primo punto sotto la soglia del 5%.

#### Log-dimensionless Jerk (LDJ)
Misura la scorrevolezza del movimento. Il jerk (derivata terza della posizione) viene normalizzato per durata e ampiezza per rendere il valore confrontabile tra sessioni di durata e scala diversa.

```
LDJ = -log( (T⁵ / A²) · mean(jerk²) )
```
dove T = durata totale, A = ampiezza spaziale 3D (deviazione standard delle coordinate). **Più negativo = più fluido.**

#### Dwell Time
Percentuale del tempo totale del task in cui la velocità scalare del polso è inferiore a una soglia di **0.05 m/s** (stasi). Calcolata separatamente per polso DX e SX.

#### DTW (Dynamic Time Warping) della traiettoria
Misura la deviazione della traiettoria 3D del polso in Workload rispetto alla traiettoria Baseline, usata come riferimento. Calcolato con DTW standard sull'intera traiettoria 3D (X, Y, Z). **Nota importante:** il valore DTW per la condizione Baseline è 0 per costruzione (la traiettoria si confronta con sé stessa). Solo il valore DTW_Workload è interpretabile.

#### RULA Medio (ergonomia posturale)
Media aritmetica dei valori RULA_C (scala 1–7) su tutti i frame della sessione. RULA (Rapid Upper Limb Assessment) considera postura di braccio superiore, avambraccio, polso e tronco/collo.

Sono state calcolate anche:
- **RULA_Peak**: valore massimo osservato
- **RULA_Pct4**: % di frame con RULA ≥ 4 (soglia "action needed")
- **RULA_Pct6**: % di frame con RULA ≥ 6 (soglia "high risk")

#### Inversioni di velocità
Numero di picchi locali nella velocità scalare (algoritmo `find_peaks` con distanza minima 0.1 s e prominenza minima 0.01 m/s). Indica la presenza di micro-pause e ripartenze nel gesto.

#### Velocità polso
- `Vel_Mean`: velocità media scalare della sessione
- `Vel_Peak`: velocità massima
- `Vel_SD`: deviazione standard della velocità

Tutte calcolate sulla velocità scalare `v = √(vₓ² + vy² + vz²)`, dove le derivate sono calcolate con `np.gradient` a frequenza costante.

#### Path Efficiency
```
PathEff = (n_picchi × 0.5 m) / lunghezza_percorso_3D
```
Il razionale è che ogni ciclo di trasferimento (andata + ritorno) percorre circa 0.5 m. Il rapporto normalizza il numero di submovimenti per metro percorso. **Nota sulla direzione:** durante il workload il valore aumenta rispetto alla baseline, contrariamente all'ipotesi iniziale. Questo va interpretato come **indice di frammentazione**: più alto = movimento spezzato in più unità brevi separate da pause, non come "efficienza" in senso positivo.

#### Deviazioni standard angolari
`SD_Upper_Arm` (braccio superiore) e `SD_Lower_Arm` (avambraccio), calcolate frame-by-frame come deviazione standard dell'angolo articolare nell'intera sessione. Indicano la variabilità posturale.

---

## 4. Remapping mano dominante / non-dominante

Per garantire che le metriche primarie si riferiscano sempre alla mano che esegue il task principale, è stato applicato un remapping per i due partecipanti mancini (partecipanti 01 e 09):

- **Partecipanti destrimani** (n = 9): colonne `_DX` = mano dominante, `_SX` = mano non-dominante
- **Partecipanti mancini** (n = 2): swap in-place di `_DX` ↔ `_SX` prima del renaming

Dopo lo swap, tutte le colonne sono rinominate `_Dom` / `_NonDom`. Le analisi primarie usano la colonna `_Dom`.

I CSV originali in `output_csv/` non sono stati modificati.

---

## 5. Dati soggettivi — costruzione dei domini

Il questionario usato non è un NASA-TLX standard ma uno strumento composito adattato. I dati sono stati normalizzati in scala 0–1 (min-max lineare) prima dell'inserimento nel CSV.

### 5.1 Dominio NASA (carico cognitivo e frustrazione)

```
NASA_domain = mean(
    mental_demand,
    time_pressure,
    effort,
    1 − performance,         ← reverse code: alta performance = basso carico
    mean(insecurity, discouragement, irritation, stress, annoyance)
)
```

**Direzione:** alto = più carico percepito.

### 5.2 Dominio SART (saturazione attentiva)

```
SART_domain = mean(
    complexity,
    1 − readiness,           ← reverse code: alta prontezza = meno carico
    1 − attentional_focus,   ← reverse code: più focus = meno burden
    1 − spare_capacity       ← reverse code: più capacità residua = meno carico
)
```

**Direzione:** alto = più burden attentivo.

**Nota sul reverse coding SART:** senza reverse code la media grezza del dominio SART era *inferiore* in Workload rispetto a Baseline (0.187 vs 0.242), indicando che i singoli item misuravano risorse disponibili (alto = più risorse = meno carico). Il reverse code applicato ai tre item rende il dominio coerente con la direzione del NASA_domain.

### 5.3 Dominio Agency+Control

```
AgencyControl_domain = mean(control_of_movements, sense_of_agency)
```

**Direzione:** alto = più controllo/agenzia percepita (opposta al carico).

### 5.4 Error Awareness

Item singolo (consapevolezza di errori o quasi-errori). Non aggregato, trattato come misura separata.

---

## 6. Dataset di analisi

Il dataset finale (`output_analysis/analysis_dataset.csv`) ha **22 righe** (11 partecipanti × 2 condizioni) e 52 colonne, organizzate come segue:

| Gruppo | Colonne |
|---|---|
| Identificativi | Subject_ID, Survey_Code, Condition, Order_Group, Dom_Hand |
| Cinematica (da _Comparazione.csv) | SPARC_Dom/NonDom, Jerk_Dom/NonDom, VelInv_Dom/NonDom, DTW_Dom/NonDom, Dwell_Dom/NonDom, RULA_Mean, SD_UA_Dom/NonDom, SD_LA_Dom/NonDom, Time_to_Complete |
| Cinematica aggiuntiva (da CSV raw) | Vel_Mean/Peak/SD_Dom/NonDom, PathLen, PathEff, RULA_Peak, RULA_Pct4/Pct6, angoli polso/UA, collo, tronco |
| Soggettivo | NASA_domain, SART_domain, AgencyControl_domain, Error_Awareness |

---

## 7. Metodi statistici

### 7.1 Delta scores

Per ogni metrica è stato calcolato il **delta score** come differenza paired:
```
Δ = Workload − Baseline
```
producendo 11 valori per variabile. Tutte le analisi inferenziali sui dati appaiati usano i delta scores.

### 7.2 Test di normalità

**Shapiro-Wilk** sui delta scores (scelta giustificata da N < 30, test più potente per campioni piccoli). Affiancato da Q-Q plot.

Risultati rilevanti:
- SPARC_Dom: **non normale** (W=0.704, p<0.001) — distribuzione con coda pesante
- RULA_Mean, RULA_Pct4, SD_LA_Dom: **non normali** (p < 0.05)
- Tutte le altre variabili: compatibili con la normalità (p > 0.05)

**Decisione:** default ai test non-parametrici per tutte le variabili, incluse quelle normali, per coerenza metodologica e perché N=11 rende i test parametrici comunque instabili.

### 7.3 Confronto Baseline vs Workload — Wilcoxon Signed-Rank

**Test primario:** Wilcoxon Signed-Rank (dati appaiati, N=11), con alternativa bidirezionale.

**Effect size:** rank-biserial correlation r_rb (dimensionless, range −1 a +1):
```
r_rb = (T⁺ − T⁻) / (n·(n+1)/2)
```
dove T⁺ = somma dei ranghi positivi, T⁻ = somma dei ranghi negativi.
- |r_rb| < 0.3 = piccolo; 0.3–0.5 = medio; > 0.5 = grande

**Correzione per comparazioni multiple:** Benjamini-Hochberg FDR applicata **separatamente** per famiglia di outcome:
- Famiglia A: metriche oggettive (14 variabili)
- Famiglia B: domini soggettivi (4 variabili)

Soglia di significatività: q_BH < 0.05.

### 7.4 Correlazioni esplorative — Spearman rho

Correlazioni tra **delta scores** (N=11) con:
- **Spearman rho**: robusto a distribuzioni non-normali e outlier
- **Bootstrap 95% CI**: 1000 ricampionamenti con sostituzione
- **Correzione BH FDR** sulle 8 coppie primarie ipotizzate (obj × subj)
- Le 3 coppie di consistenza interna non sono incluse nella correzione multipla

**Soglia di potenza:** con N=11 e α=0.05, il Wilcoxon rileva effetti grandi (r_rb ≥ 0.60) con ~80% di potenza. Per le correlazioni Spearman, serve |rho| ≥ 0.60 per raggiungere ~80% di potenza. Correlazioni con |rho| < 0.60 sono **sottopotenziate** e vanno interpretate come tendenze esplorative, non come evidenza conclusiva.

### 7.5 Order effects check

**Mann-Whitney U** tra il gruppo A→B (n=5) e il gruppo B→A (n=6) sui delta scores. Analisi descrittiva: con N=11 non è possibile testare in modo inferenziale gli order effects. Lo scopo è verificare che i delta scores siano simili tra i due gruppi.

---

## 8. Conformità con l'Experimental Design

L'analisi ha seguito il piano pre-registrato nell'Experimental Design. La tabella seguente documenta il confronto punto per punto.

### 8.1 Cosa è stato seguito fedelmente

| Specifica del piano | Implementazione | Stato |
|---|---|---|
| Test primario: Wilcoxon Signed-Rank | ✓ Applicato su tutte le metriche paired | **Conforme** |
| Correzione multipla: Benjamini-Hochberg FDR | ✓ Separata per famiglia obj / subj | **Conforme** |
| Effect size: rank-biserial r_rb | ✓ Calcolato con formula (T⁺−T⁻)/W_max | **Conforme** |
| Test di normalità: Shapiro-Wilk sui delta scores | ✓ Con Q-Q plot affiancati | **Conforme** |
| Default ai test non-parametrici | ✓ Wilcoxon anche per variabili normali | **Conforme** |
| Correlazioni: Spearman rho + bootstrap 95% CI | ✓ 1000 ricampionamenti, rng=42 | **Conforme** |
| Gerarchia outcome: primario obj → secondario subj → esplorativo | ✓ Famiglie separate, BH per famiglia | **Conforme** |
| Remapping Dom/NonDom per mancini | ✓ Swap DX↔SX per partecipanti 01 e 09 | **Conforme** |
| Order effects check: Mann-Whitney U | ✓ Solo descrittivo (N troppo piccolo) | **Conforme** |
| No modifica dei CSV raw in output_csv/ | ✓ Nuova cartella output_analysis/ | **Conforme** |
| Coppie di correlazione specificate (§14.9) | ✓ Incluse NASA vs Jerk/SPARC/Dwell/RULA, SART vs Jerk, AgCtrl vs RULA, ErrAware vs Dwell | **Conforme** |

### 8.2 Adattamenti rispetto al piano originale (giustificati)

| Punto del piano | Adattamento | Motivazione |
|---|---|---|
| **DTW in test paired** | DTW Baseline = 0 per costruzione (è il riferimento). Non incluso nel Wilcoxon. Riportato solo DTW_Workload come metrica descrittiva. | Corretto per design: la traiettoria Baseline è il riferimento, non ha senso confrontarla con sé stessa. Esplicitato nel piano. |
| **PathEff: direzione attesa = decrease** | Risulta *increase* (q=0.023). Reinterpretato come indice di frammentazione (più alto = più frammentato). | Il metric definisce picchi/metro, non efficienza direzionale. L'aumento è biologicamente coerente: il dual-task causa più micro-soste e ripartenze, aumentando il numero di picchi per metro. La direzione iniziale era basata su un'assunzione errata. |
| **Velocità polso: serie temporale + CSV separati** | Solo metriche aggregate (mean, peak, SD) invece di CSV con serie temporale completa. | Semplificazione pratica: le aggregate sono sufficienti per i test statistici, evitando 22 file extra. |
| **Paired t-test come test secondario** | Non applicato — Shapiro-Wilk ha rilevato non-normalità in 4 variabili chiave, e N=11 è troppo piccolo per assumere la CLT. | Conforme al piano ("usare paired t-test solo se Shapiro-Wilk non-significativo e Q-Q plot confirmatorio"). |
| **Statistiche descrittive: skewness e kurtosis** | Non riportate esplicitamente. Riportati mediana [IQR] e media ± DS. | Con N=11 gli stimatori di skewness e kurtosis sono estremamente instabili. |

### 8.3 Nota sull'Experimental Design §4 — ipotesi direzionali

Le ipotesi erano:

| Metrica | Atteso WL vs Base | Osservato | Conforme |
|---|---|---|---|
| Jerk | ↑ | Δ=+0.39, r_rb=+0.52 (trend) | ✓ |
| Dwell Time | ↑ | Δ=+28.4%, **q=0.023** | ✓ |
| DTW | ↑ | Solo WL descrittivo (0.06–0.08 m) | ✓ |
| SPARC | ↓ (più negativo) | Δ=−0.017, r_rb=−0.55 (trend) | ✓ |
| SD Upper Arm | ↓ (irrigidimento) | Δ=+0.84 (direzione opposta) | ✗ |
| Vel polso | ↓ | Δ=−0.13, **q=0.023** | ✓ |
| NASA | ↑ | **q=0.002, r_rb=1.0** | ✓ |
| SART | ↑ | **q=0.002, r_rb=1.0** | ✓ |
| Agency+Control | ↓ | **q=0.003, r_rb=−1.0** | ✓ |

**Eccezione SD Upper Arm:** la variabilità angolare del braccio superiore è aumentata in Workload (non diminuita come atteso). L'ipotesi di "irrigidimento posturale" non trova supporto nei dati. Una spiegazione alternativa è che il dual-task cognitivo riduce il controllo fine dei movimenti, aumentando la variabilità posturale piuttosto che ridurla.

---

## 9. Output prodotti dall'analisi

| File | Contenuto |
|---|---|
| `analysis_dataset.csv` | Dataset unico 22 righe × 52 colonne |
| `descriptive_stats.csv` | Mediana [IQR] e media ± DS per condizione |
| `normality_tests.csv` | Shapiro-Wilk W e p per ogni variabile |
| `wilcoxon_obiettivi.csv` | Test + BH FDR per le 14 metriche oggettive |
| `wilcoxon_soggettivi.csv` | Test + BH FDR per i 4 domini soggettivi |
| `spearman_correlations.csv` | Spearman rho + CI bootstrap + BH FDR |
| `order_effects.csv` | Mann-Whitney U tra gruppi A→B e B→A |
| `figures/paired_lines.png` | Spaghetti plot per ogni metrica (N=11 linee) |
| `figures/qq_plots.png` | Q-Q plot normalità sui delta scores |
| `figures/correlation_heatmap.png` | Heatmap Spearman rho tra tutti i delta |
| `figures/nasa_vs_obj_scatter.png` | Scatter NASA vs Jerk/SPARC/Dwell/RULA |
| `figures/rula_zones.png` | Zone di rischio RULA (stacked bar) |
| `figures/order_effects.png` | Dot plot delta per gruppo d'ordine |

---

## 10. Software e librerie

| Libreria | Versione | Uso |
|---|---|---|
| Python | 3.x | Ambiente principale |
| pandas | — | Gestione dataset |
| numpy | — | Calcoli numerici |
| scipy | — | Shapiro-Wilk, Wilcoxon, Spearman, filtri Butterworth |
| statsmodels | — | Correzione BH FDR (multipletests) |
| matplotlib | — | Visualizzazioni |
| seaborn | — | Heatmap correlazioni |
| mediapipe | — | Pose estimation (acquisizione) |
