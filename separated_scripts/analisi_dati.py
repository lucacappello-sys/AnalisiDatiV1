import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import shapiro, ttest_rel, wilcoxon, mannwhitneyu
from statsmodels.stats.multitest import multipletests

# ============================================================
# 0. CARICAMENTO E PREPARAZIONE
# ============================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "output_analysis", "analysis_dataset.csv")

df = pd.read_csv(dataset_path)

df["Subject_ID"]  = df["Subject_ID"].astype("category")
df["Condition"]   = pd.Categorical(df["Condition"], categories=["baseline", "workload"], ordered=True)
df["Order_Group"] = df["Order_Group"].astype("category")
df["Dom_Hand"]    = df["Dom_Hand"].astype("category")

print(f"N soggetti: {df['Subject_ID'].nunique()}")
print(f"N righe:    {len(df)}")


# ============================================================
# 1. VARIABILI CINEMATICHE
# ============================================================

kinematics_vars = [
    "SPARC_Dom", "Jerk_Dom", "VelInv_Dom", "Dwell_Dom",
    "Vel_Mean_Dom", "Vel_Peak_Dom", "Vel_Var_Dom", "PathLen_Dom",
    "RULA_Mean",
    "Wr_Flex_Mean_Dom", "Wr_Flex_Var_Dom",
    "Wr_Dev_Mean_Dom",  "Wr_Dev_Var_Dom",
    "UA_Mean_Dom",      "Var_UA_Dom",
    "LA_Mean_Dom",      "Var_LA_Dom",
    "Neck_Sag_Mean",    "Neck_Sag_Var",
    "Trunk_Sag_Mean",   "Trunk_Sag_Var",
]


# ============================================================
# 2. STATISTICHE DESCRITTIVE
# ============================================================

def descrittive(data, vars_list):
    rows = []
    for var in vars_list:
        x = pd.to_numeric(data[var], errors="coerce")
        rows.append({
            "variabile": var,
            "n":         x.notna().sum(),
            "mean":      x.mean(),
            "sd":        x.std(ddof=1),
            "min":       x.min(),
            "q1":        x.quantile(0.25),
            "median":    x.median(),
            "q3":        x.quantile(0.75),
            "max":       x.max(),
        })
    return pd.DataFrame(rows).set_index("variabile").round(4)


print("\n" + "=" * 60)
print("DESCRITTIVE PER CONDITION")
print("=" * 60)
for condition in ["baseline", "workload"]:
    print(f"\n--- {condition.upper()} ---")
    print(descrittive(df[df["Condition"] == condition], kinematics_vars).to_string())


# ============================================================
# 3. RILEVAMENTO OUTLIER (+/-2 SD) per condition
# ============================================================

soglie_rows = []
outlier_rows = []

for condition in ["baseline", "workload"]:
    idx_cond = df[df["Condition"] == condition].index
    for var in kinematics_vars:
        x     = pd.to_numeric(df.loc[idx_cond, var], errors="coerce")
        mu    = x.mean()
        sigma = x.std(ddof=1)
        lower = mu - 2 * sigma
        upper = mu + 2 * sigma

        soglie_rows.append({
            "condition": condition,
            "variabile": var,
            "mean":      round(mu,    4),
            "sd":        round(sigma, 4),
            "lower_2sd": round(lower, 4),
            "upper_2sd": round(upper, 4),
        })

        mask = x.notna() & ((x < lower) | (x > upper))
        for idx in df.loc[idx_cond][mask].index:
            outlier_rows.append({
                "variabile":  var,
                "Subject_ID": df.at[idx, "Subject_ID"],
                "Condition":  condition,
                "valore":     round(x[idx], 4),
                "lower_2sd":  round(lower,  4),
                "upper_2sd":  round(upper,  4),
            })

print("\n" + "=" * 60)
print("SOGLIE +/-2 SD PER CONDITION")
print("=" * 60)
soglie_df = pd.DataFrame(soglie_rows).set_index(["condition", "variabile"])
# dict accesso rapido: soglie_dict[(condition, var)] = (lower, upper)
soglie_dict = {
    (r["condition"], r["variabile"]): (r["lower_2sd"], r["upper_2sd"])
    for r in soglie_rows
}
print(soglie_df.to_string())

print("\n" + "=" * 60)
print("OUTLIER RILEVATI (+/-2 SD per condition)")
print("=" * 60)
if outlier_rows:
    outlier_df = pd.DataFrame(outlier_rows)
    print(outlier_df.to_string(index=False))
    print("\n--- Conteggio per variabile e condition ---")
    print(outlier_df.groupby(["variabile", "Condition"]).size()
          .rename("n_outlier").to_string())
else:
    print("Nessun outlier trovato.")
    outlier_df = pd.DataFrame()


# ============================================================
# 4. SOSTITUZIONE OUTLIER CON LA MEDIA (per condition)
# ============================================================

df_clean = df.copy()

for condition in ["baseline", "workload"]:
    idx_cond = df_clean[df_clean["Condition"] == condition].index
    for var in kinematics_vars:
        x     = pd.to_numeric(df_clean.loc[idx_cond, var], errors="coerce")
        mu    = x.mean()
        sigma = x.std(ddof=1)
        lower = mu - 2 * sigma
        upper = mu + 2 * sigma

        is_outlier = x.notna() & ((x < lower) | (x > upper))
        n_replaced = is_outlier.sum()

        if n_replaced > 0:
            mean_clean = x[~is_outlier].mean()
            df_clean[var] = df_clean[var].astype(float)
            df_clean.loc[idx_cond[is_outlier], var] = mean_clean
            print(f"  [{condition}] {var}: {n_replaced} outlier sostituiti con media={mean_clean:.4f}")

print("\n" + "=" * 60)
print("DESCRITTIVE DOPO PULIZIA (per condition)")
print("=" * 60)
for condition in ["baseline", "workload"]:
    print(f"\n--- {condition.upper()} ---")
    print(descrittive(df_clean[df_clean["Condition"] == condition], kinematics_vars).to_string())


# ============================================================
# 5. GRAFICI — SPARC_Dom e Jerk_Dom
# ============================================================

figures_dir = os.path.join(script_dir, "output_analysis", "figures")
os.makedirs(figures_dir, exist_ok=True)

COLORS   = {"baseline": "#4C9BE8", "workload": "#E8754C"}
CONDS    = ["baseline", "workload"]
VARS     = ["SPARC_Dom", "Jerk_Dom"]
subjects = df_clean["Subject_ID"].cat.categories
subj_colors = plt.cm.tab20(np.linspace(0, 1, len(subjects)))
rng = np.random.default_rng(42)

# --- Boxplot griglia tutte le variabili (dati originali) -----
ncols = 4
nrows = -(-len(kinematics_vars) // ncols)
fig0, axes0 = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
axes0 = axes0.flatten()

for i, var in enumerate(kinematics_vars):
    ax = axes0[i]
    data_orig = [
        df.loc[df["Condition"] == c, var].dropna().values.astype(float)
        for c in CONDS
    ]
    bp = ax.boxplot(data_orig, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2), sym="")
    for patch, color in zip(bp["boxes"], COLORS.values()):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Punti individuali jittered
    for j, (vals, color) in enumerate(zip(data_orig, COLORS.values()), start=1):
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(j + jitter, vals, color=color, s=25, zorder=3,
                   edgecolors="white", linewidths=0.4)

    # Linee ±2 sigma per condizione (baseline=1, workload=2)
    for j, cond in enumerate(CONDS, start=1):
        lower, upper = soglie_dict[(cond, var)]
        ax.hlines([lower, upper], j - 0.3, j + 0.3,
                  colors="red", linestyles="dashed", linewidth=1.2)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Baseline", "Workload"], fontsize=9)
    ax.set_title(var, fontsize=9, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

for ax in axes0[len(kinematics_vars):]:
    ax.set_visible(False)

fig0.suptitle("Boxplot variabili cinematiche — dati originali (outlier Tukey visibili)",
              fontsize=12, fontweight="bold")
fig0.tight_layout()
fig0.savefig(os.path.join(figures_dir, "boxplots_cinematiche.png"),
             dpi=150, bbox_inches="tight")
plt.close()
print("Salvato: boxplots_cinematiche.png")

# Layout: 2 righe (una per variabile) x 3 colonne
#   Col 0: boxplot dati ORIGINALI con outlier Tukey
#   Col 1: paired lines dati puliti
#   Col 2: violin plot dati puliti con tutti i punti
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

for row, var in enumerate(VARS):
    b_raw  = df.loc[df["Condition"] == "baseline",  var].values.astype(float)
    wl_raw = df.loc[df["Condition"] == "workload",  var].values.astype(float)
    b_cln  = df_clean.loc[df_clean["Condition"] == "baseline",  var].values.astype(float)
    wl_cln = df_clean.loc[df_clean["Condition"] == "workload",  var].values.astype(float)

    # ── Col 0: Boxplot dati puliti + punti + linee ±2σ ───────────
    ax = axes[row, 0]
    bp = ax.boxplot([b_cln, wl_cln], patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2), sym="")
    for patch, color in zip(bp["boxes"], COLORS.values()):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for j, (vals, color) in enumerate(zip([b_cln, wl_cln], COLORS.values()), start=1):
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(j + jitter, vals, color=color, s=40, zorder=3,
                   edgecolors="white", linewidths=0.5)
    for j, cond in enumerate(CONDS, start=1):
        lower, upper = soglie_dict[(cond, var)]
        ax.hlines([lower, upper], j - 0.3, j + 0.3,
                  colors="red", linestyles="dashed", linewidth=1.5)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Baseline", "Workload"])
    ax.set_title("Boxplot (dati puliti)")
    ax.set_ylabel(var)

    # ── Col 1: Paired lines dati puliti ─────────────────────────
    ax = axes[row, 1]
    for subj, color in zip(subjects, subj_colors):
        v_b  = df_clean[(df_clean["Subject_ID"] == subj) & (df_clean["Condition"] == "baseline")][var].values
        v_wl = df_clean[(df_clean["Subject_ID"] == subj) & (df_clean["Condition"] == "workload")][var].values
        if len(v_b) and len(v_wl):
            ax.plot([0, 1], [v_b[0], v_wl[0]], "-o", color=color,
                    alpha=0.65, linewidth=1.4, markersize=6)
    ax.plot([0, 1], [np.median(b_cln), np.median(wl_cln)],
            "k-o", linewidth=2.5, markersize=9, zorder=5, label="Mediana")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baseline", "Workload"])
    ax.set_title("Paired lines (dati puliti)")
    ax.legend(fontsize=8)

    # ── Col 2: Violin plot dati puliti + tutti i punti ───────────
    ax = axes[row, 2]
    parts = ax.violinplot([b_cln, wl_cln], positions=[1, 2],
                          showmedians=True, showextrema=True)
    for pc, color in zip(parts["bodies"], COLORS.values()):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    for part in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[part].set_color("black")
        parts[part].set_linewidth(1.2)
    for j, (vals, color) in enumerate(zip([b_cln, wl_cln], COLORS.values()), start=1):
        jitter = rng.uniform(-0.06, 0.06, size=len(vals))
        ax.scatter(j + jitter, vals, color=color, s=35, zorder=3,
                   edgecolors="white", linewidths=0.5)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Baseline", "Workload"])
    ax.set_title("Violin plot (dati puliti)")

    for col in range(3):
        axes[row, col].spines["top"].set_visible(False)
        axes[row, col].spines["right"].set_visible(False)

fig.suptitle("SPARC_Dom e Jerk_Dom — Baseline vs Workload",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(figures_dir, "sparc_jerk_overview.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Salvato: sparc_jerk_overview.png")


# ============================================================
# 6. SALVATAGGIO DATASET PULITO
# ============================================================

out_path = os.path.join(script_dir, "output_analysis", "analysis_dataset_clean.csv")
df_clean.to_csv(out_path, index=False)
print(f"\nDataset pulito salvato: {out_path}")
print(f"Shape: {df_clean.shape}")


# ============================================================
# 7. NORMALITA' DEI DELTA (Shapiro-Wilk) + QQ plot
# ============================================================

# Calcolo delta workload - baseline per soggetto
df_b  = df_clean[df_clean["Condition"] == "baseline"].set_index("Subject_ID")
df_wl = df_clean[df_clean["Condition"] == "workload"].set_index("Subject_ID")

common_subjects = df_b.index.intersection(df_wl.index)
delta = df_wl.loc[common_subjects, kinematics_vars] - df_b.loc[common_subjects, kinematics_vars]

print("\n" + "=" * 60)
print("SHAPIRO-WILK SUI DELTA (workload - baseline)")
print("=" * 60)
print(f"{'Variabile':<22} {'W':>7} {'p':>8}  Normale?")
print("-" * 45)

normality_rows = []
for var in kinematics_vars:
    d = delta[var].dropna().values
    W, p = shapiro(d)
    normale = "SI" if p > 0.05 else "NO"
    print(f"{var:<22} {W:>7.4f} {p:>8.4f}  {normale}")
    normality_rows.append({"variabile": var, "W": round(W, 4), "p": round(p, 4), "normale": normale})

normality_df = pd.DataFrame(normality_rows)

# QQ plot per ogni variabile
from scipy.stats import probplot

ncols = 4
nrows = -(-len(kinematics_vars) // ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.2))
axes = axes.flatten()

for i, var in enumerate(kinematics_vars):
    ax = axes[i]
    d  = delta[var].dropna().values
    (osm, osr), (slope, intercept, _) = probplot(d, dist="norm")
    ax.scatter(osm, osr, color="#4C9BE8", s=35, zorder=3)
    x_line = np.array([osm.min(), osm.max()])
    ax.plot(x_line, slope * x_line + intercept, "r--", linewidth=1.2)
    row = normality_df[normality_df["variabile"] == var].iloc[0]
    ax.set_title(f"{var}\np={row['p']:.3f}  {row['normale']}", fontsize=8, fontweight="bold")
    ax.set_xlabel("Quantili teorici", fontsize=7)
    ax.set_ylabel("Quantili campione", fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

for ax in axes[len(kinematics_vars):]:
    ax.set_visible(False)

fig.suptitle("QQ plot dei delta (workload - baseline)", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(figures_dir, "qqplot_delta.png"), dpi=150, bbox_inches="tight")
plt.close()
print("\nSalvato: qqplot_delta.png")


# ── Shapiro-Wilk per condition (descrittivo) ─────────────────
print("\n" + "=" * 60)
print("SHAPIRO-WILK PER CONDITION (descrittivo)")
print("=" * 60)
print(f"{'Variabile':<22} {'W_base':>8} {'p_base':>8} {'Norm_B':>7}  "
      f"{'W_wl':>8} {'p_wl':>8} {'Norm_WL':>8}")
print("-" * 75)

cond_normality_rows = []
for var in kinematics_vars:
    results = {}
    for cond, data in [("baseline", df_b), ("workload", df_wl)]:
        d = pd.to_numeric(data[var], errors="coerce").dropna().values
        W, p = shapiro(d)
        results[cond] = (round(W, 4), round(p, 4), "SI" if p > 0.05 else "NO")

    W_b, p_b, n_b = results["baseline"]
    W_wl, p_wl, n_wl = results["workload"]
    print(f"{var:<22} {W_b:>8.4f} {p_b:>8.4f} {n_b:>7}  {W_wl:>8.4f} {p_wl:>8.4f} {n_wl:>8}")
    cond_normality_rows.append({
        "variabile":    var,
        "W_baseline":   W_b,  "p_baseline":  p_b,  "normale_baseline":  n_b,
        "W_workload":   W_wl, "p_workload":  p_wl, "normale_workload":  n_wl,
    })

cond_normality_df = pd.DataFrame(cond_normality_rows)


# ============================================================
# 8. EFFETTO CONDITION — paired t-test o Wilcoxon
# ============================================================
# Per ogni variabile:
#   - se delta normale (Shapiro p > 0.05) → paired t-test
#       t = mean(delta) / (sd(delta) / sqrt(N))
#       Cohen's d_z = mean(delta) / sd(delta)
#   - se delta NON normale → Wilcoxon signed-rank
#       ranghi delle |delta|, somma ranghi positivi vs negativi
#       rank-biserial r = 1 - (2*W / (N*(N+1)/2))

print("\n" + "=" * 60)
print("EFFETTO CONDITION (baseline vs workload)")
print("=" * 60)
print(f"{'Variabile':<22} {'Test':<10} {'Stat':>8} {'p':>8} {'Effect':>8}  Sig?")
print("-" * 65)

condition_rows = []
N_subj = len(common_subjects)

for var in kinematics_vars:
    d = delta[var].dropna().values
    normale = normality_df.loc[normality_df["variabile"] == var, "normale"].values[0]

    if normale == "SI":
        # Paired t-test: testa se mean(delta) != 0
        stat, p = ttest_rel(
            df_wl.loc[common_subjects, var].values.astype(float),
            df_b.loc[common_subjects, var].values.astype(float)
        )
        test_name = "t-test"
        # Cohen's d_z: dimensione dell'effetto in unità di SD delle differenze
        effect = np.mean(d) / np.std(d, ddof=1)
        effect_name = "d_z"
    else:
        # Wilcoxon signed-rank: non parametrico, lavora sui ranghi
        stat, p = wilcoxon(
            df_wl.loc[common_subjects, var].values.astype(float),
            df_b.loc[common_subjects, var].values.astype(float),
            alternative="two-sided"
        )
        test_name = "Wilcoxon"
        # Rank-biserial r: varia tra -1 e +1
        effect = 1 - (2 * stat) / (N_subj * (N_subj + 1) / 2)
        effect_name = "r_rb"

    sig = "*" if p < 0.05 else ""
    print(f"{var:<22} {test_name:<10} {stat:>8.3f} {p:>8.4f} {effect:>8.3f}  {sig}")
    condition_rows.append({
        "variabile":    var,
        "test":         test_name,
        "statistica":   round(stat,   4),
        "p":            round(p,      4),
        effect_name:    round(effect, 4),
        "significativo": "SI" if p < 0.05 else "NO",
    })

condition_df = pd.DataFrame(condition_rows)

# ── FDR Benjamini-Hochberg ────────────────────────────────────
# Corregge i p-value per confronti multipli (21 test simultanei).
# Per ogni p-value grezzo al rango k (ordinati dal più piccolo),
# il q-value = p * m/k  (con m = numero totale di test).
# Controlla la proporzione attesa di falsi positivi tra i significativi
# (es. q < 0.05 → al più il 5% dei risultati significativi sono falsi).
reject, p_fdr, _, _ = multipletests(condition_df["p"], method="fdr_bh", alpha=0.05)
condition_df["p_fdr"]     = p_fdr.round(4)
condition_df["sig_fdr"]   = reject
condition_df["sig"]       = condition_df["p"] < 0.05

print(f"\n* p < 0.05  |  N = {N_subj}")
print("\n" + "=" * 60)
print("CONFRONTO p grezzo vs p corretto FDR (Benjamini-Hochberg)")
print("=" * 60)
print(f"{'Variabile':<22} {'p_raw':>8} {'Sig_raw':>8} {'p_fdr':>8} {'Sig_fdr':>8}")
print("-" * 58)
for _, row in condition_df.iterrows():
    print(f"{row['variabile']:<22} {row['p']:>8.4f} {'*' if row['sig'] else '':>8} "
          f"{row['p_fdr']:>8.4f} {'*' if row['sig_fdr'] else '':>8}")
n_raw = condition_df["sig"].sum()
n_fdr = condition_df["sig_fdr"].sum()
print(f"\nSignificative prima della correzione: {n_raw}/21")
print(f"Significative dopo FDR:               {n_fdr}/21")


# ============================================================
# 9. GRAFICI RISULTATI — effect size plot + volcano plot
# ============================================================

# Prepara colonna effect size unificata (d_z per t-test, r_rb per Wilcoxon)
condition_df["effect"] = condition_df["d_z"].fillna(condition_df["r_rb"])
condition_df["sig"]    = condition_df["p"] < 0.05
condition_df_sorted    = condition_df.sort_values("effect")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

# ── Effect size plot ─────────────────────────────────────────
# Mostra l'effect size per ogni variabile come barra orizzontale.
# Rosso = significativo, grigio = non significativo.
# La dimensione dell'effetto indica sia direzione (+ workload > baseline)
# che magnitudine (0.2 piccolo, 0.5 medio, 0.8 grande per d_z/r).

bar_colors = ["#E8754C" if s else "#AAAAAA" for s in condition_df_sorted["sig"]]
bars = ax1.barh(condition_df_sorted["variabile"], condition_df_sorted["effect"],
                color=bar_colors, edgecolor="white", height=0.7)

ax1.axvline(0, color="black", linewidth=1)
# Soglie convenzionali effetto piccolo/medio/grande
for val, label in [(0.2, "piccolo"), (0.5, "medio"), (0.8, "grande")]:
    for sign in [1, -1]:
        ax1.axvline(sign * val, color="gray", linewidth=0.8,
                    linestyle=":", alpha=0.6)

# Etichetta test e p-value sulle barre significative
for _, row in condition_df_sorted.iterrows():
    if row["sig"]:
        x_pos = row["effect"] + (0.03 if row["effect"] >= 0 else -0.03)
        ha = "left" if row["effect"] >= 0 else "right"
        ax1.text(x_pos, row["variabile"],
                 f"p={row['p']:.3f} ({row['test']})",
                 va="center", ha=ha, fontsize=7, color="#333333")

ax1.set_xlabel("Effect size  (d_z per t-test | r per Wilcoxon)", fontsize=9)
ax1.set_title("Effect size per variabile\n(rosso = p < 0.05)", fontsize=10, fontweight="bold")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# ── Volcano plot ─────────────────────────────────────────────
# Asse x = effect size, asse y = -log10(p).
# Più un punto è in alto → più è significativo.
# Più è lontano da 0 in x → più l'effetto è grande.
# La linea tratteggiata orizzontale segna la soglia p = 0.05.

neg_log_p  = -np.log10(condition_df["p"])
# 3 colori: significativo anche dopo FDR (rosso), solo grezzo (arancione), non sig (grigio)
def point_color(row):
    if row["sig_fdr"]:   return "#C0392B"
    if row["sig"]:       return "#E8A44C"
    return "#AAAAAA"
point_colors = [point_color(row) for _, row in condition_df.iterrows()]

ax2.scatter(condition_df["effect"], neg_log_p,
            c=point_colors, s=80, edgecolors="white", linewidths=0.5, zorder=3)

# Soglia p grezzo = 0.05
ax2.axhline(-np.log10(0.05), color="gray", linestyle="dashed",
            linewidth=1.2, label="p = 0.05 (grezzo)")
# Soglia FDR: la più piccola p_fdr significativa
if condition_df["sig_fdr"].any():
    fdr_threshold = condition_df.loc[condition_df["sig_fdr"], "p"].max()
    ax2.axhline(-np.log10(fdr_threshold), color="red", linestyle="dashed",
                linewidth=1.2, label=f"soglia FDR (p={fdr_threshold:.3f})")
ax2.axvline(0, color="black", linewidth=0.8)

# Etichette per i punti significativi (almeno grezzo)
for _, row in condition_df.iterrows():
    if row["sig"]:
        ax2.annotate(row["variabile"],
                     xy=(row["effect"], -np.log10(row["p"])),
                     xytext=(5, 3), textcoords="offset points",
                     fontsize=7, color="#333333")

ax2.set_xlabel("Effect size  (d_z | r)", fontsize=9)
ax2.set_ylabel("-log10(p)", fontsize=9)
ax2.set_title("Volcano plot\nrosso=sig FDR  arancione=solo p<0.05  grigio=n.s.",
              fontsize=10, fontweight="bold")
ax2.legend(fontsize=8)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.suptitle("Effetto Condition (baseline vs workload) — N=18",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(figures_dir, "condition_effect.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Salvato: condition_effect.png")


# ============================================================
# 10. ORDER EFFECT CHECK
# ============================================================
# Nel disegno crossover metà soggetti ha fatto A->B, metà B->A.
# Se l'ordine conta (apprendimento, fatica, carry-over),
# i delta workload-baseline sarebbero sistematicamente diversi
# tra i due gruppi → confondente.
#
# Test: Mann-Whitney U sui delta tra i due gruppi d'ordine.
# Usiamo sempre Mann-Whitney (non-parametrico) perché ogni gruppo
# ha solo ~9 soggetti — troppo pochi per stimare la normalità.
# H0: i delta dei due gruppi provengono dalla stessa distribuzione.
# Se p > 0.05 per tutte le variabili → nessun order effect rilevabile.

# Aggiungi Order_Group al dataframe dei delta
order_map = (df_clean[df_clean["Condition"] == "baseline"]
             .set_index("Subject_ID")["Order_Group"]
             .astype(str))
delta_ord = delta.copy()
delta_ord["Order_Group"] = order_map.loc[delta_ord.index].values

grp_ab = delta_ord[delta_ord["Order_Group"] == "A->B"]
grp_ba = delta_ord[delta_ord["Order_Group"] == "B->A"]

print("\n" + "=" * 60)
print("ORDER EFFECT CHECK (Mann-Whitney U sui delta)")
print(f"A->B: n={len(grp_ab)}   B->A: n={len(grp_ba)}")
print("=" * 60)
print(f"{'Variabile':<22} {'Med A->B':>9} {'Med B->A':>9} {'U':>7} {'p':>8}  Sig?")
print("-" * 62)

order_rows = []
for var in kinematics_vars:
    a = grp_ab[var].dropna().values.astype(float)
    b = grp_ba[var].dropna().values.astype(float)
    U, p = mannwhitneyu(a, b, alternative="two-sided")
    sig = "*" if p < 0.05 else ""
    print(f"{var:<22} {np.median(a):>9.3f} {np.median(b):>9.3f} {U:>7.0f} {p:>8.4f}  {sig}")
    order_rows.append({
        "variabile":   var,
        "median_AB":   round(np.median(a), 4),
        "median_BA":   round(np.median(b), 4),
        "U":           U,
        "p":           round(p, 4),
        "significativo": "SI" if p < 0.05 else "NO",
    })

order_df = pd.DataFrame(order_rows)
n_sig = (order_df["p"] < 0.05).sum()
print(f"\n* p < 0.05  |  Variabili con order effect: {n_sig}/{len(kinematics_vars)}")

# ── Grafico: delta per gruppo d'ordine ───────────────────────
# Per ogni variabile, dot plot dei delta separati per A->B e B->A.
# Se i due gruppi si sovrappongono bene → nessun order effect.

ncols = 4
nrows = -(-len(kinematics_vars) // ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.8, nrows * 3.2))
axes = axes.flatten()

ORDER_COLORS = {"A->B": "#5B8DB8", "B->A": "#C97B3A"}

for i, var in enumerate(kinematics_vars):
    ax = axes[i]
    for j, (grp_label, grp_data, x_pos) in enumerate([
        ("A->B", grp_ab, 0), ("B->A", grp_ba, 1)
    ]):
        vals   = grp_data[var].dropna().values.astype(float)
        color  = ORDER_COLORS[grp_label]
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(x_pos + jitter, vals, color=color, s=45,
                   label=grp_label, zorder=3, edgecolors="white", linewidths=0.4)
        ax.plot([x_pos - 0.15, x_pos + 0.15],
                [np.median(vals), np.median(vals)],
                color=color, linewidth=2.5, zorder=4)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    p_val = order_df.loc[order_df["variabile"] == var, "p"].values[0]
    sig_label = f"p={p_val:.3f}" + (" *" if p_val < 0.05 else "")
    ax.set_title(f"{var}\n{sig_label}", fontsize=8, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["A→B", "B→A"], fontsize=9)
    ax.set_ylabel("Delta (WL - Base)", fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if i == 0:
        ax.legend(fontsize=7)

for ax in axes[len(kinematics_vars):]:
    ax.set_visible(False)

fig.suptitle("Order effect — delta (workload - baseline) per gruppo d'ordine\n"
             "Linea orizzontale = mediana del gruppo",
             fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(figures_dir, "order_effect.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Salvato: order_effect.png")


# ============================================================
# 11. PCA SUI DELTA
# ============================================================
# Obiettivo: ridurre le 21 variabili a pochi componenti principali
# che spiegano la maggior parte della varianza nei delta.
# Permette di vedere quali variabili variano insieme e come si
# distribuiscono i soggetti nello spazio delle componenti.
#
# Implementazione via SVD (equivalente a sklearn PCA):
#   1. Standardizza i delta (media=0, sd=1) per ogni variabile
#      → necessario perché le variabili hanno scale diverse
#   2. SVD: X = U * S * Vt
#      - V (loadings): direzioni dei componenti nello spazio variabili
#      - U * S (scores): posizione dei soggetti nello spazio componenti
#   3. Varianza spiegata = S^2 / sum(S^2)

# Matrice delta: soggetti × variabili (rimuovi righe con NaN)
delta_mat = delta[kinematics_vars].dropna()
subj_ids  = delta_mat.index.tolist()

# Standardizzazione
X = delta_mat.values.astype(float)
X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

# SVD
U, S, Vt = np.linalg.svd(X_std, full_matrices=False)
scores    = U * S                          # coordinate soggetti (N × k)
loadings  = Vt.T                           # pesi variabili    (p × k)
var_exp   = (S ** 2) / np.sum(S ** 2)     # varianza spiegata per componente
cum_var   = np.cumsum(var_exp)
n_comp    = len(var_exp)   # min(N, p) = 18 for 18x21 matrix

print("\n" + "=" * 60)
print("PCA SUI DELTA (workload - baseline)")
print("=" * 60)
print(f"{'PC':<5} {'Var. spiegata':>15} {'Cumulata':>10}")
print("-" * 33)
for i in range(min(6, n_comp)):
    print(f"PC{i+1:<3} {var_exp[i]*100:>14.1f}%  {cum_var[i]*100:>9.1f}%")

# ── Fig 1: Scree plot ────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

pcs = np.arange(1, n_comp + 1)
ax1.bar(pcs, var_exp * 100, color="#4C9BE8", alpha=0.8, edgecolor="white")
ax1.set_xlabel("Componente principale")
ax1.set_ylabel("Varianza spiegata (%)")
ax1.set_title("Scree plot")
ax1.set_xticks(pcs)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

ax2.plot(pcs, cum_var * 100, "o-", color="#E8754C", linewidth=2, markersize=6)
ax2.axhline(80, color="gray", linestyle="--", linewidth=1, label="80%")
ax2.set_xlabel("Numero di componenti")
ax2.set_ylabel("Varianza cumulata (%)")
ax2.set_title("Varianza cumulata")
ax2.set_xticks(pcs)
ax2.legend(fontsize=9)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.suptitle("PCA sui delta — varianza spiegata", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(figures_dir, "pca_scree.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Salvato: pca_scree.png")

# ── Fig 2: Biplot PC1 vs PC2 ────────────────────────────────
# Soggetti come punti, variabili come frecce (loadings scalati).
# Frecce rosse = variabili significative nell'effetto condition.
sig_vars = set(condition_df.loc[condition_df["sig_fdr"], "variabile"])

fig, ax = plt.subplots(figsize=(10, 8))

# Punti soggetti
order_colors_subj = [ORDER_COLORS.get(
    str(order_map.loc[s]) if s in order_map.index else "A->B", "#888888"
) for s in subj_ids]
ax.scatter(scores[:, 0], scores[:, 1], c=order_colors_subj,
           s=80, zorder=3, edgecolors="white", linewidths=0.5)
for i, s in enumerate(subj_ids):
    ax.annotate(str(s), (scores[i, 0], scores[i, 1]),
                xytext=(4, 4), textcoords="offset points", fontsize=8)

# Frecce variabili (loadings scalati per visibilità)
scale = np.max(np.abs(scores[:, :2])) * 0.9
for j, var in enumerate(kinematics_vars):
    lx, ly = loadings[j, 0] * scale, loadings[j, 1] * scale
    color  = "#C0392B" if var in sig_vars else "#888888"
    ax.annotate("", xy=(lx, ly), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
    ax.text(lx * 1.08, ly * 1.08, var, fontsize=7, color=color, ha="center")

ax.axhline(0, color="gray", linewidth=0.5)
ax.axvline(0, color="gray", linewidth=0.5)
ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% varianza)", fontsize=10)
ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% varianza)", fontsize=10)
ax.set_title("Biplot PCA — soggetti e variabili\n"
             "Frecce rosse = significative dopo FDR  |  colore soggetto = ordine",
             fontsize=10, fontweight="bold")

# Legenda ordine
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=ORDER_COLORS["A->B"],
           markersize=9, label="A→B"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=ORDER_COLORS["B->A"],
           markersize=9, label="B→A"),
]
ax.legend(handles=legend_elements, fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(os.path.join(figures_dir, "pca_biplot.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Salvato: pca_biplot.png")

# ── Fig 3: Heatmap loadings PC1-PC4 ─────────────────────────
n_show = min(4, n_comp)
load_df = pd.DataFrame(
    loadings[:, :n_show],
    index=kinematics_vars,
    columns=[f"PC{i+1}" for i in range(n_show)]
)

fig, ax = plt.subplots(figsize=(7, 9))
im = ax.imshow(load_df.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, label="Loading")
ax.set_xticks(range(n_show))
ax.set_xticklabels([f"PC{i+1}\n({var_exp[i]*100:.1f}%)" for i in range(n_show)], fontsize=9)
ax.set_yticks(range(len(kinematics_vars)))
ax.set_yticklabels(
    [f"{'*' if v in sig_vars else ' '} {v}" for v in kinematics_vars],
    fontsize=8
)
for i in range(len(kinematics_vars)):
    for j in range(n_show):
        val = load_df.values[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=7, color="white" if abs(val) > 0.5 else "black")
ax.set_title("Loadings PCA (prime 4 componenti)\n* = significativa dopo FDR",
             fontsize=10, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(figures_dir, "pca_loadings.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Salvato: pca_loadings.png")

# ============================================================
# SEZIONE 12 — K-MEANS k=2 SULLE CONDIZIONI
# ============================================================
from scipy.cluster.vq import kmeans2

X_km     = df_clean[kinematics_vars].astype(float).values          # 36 × 21
y_true   = (df_clean["Condition"] == "workload").astype(int).values # 0=baseline, 1=workload
subj_full = df_clean["Subject_ID"].values
cond_full = df_clean["Condition"].values

mu_km    = X_km.mean(axis=0)
sigma_km = X_km.std(axis=0, ddof=1)
X_km_std = (X_km - mu_km) / sigma_km

best_labels    = None
best_inertia   = np.inf
best_centroids = None
for seed in range(100):
    np.random.seed(seed)
    centroids, labels = kmeans2(X_km_std, 2, iter=300, minit="points")
    inertia = sum(
        np.sum((X_km_std[labels == k] - centroids[k]) ** 2)
        for k in range(2)
    )
    if inertia < best_inertia:
        best_inertia   = inertia
        best_labels    = labels.copy()
        best_centroids = centroids.copy()

# Allinea: cluster 0 = baseline, cluster 1 = workload (majority vote)
acc0 = np.mean(best_labels == y_true)
acc1 = np.mean((1 - best_labels) == y_true)
if acc1 > acc0:
    best_labels = 1 - best_labels

tp = int(np.sum((best_labels == 1) & (y_true == 1)))
tn = int(np.sum((best_labels == 0) & (y_true == 0)))
fp = int(np.sum((best_labels == 1) & (y_true == 0)))
fn = int(np.sum((best_labels == 0) & (y_true == 1)))
accuracy = (tp + tn) / len(y_true)

print("\n" + "=" * 60)
print("K-MEANS k=2 SULLE CONDIZIONI")
print("=" * 60)
print(f"Accuracy (cluster vs condizione): {accuracy*100:.1f}%")
print(f"\nMatrice di confusione:")
print(f"                   Pred baseline  Pred workload")
print(f"  True baseline         {tn:2d}              {fp:2d}")
print(f"  True workload         {fn:2d}              {tp:2d}")

# Errori dettagliati
errors = np.where(best_labels != y_true)[0]
if len(errors) > 0:
    print(f"\nRighe classificate in modo errato ({len(errors)}):")
    for idx in errors:
        print(f"  Soggetto {subj_full[idx]}, condizione {cond_full[idx]}")
else:
    print("\nNessun errore di classificazione.")

# ── PCA sulle 36 righe per visualizzazione ───────────────────
U_f, S_f, Vt_f = np.linalg.svd(X_km_std, full_matrices=False)
scores_f  = U_f * S_f
var_exp_f = (S_f ** 2) / np.sum(S_f ** 2)

COND_COLORS = {"baseline": "#4C9BE8", "workload": "#E8754C"}
correct_mask = best_labels == y_true

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6))

# Sinistra: colore = condizione vera
for cond, col in COND_COLORS.items():
    mask = cond_full == cond
    ax_l.scatter(scores_f[mask, 0], scores_f[mask, 1],
                 c=col, s=70, edgecolors="white", linewidths=0.5,
                 label=cond, zorder=3)
for i in range(len(subj_full)):
    ax_l.annotate(str(subj_full[i]),
                  (scores_f[i, 0], scores_f[i, 1]),
                  xytext=(3, 3), textcoords="offset points", fontsize=7)
ax_l.axhline(0, color="gray", linewidth=0.4)
ax_l.axvline(0, color="gray", linewidth=0.4)
ax_l.set_xlabel(f"PC1 ({var_exp_f[0]*100:.1f}%)", fontsize=10)
ax_l.set_ylabel(f"PC2 ({var_exp_f[1]*100:.1f}%)", fontsize=10)
ax_l.set_title("Condizione vera", fontsize=10, fontweight="bold")
ax_l.legend(fontsize=9)
ax_l.spines["top"].set_visible(False)
ax_l.spines["right"].set_visible(False)

# Destra: colore = corretto/errato, marker = condizione vera
markers = {"baseline": "o", "workload": "s"}
for cond, marker in markers.items():
    mask = cond_full == cond
    idx_correct = np.where(mask & correct_mask)[0]
    idx_wrong   = np.where(mask & ~correct_mask)[0]
    if len(idx_correct):
        ax_r.scatter(scores_f[idx_correct, 0], scores_f[idx_correct, 1],
                     c="#2ECC71", marker=marker, s=80,
                     edgecolors="white", linewidths=0.5, zorder=3,
                     label=f"{cond} corretto")
    if len(idx_wrong):
        ax_r.scatter(scores_f[idx_wrong, 0], scores_f[idx_wrong, 1],
                     c="#E74C3C", marker=marker, s=100,
                     edgecolors="black", linewidths=0.8, zorder=4,
                     label=f"{cond} errato")

for i in range(len(subj_full)):
    ax_r.annotate(str(subj_full[i]),
                  (scores_f[i, 0], scores_f[i, 1]),
                  xytext=(3, 3), textcoords="offset points", fontsize=7)
ax_r.axhline(0, color="gray", linewidth=0.4)
ax_r.axvline(0, color="gray", linewidth=0.4)
ax_r.set_xlabel(f"PC1 ({var_exp_f[0]*100:.1f}%)", fontsize=10)
ax_r.set_ylabel(f"PC2 ({var_exp_f[1]*100:.1f}%)", fontsize=10)
ax_r.set_title(f"Classificazione K-Means (accuracy {accuracy*100:.1f}%)\n"
               "cerchio=baseline  quadrato=workload",
               fontsize=10, fontweight="bold")
ax_r.legend(fontsize=8, loc="best")
ax_r.spines["top"].set_visible(False)
ax_r.spines["right"].set_visible(False)

fig.suptitle("K-Means k=2 — separazione baseline vs workload", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(figures_dir, "kmeans_conditions.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Salvato: kmeans_conditions.png")
