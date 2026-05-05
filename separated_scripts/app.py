import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import shapiro, ttest_rel, wilcoxon, mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestPower

# ── Paths ────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR  = os.path.join(SCRIPT_DIR, "output_analysis", "figures")
DATASET_PATH = os.path.join(SCRIPT_DIR, "output_analysis", "analysis_dataset.csv")
CLEAN_PATH   = os.path.join(SCRIPT_DIR, "output_analysis", "analysis_dataset_clean.csv")

KINEMATICS_VARS = [
    "SPARC_Dom", "Jerk_Dom", "VelInv_Dom", "Dwell_Dom",
    "Vel_Mean_Dom", "Vel_Peak_Dom", "Vel_SD_Dom", "PathLen_Dom",
    "RULA_Mean",
    "Wr_Flex_Mean_Dom", "Wr_Flex_SD_Dom",
    "Wr_Dev_Mean_Dom",  "Wr_Dev_SD_Dom",
    "UA_Mean_Dom",      "SD_UA_Dom",
    "LA_Mean_Dom",      "SD_LA_Dom",
    "Neck_Sag_Mean",    "Neck_Sag_SD",
    "Trunk_Sag_Mean",   "Trunk_Sag_SD",
]

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="VRula BodyTracking — Analisi Cinematica",
    page_icon="🦾",
    layout="wide",
)

st.title("VRula BodyTracking — Analisi Dati Cinematici")
st.markdown(
    "**N=18 soggetti × 2 condizioni** (baseline / workload) | "
    "Design within-subject crossover A/B | 21 variabili cinematiche"
)

# ── Helper ───────────────────────────────────────────────────
def show_fig(filename, caption=""):
    path = os.path.join(FIGURES_DIR, filename)
    if os.path.exists(path):
        st.image(path, caption=caption, width="stretch")
    else:
        st.warning(f"Figura non trovata: `{filename}`. Eseguire prima il notebook per generarla.")

# ── Load data ────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATASET_PATH)
    df["Subject_ID"]  = df["Subject_ID"].astype("category")
    df["Condition"]   = pd.Categorical(df["Condition"], categories=["baseline", "workload"], ordered=True)
    df["Order_Group"] = df["Order_Group"].astype("category")
    df["Dom_Hand"]    = df["Dom_Hand"].astype("category")

    df_clean = pd.read_csv(CLEAN_PATH)
    df_clean["Condition"] = pd.Categorical(df_clean["Condition"], categories=["baseline", "workload"], ordered=True)
    return df, df_clean

# ── Run analysis (cached) ────────────────────────────────────
@st.cache_data
def run_analysis(clean_path):
    df_clean = pd.read_csv(clean_path)
    df_clean["Condition"] = pd.Categorical(df_clean["Condition"], categories=["baseline","workload"], ordered=True)

    df_b  = df_clean[df_clean["Condition"]=="baseline"].set_index("Subject_ID")
    df_wl = df_clean[df_clean["Condition"]=="workload"].set_index("Subject_ID")
    common = df_b.index.intersection(df_wl.index)
    delta  = df_wl.loc[common, KINEMATICS_VARS] - df_b.loc[common, KINEMATICS_VARS]

    # Normality
    norm_rows = []
    for var in KINEMATICS_VARS:
        d = delta[var].dropna().values
        W, p = shapiro(d)
        norm_rows.append({"variabile": var, "W": round(W,4), "p_shapiro": round(p,4),
                          "normale": "SI" if p > 0.05 else "NO"})
    norm_df = pd.DataFrame(norm_rows)

    # Condition effect
    N = len(common)
    cond_rows = []
    for var in KINEMATICS_VARS:
        d = delta[var].dropna().values
        normale = norm_df.loc[norm_df["variabile"]==var, "normale"].values[0]
        if normale == "SI":
            stat, p = ttest_rel(df_wl.loc[common, var].values.astype(float),
                                df_b.loc[common,  var].values.astype(float))
            test, eff, eff_name = "t-test", np.mean(d)/np.std(d, ddof=1), "d_z"
        else:
            stat, p = wilcoxon(df_wl.loc[common, var].values.astype(float),
                               df_b.loc[common,  var].values.astype(float),
                               alternative="two-sided")
            test, eff, eff_name = "Wilcoxon", 1-(2*stat)/(N*(N+1)/2), "r_rb"
        cond_rows.append({"variabile": var, "test": test, "statistica": round(stat,4),
                          "p": round(p,4), eff_name: round(eff,4)})

    cond_df = pd.DataFrame(cond_rows)
    if "d_z"  not in cond_df: cond_df["d_z"]  = np.nan
    if "r_rb" not in cond_df: cond_df["r_rb"] = np.nan
    cond_df["effect"] = cond_df["d_z"].fillna(cond_df["r_rb"])
    reject, p_fdr, _, _ = multipletests(cond_df["p"], method="fdr_bh", alpha=0.05)
    cond_df["p_fdr"]   = p_fdr.round(4)
    cond_df["sig_fdr"] = reject
    cond_df["sig"]     = cond_df["p"] < 0.05
    sig_vars = set(cond_df.loc[cond_df["sig_fdr"], "variabile"])

    # Order effect
    order_map = df_clean[df_clean["Condition"]=="baseline"].set_index("Subject_ID")["Order_Group"].astype(str)
    delta_ord = delta.copy()
    delta_ord["Order_Group"] = order_map.loc[delta_ord.index].values
    grp_ab = delta_ord[delta_ord["Order_Group"]=="A->B"]
    grp_ba = delta_ord[delta_ord["Order_Group"]=="B->A"]
    ord_rows = []
    for var in KINEMATICS_VARS:
        a = grp_ab[var].dropna().values.astype(float)
        b = grp_ba[var].dropna().values.astype(float)
        U, p = mannwhitneyu(a, b, alternative="two-sided")
        ord_rows.append({"variabile": var, "median_AB": round(np.median(a),4),
                         "median_BA": round(np.median(b),4), "U": round(U,1), "p": round(p,4),
                         "sig": "SI" if p < 0.05 else "NO"})
    order_df = pd.DataFrame(ord_rows)

    # Power analysis
    ttest_pwr = TTestPower()
    pwr_rows = []
    for _, row in cond_df.iterrows():
        var = row["variabile"]
        d   = delta[var].dropna().values
        d_z = np.mean(d) / np.std(d, ddof=1)
        pwr = ttest_pwr.solve_power(effect_size=abs(d_z), nobs=len(d), alpha=0.05, alternative="two-sided")
        if row["sig_fdr"]:
            interp = "sig. FDR"
        elif pwr >= 0.80:
            interp = "potenza adeguata (effetto assente?)"
        elif pwr >= 0.50:
            interp = "potenza moderata"
        else:
            interp = "sotto-potenziata"
        pwr_rows.append({"variabile": var, "d_z": round(d_z,4), "power": round(pwr,4),
                         "sig_fdr": row["sig_fdr"], "interpretazione": interp})
    power_df = pd.DataFrame(pwr_rows)

    # Correlation heatmap (significant vars)
    sig_var_list = sorted(sig_vars)
    delta_sig    = delta[sig_var_list].dropna()
    n_sv         = len(sig_var_list)
    corr_mat     = np.zeros((n_sv, n_sv))
    pval_mat     = np.ones((n_sv, n_sv))
    for i, v1 in enumerate(sig_var_list):
        for j, v2 in enumerate(sig_var_list):
            if i == j:
                corr_mat[i, j] = 1.0
            else:
                r, p = spearmanr(delta_sig[v1], delta_sig[v2])
                corr_mat[i, j] = r
                pval_mat[i, j] = p
    corr_df = pd.DataFrame(corr_mat, index=sig_var_list, columns=sig_var_list).round(3)

    return delta, norm_df, cond_df, sig_vars, order_df, power_df, corr_df, corr_mat, pval_mat

# ── Descriptive helper ───────────────────────────────────────
def descrittive(data, vars_list):
    rows = []
    for var in vars_list:
        x = pd.to_numeric(data[var], errors="coerce")
        rows.append({"n": x.notna().sum(), "mean": x.mean(), "sd": x.std(ddof=1),
                     "min": x.min(), "q1": x.quantile(0.25), "median": x.median(),
                     "q3": x.quantile(0.75), "max": x.max()})
    return pd.DataFrame(rows, index=vars_list).round(4)

# ── Load ─────────────────────────────────────────────────────
if not os.path.exists(DATASET_PATH):
    st.error(f"Dataset non trovato: `{DATASET_PATH}`")
    st.stop()

df, df_clean = load_data()
delta, norm_df, cond_df, sig_vars, order_df, power_df, corr_df, corr_mat, pval_mat = run_analysis(CLEAN_PATH)

# ── Tabs ─────────────────────────────────────────────────────
tabs = st.tabs([
    "📋 Dataset",
    "📊 Descrittive",
    "🔍 Outlier",
    "⚡ Effetto Condizione",
    "🔀 Order Effect",
    "🧮 PCA",
    "🎯 K-Means",
    "💪 Power Analysis",
    "🔗 Correlazioni Delta",
    "📝 Considerazioni Finali",
])

# ── Tab 0: Dataset ───────────────────────────────────────────
with tabs[0]:
    st.header("Dataset")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Soggetti", df["Subject_ID"].nunique())
    col2.metric("Righe totali", len(df))
    col3.metric("Condizioni", 2)
    col4.metric("Variabili", len(KINEMATICS_VARS))

    st.subheader("Composizione gruppi")
    group_counts = df.drop_duplicates("Subject_ID")["Order_Group"].value_counts().reset_index()
    group_counts.columns = ["Gruppo", "N soggetti"]
    st.dataframe(group_counts, width="content")

    st.subheader("Prime righe del dataset originale")
    st.dataframe(df.head(10), width="stretch")

# ── Tab 1: Descrittive ───────────────────────────────────────
with tabs[1]:
    st.header("Statistiche descrittive")
    cond_sel = st.radio("Condizione", ["baseline", "workload", "entrambe"], horizontal=True)

    if cond_sel == "entrambe":
        col_b, col_wl = st.columns(2)
        with col_b:
            st.subheader("Baseline")
            st.dataframe(descrittive(df_clean[df_clean["Condition"]=="baseline"], KINEMATICS_VARS),
                         width="stretch")
        with col_wl:
            st.subheader("Workload")
            st.dataframe(descrittive(df_clean[df_clean["Condition"]=="workload"], KINEMATICS_VARS),
                         width="stretch")
    else:
        st.dataframe(descrittive(df_clean[df_clean["Condition"]==cond_sel], KINEMATICS_VARS),
                     width="stretch")

    st.subheader("Boxplot variabili cinematiche (dati originali)")
    show_fig("boxplots_cinematiche.png")

    st.subheader("SPARC e Jerk — boxplot, paired lines, violin")
    show_fig("sparc_jerk_overview.png")

# ── Tab 2: Outlier ───────────────────────────────────────────
with tabs[2]:
    st.header("Rilevamento outlier (±2 SD per condizione)")

    # Ricalcola soglie e outlier
    outlier_rows = []
    for condition in ["baseline", "workload"]:
        idx_cond = df[df["Condition"] == condition].index
        for var in KINEMATICS_VARS:
            x = pd.to_numeric(df.loc[idx_cond, var], errors="coerce")
            mu, sigma = x.mean(), x.std(ddof=1)
            lower, upper = mu - 2*sigma, mu + 2*sigma
            mask = x.notna() & ((x < lower) | (x > upper))
            for idx in df.loc[idx_cond][mask].index:
                outlier_rows.append({"variabile": var, "Subject_ID": df.at[idx,"Subject_ID"],
                                     "Condition": condition, "valore": round(x[idx],4),
                                     "lower_2sd": round(lower,4), "upper_2sd": round(upper,4)})

    if outlier_rows:
        outlier_df = pd.DataFrame(outlier_rows)
        st.metric("Totale outlier trovati", len(outlier_df))
        st.dataframe(outlier_df, width="stretch")
    else:
        st.success("Nessun outlier trovato.")

    st.info("Gli outlier sono stati sostituiti con la media dei non-outlier della stessa condizione. "
            "Il dataset pulito è salvato in `output_analysis/analysis_dataset_clean.csv`.")

# ── Tab 3: Effetto condizione ────────────────────────────────
with tabs[3]:
    st.header("Effetto condizione (baseline vs workload)")
    st.markdown(
        "Per ogni variabile: **paired t-test** se i delta sono normali (Shapiro p>0.05), "
        "**Wilcoxon signed-rank** altrimenti. Correzione **FDR Benjamini-Hochberg** su 21 test."
    )

    n_sig_raw = cond_df["sig"].sum()
    n_sig_fdr = cond_df["sig_fdr"].sum()
    col1, col2, col3 = st.columns(3)
    col1.metric("Significative (p<0.05)", f"{n_sig_raw}/21")
    col2.metric("Significative dopo FDR", f"{n_sig_fdr}/21")
    col3.metric("Soggetti analizzati", len(delta))

    st.subheader("Tabella risultati")
    display_cols = ["variabile","test","p","sig","p_fdr","sig_fdr","effect"]
    st.dataframe(
        cond_df[display_cols].set_index("variabile").style.map(
            lambda v: "background-color: #ffd6cc" if v is True else "",
            subset=["sig_fdr"]
        ),
        width="stretch"
    )

    st.subheader("Variabili significative dopo FDR")
    st.write(", ".join(sorted(sig_vars)))

    st.subheader("Effect size + Volcano plot")
    show_fig("condition_effect.png")

# ── Tab 4: Order effect ──────────────────────────────────────
with tabs[4]:
    st.header("Order effect check (Mann-Whitney U sui delta)")
    st.markdown(
        "Confronto dei delta workload−baseline tra il gruppo **A→B** e il gruppo **B→A**. "
        "Un order effect sistematico indicherebbe un confondente (apprendimento o fatica)."
    )

    n_ord_sig = (order_df["p"] < 0.05).sum()
    st.metric("Variabili con p<0.05", f"{n_ord_sig}/21")

    st.dataframe(order_df.set_index("variabile"), width="stretch")

    if n_ord_sig > 0:
        st.warning(f"{n_ord_sig} variabile/i mostrano un possibile order effect (p<0.05, non corretto per FDR).")
    else:
        st.success("Nessun order effect sistematico rilevato.")

    show_fig("order_effect.png")

# ── Tab 5: PCA ───────────────────────────────────────────────
with tabs[5]:
    st.header("PCA sui delta (workload − baseline)")
    st.markdown("SVD sulla matrice 18×21 dei delta standardizzati. Variabili con frecce rosse = significative FDR.")

    # Ricalcola varianza spiegata
    X_std = (delta[KINEMATICS_VARS].dropna().values.astype(float))
    X_std = (X_std - X_std.mean(axis=0)) / X_std.std(axis=0, ddof=1)
    _, S, _ = np.linalg.svd(X_std, full_matrices=False)
    var_exp = (S**2) / np.sum(S**2)
    cum_var = np.cumsum(var_exp)

    var_df = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(len(var_exp))],
        "Varianza (%)": (var_exp*100).round(1),
        "Cumulata (%)": (cum_var*100).round(1),
    }).set_index("PC")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Varianza spiegata")
        st.dataframe(var_df.head(10), width="stretch")
    with col2:
        show_fig("pca_scree.png", "Scree plot + varianza cumulata")

    st.subheader("Biplot PC1 × PC2")
    show_fig("pca_biplot.png")

    st.subheader("Heatmap loadings PC1–PC4")
    show_fig("pca_loadings.png")

# ── Tab 6: K-Means ───────────────────────────────────────────
with tabs[6]:
    st.header("K-Means k=2 sulle condizioni")
    st.markdown(
        "K-Means applicato alle **36 righe × 21 variabili** (dati puliti, standardizzati). "
        "Verifica se baseline e workload sono separabili in modo non supervisionato."
    )

    st.info("Il K-Means viene eseguito nel notebook. Qui vengono mostrati i risultati pre-calcolati.")
    show_fig("kmeans_conditions.png",
             "Sinistra: condizione vera | Destra: corretto (verde) / errato (rosso)")

# ── Tab 7: Power Analysis ─────────────────────────────────────
with tabs[7]:
    st.header("Power Analysis post-hoc")
    st.markdown(
        "Per ogni variabile: potenza osservata del paired t-test con il Cohen's d_z misurato, "
        "N=18, α=0.05. "
        "**Variabili non sig. con bassa potenza** → il campione era probabilmente troppo piccolo. "
        "**Variabili non sig. con alta potenza** → l'effetto è probabilmente assente."
    )

    col1, col2 = st.columns(2)
    under_powered = power_df[(~power_df["sig_fdr"]) & (power_df["power"] < 0.80)]
    adequate      = power_df[(~power_df["sig_fdr"]) & (power_df["power"] >= 0.80)]
    col1.metric("Non sig. + sotto-potenziate", len(under_powered))
    col2.metric("Non sig. + potenza adeguata", len(adequate))

    st.subheader("Tabella power analysis")
    st.dataframe(power_df.set_index("variabile"), width="stretch")

    if not under_powered.empty:
        st.subheader("Variabili sotto-potenziate (effetto incerto)")
        st.dataframe(under_powered.set_index("variabile")[["d_z","power","interpretazione"]],
                     width="stretch")

    if not adequate.empty:
        st.subheader("Variabili con potenza adeguata ma non significative (effetto assente?)")
        st.dataframe(adequate.set_index("variabile")[["d_z","power","interpretazione"]],
                     width="stretch")

    st.subheader("Grafico power analysis")
    fig_pwr, ax_pwr = plt.subplots(figsize=(10, 7))
    bar_cols = ["#2ECC71" if s else "#E74C3C" for s in power_df["sig_fdr"]]
    ax_pwr.barh(power_df["variabile"], power_df["power"], color=bar_cols, alpha=0.85, edgecolor="white")
    ax_pwr.axvline(0.80, color="black", linestyle="--", linewidth=1.5, label="80% (soglia convenzionale)")
    ax_pwr.axvline(0.50, color="gray",  linestyle=":",  linewidth=1.2, label="50%")
    ax_pwr.set_xlabel("Potenza osservata (alpha=0.05, N=18)")
    ax_pwr.set_title("Power Analysis post-hoc\nverde = sig FDR   rosso = non significativa", fontweight="bold")
    ax_pwr.legend(fontsize=9)
    ax_pwr.spines["top"].set_visible(False); ax_pwr.spines["right"].set_visible(False)
    fig_pwr.tight_layout()
    st.pyplot(fig_pwr)
    plt.close(fig_pwr)

# ── Tab 8: Correlazioni Delta ─────────────────────────────────
with tabs[8]:
    st.header("Heatmap correlazione tra i delta (variabili sig. FDR)")
    st.markdown(
        "Correlazioni **Spearman** tra i delta scores delle variabili significative dopo FDR. "
        "Risponde alla domanda: *quando Jerk aumenta sotto workload, aumenta anche PathLen o Dwell?*"
    )

    st.subheader("Matrice di correlazione")
    st.dataframe(corr_df.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1),
                 width="stretch")

    sig_var_list_plot = sorted(sig_vars)
    n_sv_plot = len(sig_var_list_plot)
    fig_corr, ax_corr = plt.subplots(figsize=(9, 8))
    im = ax_corr.imshow(corr_mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax_corr, label="Spearman rho")
    ax_corr.set_xticks(range(n_sv_plot))
    ax_corr.set_xticklabels(sig_var_list_plot, rotation=45, ha="right", fontsize=9)
    ax_corr.set_yticks(range(n_sv_plot))
    ax_corr.set_yticklabels(sig_var_list_plot, fontsize=9)
    for i in range(n_sv_plot):
        for j in range(n_sv_plot):
            r = corr_mat[i, j]
            stars = ""
            if i != j:
                p = pval_mat[i, j]
                stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            label = f"{r:.2f}\n{stars}" if stars else f"{r:.2f}"
            ax_corr.text(j, i, label, ha="center", va="center",
                         fontsize=8, color="white" if abs(r) > 0.6 else "black")
    ax_corr.set_title("Heatmap correlazione Spearman tra delta\n* p<0.05  ** p<0.01  *** p<0.001", fontweight="bold")
    fig_corr.tight_layout()
    st.pyplot(fig_corr)
    plt.close(fig_corr)

    # Coppie più correlate
    sig_var_list = sorted(sig_vars)
    n_sv = len(sig_var_list)
    pairs = []
    for i in range(n_sv):
        for j in range(i+1, n_sv):
            r = corr_mat[i, j]
            p = pval_mat[i, j]
            if p < 0.05:
                pairs.append({"var1": sig_var_list[i], "var2": sig_var_list[j],
                               "rho": round(r,3), "p": round(p,4)})
    if pairs:
        st.subheader("Coppie significative (p<0.05, non corretto)")
        pairs_df = pd.DataFrame(pairs).sort_values("rho", key=abs, ascending=False)
        st.dataframe(pairs_df, width="stretch")
    else:
        st.info("Nessuna coppia con p<0.05.")

# ── Tab 9: Considerazioni Finali ─────────────────────────────
with tabs[9]:
    st.header("Considerazioni Finali")

    n_sig_fdr = int(cond_df["sig_fdr"].sum())
    sig_list  = sorted(sig_vars)
    n_ord_sig = int((order_df["p"] < 0.05).sum())

    # ── Riepilogo metriche chiave ─────────────────────────────
    st.subheader("Riepilogo dei risultati principali")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Variabili significative (FDR)", f"{n_sig_fdr} / 21")
    c2.metric("Accuracy K-Means (baseline/workload)", "77.8%")
    c3.metric("Varianza spiegata PCA — PC1", "22.5%")
    c4.metric("Order effect (p<0.05, non corr.)", f"{n_ord_sig} / 21")

    st.divider()

    # ── Effetto del workload ──────────────────────────────────
    st.subheader("1 — Effetto del workload sul movimento")
    st.markdown(
        f"""
Il workload cognitivo ha prodotto cambiamenti statisticamente significativi (FDR q<0.05) in
**{n_sig_fdr} delle 21 variabili** analizzate:

**Fluidità e velocità del movimento del polso dominante:**
- **SPARC** e **Jerk**: riduzione della fluidità del movimento (il gesto diventa meno levigato e più frammentato).
- **Vel_Mean** e **Vel_SD**: velocità media più bassa e meno variabile — il partecipante rallenta e omogenizza il ritmo.
- **Dwell**: aumento del tempo di stazionamento; il cursore resta fermo più a lungo.
- **PathLen**: riduzione della lunghezza del percorso — meno esplorazione spaziale complessiva.

**Variabilità posturale distale:**
- **SD_LA** (deviazione standard dell'angolo avambraccio): maggiore irregolarità dell'avambraccio sotto carico.
- **Neck_Sag_SD**: maggiore oscillazione sagittale del collo — la testa non mantiene una posizione stabile.

**Variabili non significative:** le misure di posizione media (angoli medi degli arti, RULA_Mean)
non cambiano in modo rilevante. Il workload altera la *variabilità* e la *fluidità* del
movimento, non la postura media mantenuta.
"""
    )

    st.divider()

    # ── Separabilità delle condizioni ────────────────────────
    st.subheader("2 — Separabilità delle condizioni")
    st.markdown(
        """
Il K-Means k=2 sulle 36 osservazioni (36 × 21 variabili) ha classificato correttamente il
**77.8%** dei campioni (28/36) senza usare le etichette di condizione.
Questo conferma che il profilo cinematico differisce tra baseline e workload in modo
sufficientemente marcato da emergere anche in analisi non supervisionata.

La sovrapposizione residua (22.2% di errori) riflette la **variabilità inter-individuale**:
alcuni soggetti mostrano un effetto workload attenuato, altri presentano profili molto
variabili indipendentemente dalla condizione.
"""
    )

    st.divider()

    # ── Struttura multidimensionale della risposta ────────────
    st.subheader("3 — Struttura multidimensionale della risposta")
    st.markdown(
        """
La PCA sui delta (workload − baseline) mostra che la risposta al workload **non è
unidimensionale**: la PC1 spiega solo il 22.5% della varianza, e occorrono 6 componenti
per raggiungere il 80%. Questo significa che i soggetti rispondono al carico cognitivo
in modi parzialmente diversi — non c'è un unico "profilo di risposta" condiviso da tutti.

Le correlazioni Spearman tra i delta delle variabili significative (tab *Correlazioni Delta*)
rivelano le co-variazioni tipiche: SPARC, Jerk, Vel_Mean e PathLen tendono a cambiare
insieme (coerenza del cluster di fluidità), mentre la variabilità posturale (SD_LA,
Neck_Sag_SD) forma un secondo pattern parzialmente indipendente.
"""
    )

    st.divider()

    # ── Validità del design ───────────────────────────────────
    st.subheader("4 — Validità del design sperimentale")
    st.markdown(
        f"""
**Order effect:** solo {n_ord_sig}/21 variabili mostrano una differenza tra i delta del
gruppo A→B e del gruppo B→A (p<0.05, non corretto per FDR). Questo risultato isolato
non è sufficiente a ipotizzare un confondente sistematico da apprendimento o fatica.
Il bilanciamento A/B del design crossover sembra aver controllato adeguatamente l'ordine.

**Power analysis:** le variabili non significative si dividono in due categorie:
- *Sotto-potenziate* (potenza < 80%): effetti di piccola entità che il campione di N=18
  non aveva sufficiente potere per rilevare. Studi futuri con N maggiore potrebbero
  rivelare effetti aggiuntivi.
- *Adeguatamente potenziate ma non significative*: effetti probabilmente assenti o
  trascurabili (es. posizione media degli arti, RULA_Mean).
"""
    )

    st.divider()

    # ── Limitazioni ───────────────────────────────────────────
    st.subheader("5 — Limitazioni e prospettive")
    st.markdown(
        """
- **Campione:** N=18 garantisce buona potenza per effetti medi-grandi (Cohen's d_z ≥ 0.6),
  ma resta limitato per effetti piccoli nelle variabili posturali.
- **Outlier:** la sostituzione con la media dei non-outlier (criterio ±2 SD per condizione)
  è conservativa. Un'analisi di sensitività con i dati originali potrebbe essere utile.
- **Generalizzabilità:** il paradigma VR scelto è specifico; i pattern cinematici potrebbero
  variare con task di interazione diversi o con popolazioni cliniche.
- **Direzionalità:** l'analisi è correlazionale/osservazionale; il disegno within-subject
  fornisce evidenza di cambiamento entro soggetto ma non di causalità diretta tra workload
  e modifiche cinematiche.
"""
    )

    st.divider()

    # ── Variabili raccomandate come outcome primari ───────────
    st.subheader("6 — Variabili raccomandate come outcome sensibili al workload")
    recommended = [v for v in ["SPARC_Dom", "Vel_Mean_Dom", "Dwell_Dom", "PathLen_Dom",
                               "Jerk_Dom", "Vel_SD_Dom", "SD_LA_Dom", "Neck_Sag_SD"]
                   if v in sig_vars]
    other_sig = [v for v in sig_list if v not in recommended]
    rec_rows = []
    for v in recommended + other_sig:
        row = cond_df[cond_df["variabile"] == v].iloc[0]
        rec_rows.append({
            "Variabile": v,
            "Test": row["test"],
            "p (grezzo)": row["p"],
            "p (FDR)": row["p_fdr"],
            "Effect size": round(row["effect"], 3),
        })
    st.dataframe(pd.DataFrame(rec_rows).set_index("Variabile"), width="stretch")


# ── Footer ───────────────────────────────────────────────────
st.divider()
st.caption("VRula BodyTracking Study | analisi_dati.ipynb | Generato con Python + Streamlit")
