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
    "Vel_Mean_Dom", "Vel_Peak_Dom", "Vel_Var_Dom", "PathLen_Dom",
    "RULA_Mean",
    "Wr_Flex_Mean_Dom", "Wr_Flex_Var_Dom",
    "Wr_Dev_Mean_Dom",  "Wr_Dev_Var_Dom",
    "UA_Mean_Dom",      "Var_UA_Dom",
    "LA_Mean_Dom",      "Var_LA_Dom",
    "Neck_Sag_Mean",    "Neck_Sag_Var",
    "Trunk_Sag_Mean",   "Trunk_Sag_Var",
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

# ── Composite analysis (cached) ──────────────────────────────────────────
@st.cache_data
def run_composite_analysis(clean_path):
    df_c = pd.read_csv(clean_path)
    df_c["Condition"] = pd.Categorical(df_c["Condition"], categories=["baseline","workload"], ordered=True)

    domains = {
        "movement": ["SPARC_Dom","Jerk_Dom","VelInv_Dom","Dwell_Dom",
                     "Vel_Mean_Dom","Vel_Peak_Dom","Vel_Var_Dom","PathLen_Dom"],
        "wrist":    ["Wr_Flex_Mean_Dom","Wr_Flex_Var_Dom","Wr_Dev_Mean_Dom","Wr_Dev_Var_Dom"],
        "posture":  ["Neck_Sag_Mean","Neck_Sag_Var","Trunk_Sag_Mean","Trunk_Sag_Var"],
        "arm":      ["UA_Mean_Dom","Var_UA_Dom","LA_Mean_Dom","Var_LA_Dom"],
    }
    all_vars_c = [v for grp in domains.values() for v in grp] + ["RULA_Mean"]

    df_z = df_c.copy()
    for var in all_vars_c:
        mu = df_c[var].astype(float).mean()
        sd = df_c[var].astype(float).std(ddof=1)
        df_z[var] = (df_c[var].astype(float) - mu) / sd

    for domain, vars_list in domains.items():
        df_z[f"comp_{domain}"] = df_z[vars_list].mean(axis=1)
    df_z["comp_RULA"] = df_z["RULA_Mean"]

    composite_cols   = [f"comp_{d}" for d in domains] + ["comp_RULA"]
    composite_labels = list(domains.keys()) + ["RULA"]

    df_z_b  = df_z[df_z["Condition"]=="baseline"].set_index("Subject_ID")
    df_z_wl = df_z[df_z["Condition"]=="workload"].set_index("Subject_ID")
    common_z = df_z_b.index.intersection(df_z_wl.index)

    delta_comp = pd.DataFrame(index=common_z)
    for col in composite_cols:
        delta_comp[col] = df_z_wl.loc[common_z, col] - df_z_b.loc[common_z, col]

    comp_results = []
    for col, label in zip(composite_cols, composite_labels):
        d   = delta_comp[col].dropna().values
        b_v = df_z_b.loc[common_z, col].values.astype(float)
        w_v = df_z_wl.loc[common_z, col].values.astype(float)
        _, p_sw = shapiro(d)
        if p_sw > 0.05:
            stat, p   = ttest_rel(w_v, b_v)
            test_name = "t-test"
            eff       = np.mean(d) / np.std(d, ddof=1)
        else:
            stat, p   = wilcoxon(w_v, b_v, alternative="two-sided")
            test_name = "Wilcoxon"
            eff       = 1 - (2*stat) / (len(d)*(len(d)+1)/2)
        comp_results.append({"composite": label, "test": test_name,
                             "stat": round(stat,4), "p_raw": round(p,4),
                             "effect": round(eff,4)})

    comp_df = pd.DataFrame(comp_results)
    reject_c, p_fdr_c, _, _ = multipletests(comp_df["p_raw"], method="fdr_bh", alpha=0.05)
    comp_df["p_fdr"]   = p_fdr_c.round(4)
    comp_df["sig_fdr"] = reject_c

    sig_domains = [lbl for lbl, sig in zip(composite_labels, reject_c) if sig and lbl in domains]
    dir_tables = {}
    for label in sig_domains:
        dir_rows = []
        for var in domains[label]:
            d_var  = (df_z_wl.loc[common_z, var] - df_z_b.loc[common_z, var]).values.astype(float)
            mean_d = np.mean(d_var)
            _, p_sw = shapiro(d_var)
            if p_sw > 0.05:
                _, p_t = ttest_rel(df_z_wl.loc[common_z, var].values.astype(float),
                                    df_z_b.loc[common_z, var].values.astype(float))
            else:
                _, p_t = wilcoxon(df_z_wl.loc[common_z, var].values.astype(float),
                                   df_z_b.loc[common_z, var].values.astype(float),
                                   alternative="two-sided")
            dir_rows.append({"variabile": var,
                             "mean_delta_z": round(mean_d, 4),
                             "direzione": "↑ aumenta" if mean_d > 0 else "↓ diminuisce",
                             "p": round(p_t, 4)})
        dir_tables[label] = (pd.DataFrame(dir_rows)
                               .sort_values("mean_delta_z", key=abs, ascending=False)
                               .set_index("variabile"))

    return domains, composite_cols, composite_labels, comp_df, sig_domains, dir_tables, df_z_b, df_z_wl, common_z

# ── PCA within-domain + regression (cached) ──────────────────────────────
@st.cache_data
def run_pca_regression(clean_path):
    import statsmodels.formula.api as smf
    df_c = pd.read_csv(clean_path)
    df_c["Condition"] = pd.Categorical(df_c["Condition"], categories=["baseline","workload"], ordered=True)

    domains_pca = {
        "movement": ["SPARC_Dom","Jerk_Dom","VelInv_Dom","Dwell_Dom",
                     "Vel_Mean_Dom","Vel_Peak_Dom","Vel_Var_Dom","PathLen_Dom"],
        "wrist":    ["Wr_Flex_Mean_Dom","Wr_Flex_Var_Dom","Wr_Dev_Mean_Dom","Wr_Dev_Var_Dom"],
        "posture":  ["Neck_Sag_Mean","Neck_Sag_Var","Trunk_Sag_Mean","Trunk_Sag_Var"],
        "arm":      ["UA_Mean_Dom","Var_UA_Dom","LA_Mean_Dom","Var_LA_Dom"],
    }

    df_m = df_c.copy()
    df_m["condition_num"] = (df_m["Condition"] == "workload").astype(int)
    df_m["Subject_ID"]    = df_m["Subject_ID"].astype(str)
    pc1_loadings = {}
    var_exp_pc1  = {}

    for domain, vars_list in domains_pca.items():
        X      = df_c[vars_list].astype(float).values
        X_std  = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
        U, S, Vt = np.linalg.svd(X_std, full_matrices=False)
        ve     = (S[0]**2) / np.sum(S**2)
        pc1    = U[:, 0] * S[0]
        load1  = Vt[0, :]
        if np.corrcoef(pc1, df_m["condition_num"].values)[0, 1] < 0:
            pc1, load1 = -pc1, -load1
        df_m[f"PC1_{domain}"]  = pc1
        pc1_loadings[domain]   = pd.Series(load1, index=vars_list)
        var_exp_pc1[domain]    = ve

    rula_mu = df_c["RULA_Mean"].astype(float).mean()
    rula_sd = df_c["RULA_Mean"].astype(float).std(ddof=1)
    df_m["RULA_std"] = (df_c["RULA_Mean"].astype(float) - rula_mu) / rula_sd

    formula = ("condition_num ~ PC1_movement + PC1_wrist + PC1_posture + PC1_arm "
               "+ RULA_std + C(Subject_ID)")
    model = smf.ols(formula, data=df_m).fit()

    pred_names = ["PC1_movement","PC1_wrist","PC1_posture","PC1_arm","RULA_std"]
    reg_rows = []
    for pred in pred_names:
        beta = model.params[pred]
        p    = model.pvalues[pred]
        ci   = model.conf_int().loc[pred]
        reg_rows.append({"predittore": pred,
                         "β": round(beta,4),
                         "CI low":  round(ci.iloc[0],4),
                         "CI high": round(ci.iloc[1],4),
                         "p": round(p,4),
                         "sig": p < 0.05})
    reg_df = pd.DataFrame(reg_rows).set_index("predittore")

    sig_pred_domains = [p.replace("PC1_","") for p in pred_names
                        if p.startswith("PC1_") and model.pvalues[p] < 0.05]

    return domains_pca, pc1_loadings, var_exp_pc1, reg_df, sig_pred_domains, model.rsquared, len(df_m)

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
domains, composite_cols, composite_labels, comp_df, sig_domains, dir_tables, df_z_b, df_z_wl, common_z = run_composite_analysis(CLEAN_PATH)
domains_pca, pc1_loadings, var_exp_pc1, reg_df, sig_pred_domains, r_squared, n_obs = run_pca_regression(CLEAN_PATH)

# ── Tabs ─────────────────────────────────────────────────────
tabs = st.tabs([
    "📋 Dataset",
    "📊 Descrittive",
    "🔍 Outlier",
    "⚡ Effetto Condizione",
    "🔀 Order Effect",
    "🧮 PCA",
    "🎯 K-Means",
    "🔗 Correlazioni Delta",
    "🧩 Variabili Composite",
    "📐 PCA + Regressione",
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

# ── Tab 8: Correlazioni Delta ─────────────────────────────────
with tabs[7]:
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

# ── Tab 9: Variabili Composite ───────────────────────────────
with tabs[8]:
    st.header("Variabili composite (domain scores)")
    st.markdown(
        "Approccio per risolvere la **multicollinearità** tra variabili dello stesso dominio: "
        "ogni variabile è z-scorata sul dataset pooled, poi le z-score sono mediate per dominio. "
        "Riduzione da 21 test a **5** (4 composite + RULA)."
    )

    st.subheader("Composizione dei domini")
    domain_info = {
        "movement": ["SPARC_Dom","Jerk_Dom","VelInv_Dom","Dwell_Dom",
                     "Vel_Mean_Dom","Vel_Peak_Dom","Vel_Var_Dom","PathLen_Dom"],
        "wrist":    ["Wr_Flex_Mean_Dom","Wr_Flex_Var_Dom","Wr_Dev_Mean_Dom","Wr_Dev_Var_Dom"],
        "posture":  ["Neck_Sag_Mean","Neck_Sag_Var","Trunk_Sag_Mean","Trunk_Sag_Var"],
        "arm":      ["UA_Mean_Dom","Var_UA_Dom","LA_Mean_Dom","Var_LA_Dom"],
        "RULA (standalone)": ["RULA_Mean"],
    }
    domain_df = pd.DataFrame([
        {"Composite": k, "N variabili": len(v), "Variabili": ", ".join(v)}
        for k, v in domain_info.items()
    ]).set_index("Composite")
    st.dataframe(domain_df, width="stretch")

    st.divider()
    st.subheader("Risultati — effetto condizione")

    n_sig_c = comp_df["sig_fdr"].sum()
    c1, c2 = st.columns(2)
    c1.metric("Composite significativi (FDR)", f"{n_sig_c} / 5")
    c2.metric("Test totali (vs 21 variabili)", "5")

    st.dataframe(
        comp_df.set_index("composite").style.map(
            lambda v: "background-color: #d4f1d4" if v is True else "", subset=["sig_fdr"]
        ),
        width="stretch"
    )

    st.subheader("Effect size per composite")
    fig_ce, ax_ce = plt.subplots(figsize=(8, 3.5))
    colors_c = ["#2ECC71" if s else "#BDC3C7" for s in comp_df["sig_fdr"]]
    ax_ce.barh(comp_df["composite"], comp_df["effect"], color=colors_c,
               edgecolor="white", alpha=0.85)
    ax_ce.axvline(0, color="black", linewidth=0.8)
    for _, row in comp_df.iterrows():
        xoff = 0.02 if row["effect"] >= 0 else -0.02
        ha   = "left" if row["effect"] >= 0 else "right"
        ax_ce.text(row["effect"]+xoff,
                   comp_df.index[comp_df["composite"]==row["composite"]].tolist()[0],
                   f"q={row['p_fdr']:.3f}" + (" ✓" if row["sig_fdr"] else ""),
                   va="center", ha=ha, fontsize=8)
    ax_ce.set_xlabel("Effect size (d_z o r_rb)")
    ax_ce.set_title("Effetto condizione — composite scores\nverde = sig. FDR", fontweight="bold")
    ax_ce.spines["top"].set_visible(False); ax_ce.spines["right"].set_visible(False)
    fig_ce.tight_layout()
    st.pyplot(fig_ce)
    plt.close(fig_ce)

    st.subheader("Paired lines per composite")
    fig_cp, axes_cp = plt.subplots(1, len(composite_cols), figsize=(4*len(composite_cols), 4))
    for ax_cp, col, label in zip(axes_cp, composite_cols, composite_labels):
        b_vals = df_z_b.loc[common_z, col].values.astype(float)
        w_vals = df_z_wl.loc[common_z, col].values.astype(float)
        for bv, wv in zip(b_vals, w_vals):
            ax_cp.plot([0, 1], [bv, wv],
                       color="#E74C3C" if wv > bv else "#3498DB", alpha=0.45, linewidth=1)
        ax_cp.boxplot([b_vals, w_vals], positions=[0, 1], widths=0.3,
                      medianprops=dict(color="black", linewidth=2))
        ax_cp.set_xticks([0, 1]); ax_cp.set_xticklabels(["baseline", "workload"], fontsize=8)
        row = comp_df[comp_df["composite"] == label].iloc[0]
        ax_cp.set_title(f"{label}{' *' if row['sig_fdr'] else ''}\nq={row['p_fdr']:.3f}",
                        fontweight="bold", fontsize=9)
        ax_cp.spines["top"].set_visible(False); ax_cp.spines["right"].set_visible(False)
    fig_cp.suptitle("Composite scores — baseline vs workload", fontweight="bold")
    fig_cp.tight_layout()
    st.pyplot(fig_cp)
    plt.close(fig_cp)

    if sig_domains:
        st.divider()
        st.subheader("Direzione variabili interne nei composite significativi")
        for label in sig_domains:
            st.markdown(f"**{label}**")
            st.dataframe(dir_tables[label], width="stretch")
    else:
        st.info("Nessun composite significativo dopo FDR.")


# ── Tab 10: PCA + Regressione ────────────────────────────────
with tabs[9]:
    st.header("PCA within-domain + Regressione condizione")
    st.markdown(
        "Per ogni dominio viene estratta la **PC1** (combinazione lineare ottimale delle variabili). "
        "Le PC1 dei 4 domini + RULA sono usate come predittori in una **OLS con subject fixed effects**, "
        "che gestisce il design within-subject controllando le differenze individuali."
    )

    st.subheader("Varianza spiegata da PC1 per dominio")
    ve_df = pd.DataFrame([
        {"Dominio": d, "PC1 varianza (%)": round(v*100, 1), "N variabili": len(vars_list)}
        for (d, vars_list), v in zip(domains_pca.items(), var_exp_pc1.values())
    ]).set_index("Dominio")
    st.dataframe(ve_df, width="content")

    st.divider()
    st.subheader("Risultati regressione")
    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{r_squared:.3f}")
    c2.metric("Predittori significativi", f"{reg_df['sig'].sum()} / 5")
    c3.metric("Osservazioni", n_obs)

    st.dataframe(
        reg_df.style.map(lambda v: "background-color: #d4f1d4" if v is True else "",
                         subset=["sig"]),
        width="stretch"
    )

    st.subheader("β coefficients (con IC 95%)")
    fig_rb, ax_rb = plt.subplots(figsize=(8, 4))
    colors_rb = ["#2ECC71" if s else "#BDC3C7" for s in reg_df["sig"]]
    y_pos = list(range(len(reg_df)))
    ax_rb.barh(y_pos, reg_df["β"], color=colors_rb, edgecolor="white", alpha=0.85)
    ax_rb.errorbar(reg_df["β"], y_pos,
                   xerr=[reg_df["β"] - reg_df["CI low"], reg_df["CI high"] - reg_df["β"]],
                   fmt="none", color="black", linewidth=1.5, capsize=4)
    ax_rb.axvline(0, color="black", linewidth=0.8)
    ax_rb.set_yticks(y_pos); ax_rb.set_yticklabels(reg_df.index)
    ax_rb.set_xlabel("β (coefficiente di regressione)")
    ax_rb.set_title("Regressione condizione ~ PC1 domini\nverde = p<0.05", fontweight="bold")
    ax_rb.spines["top"].set_visible(False); ax_rb.spines["right"].set_visible(False)
    fig_rb.tight_layout()
    st.pyplot(fig_rb)
    plt.close(fig_rb)

    if sig_pred_domains:
        st.divider()
        st.subheader("Loadings PC1 — domini significativi")
        st.markdown("I loadings spiegano *cosa significa* un punteggio PC1 alto: "
                    "variabili con loading positivo ↑ aumentano nel workload, negative ↓ diminuiscono.")
        for domain in sig_pred_domains:
            st.markdown(f"**{domain}**")
            loads = pc1_loadings[domain].sort_values(key=abs, ascending=False)
            fig_ld, ax_ld = plt.subplots(figsize=(6, max(3, len(loads)*0.45)))
            colors_ld = ["#E74C3C" if v > 0 else "#3498DB" for v in loads.values]
            ax_ld.barh(loads.index, loads.values, color=colors_ld, edgecolor="white", alpha=0.85)
            ax_ld.axvline(0, color="black", linewidth=0.8)
            ax_ld.set_xlabel("Loading PC1")
            ax_ld.set_title(f"PC1 '{domain}' — rosso ↑ workload  |  blu ↓ workload",
                            fontweight="bold", fontsize=9)
            ax_ld.spines["top"].set_visible(False); ax_ld.spines["right"].set_visible(False)
            fig_ld.tight_layout()
            st.pyplot(fig_ld)
            plt.close(fig_ld)

        st.subheader("Loadings tutti i domini")
        fig_all, axes_all = plt.subplots(1, len(domains_pca), figsize=(4.5*len(domains_pca), 4))
        for ax_all, (domain, vars_list) in zip(axes_all, domains_pca.items()):
            loads = pc1_loadings[domain]
            is_sig = domain in sig_pred_domains
            c_all = ["#2ECC71" if is_sig else "#4C9BE8"] * len(loads)
            ax_all.barh(loads.index, loads.values, color=c_all, edgecolor="white", alpha=0.85)
            ax_all.axvline(0, color="black", linewidth=0.8)
            ax_all.set_title(f"{domain} ({var_exp_pc1[domain]*100:.0f}%)"
                             + (" ✓" if is_sig else ""),
                             fontweight="bold", fontsize=9)
            ax_all.set_xlabel("Loading", fontsize=8)
            ax_all.tick_params(axis="y", labelsize=7)
            ax_all.spines["top"].set_visible(False); ax_all.spines["right"].set_visible(False)
        fig_all.suptitle("Loadings PC1 per tutti i domini", fontweight="bold")
        fig_all.tight_layout()
        st.pyplot(fig_all)
        plt.close(fig_all)
    else:
        st.info("Nessun dominio significativo nella regressione (p<0.05).")


# ── Tab 11: Considerazioni Finali ────────────────────────────
with tabs[10]:
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
- **Neck_Sag_Var**: maggiore oscillazione sagittale del collo — la testa non mantiene una posizione stabile.

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
Neck_Sag_Var) forma un secondo pattern parzialmente indipendente.
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
                               "Jerk_Dom", "Vel_Var_Dom", "Var_LA_Dom", "Neck_Sag_Var"]
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
