import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import os

# --- 1. CONFIGURAZIONE ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

FOLDER_REPORTS  = os.path.join(script_dir, 'output_reports')
os.makedirs(FOLDER_REPORTS, exist_ok=True)
DWELL_THRESHOLD  = 0.05   # m/s


# --- 2. FUNZIONI DI BASE ---

def _read_csv(file_path):
    """
    Carica CSV da Processing.py o MediaPipeTrac.py.
    Gestisce formato standard (. decimale, virgola separatore)
    e formato Excel italiano (, decimale, ; separatore).

    Strategia: legge la prima riga e conta i separatori — il più frequente
    è quello usato dal file. Non si affida ai tipi delle colonne.
    """
    with open(file_path, 'r', encoding='utf-8-sig') as fh:
        first_line = fh.readline()

    n_semi  = first_line.count(';')
    n_comma = first_line.count(',')

    if n_semi > n_comma:
        # Formato Excel italiano: ; separatore, , decimale
        return pd.read_csv(file_path, sep=';', decimal=',', encoding='utf-8-sig')
    else:
        # Formato standard: , separatore, . decimale
        return pd.read_csv(file_path, sep=',', decimal='.', encoding='utf-8-sig')


def _lowpass(df, col, b, a):
    return filtfilt(b, a, df[col].interpolate().ffill().bfill().values)


def _velocity(xf, yf, zf, fs):
    """Velocità scalare con fs costante — stabile a ~10 Hz."""
    vx = np.gradient(xf, 1/fs)
    vy = np.gradient(yf, 1/fs)
    vz = np.gradient(zf, 1/fs)
    return np.sqrt(vx**2 + vy**2 + vz**2)


def _log_dim_jerk(xf, yf, zf, fs):
    """
    Log-dimensionless jerk (LDJ).
    Normalizza il jerk per durata² e ampiezza del movimento → confrontabile
    tra sessioni di durata e scala diverse. Più negativo = più fluido.

    Formula: LDJ = -log( (T^5 / A^2) * mean(jerk²) )
    dove T = durata, A = ampiezza 3D (deviazione standard delle coordinate).
    """
    dt = 1.0 / fs
    vx = np.gradient(xf, dt); vy = np.gradient(yf, dt); vz = np.gradient(zf, dt)
    ax = np.gradient(vx, dt); ay = np.gradient(vy, dt); az = np.gradient(vz, dt)
    jx = np.gradient(ax, dt); jy = np.gradient(ay, dt); jz = np.gradient(az, dt)
    jerk = np.nan_to_num(np.sqrt(jx**2 + jy**2 + jz**2))

    T = len(xf) / fs
    A = np.sqrt(np.var(xf) + np.var(yf) + np.var(zf))
    if A < 1e-6 or T < 1e-6:
        return float('nan'), jerk
    ldj = -np.log((T**5 / A**2) * np.mean(jerk**2) + 1e-10)
    return float(ldj), jerk


def _path_length(xf, yf, zf):
    return float(np.sum(np.sqrt(np.diff(xf)**2 + np.diff(yf)**2 + np.diff(zf)**2)))


def _sparc(vel, fs, fc=10.0, amp_th=0.05):
    """SPARC — fluidità spettrale. Più negativo = più scattoso."""
    nfft  = max(128, 2 ** int(np.ceil(np.log2(len(vel) * 4))))
    Mf    = np.abs(np.fft.rfft(vel, n=nfft)) / len(vel)
    freq  = np.fft.rfftfreq(nfft, d=1/fs)
    idx   = freq <= fc
    Mf, freq = Mf[idx], freq[idx]
    Mf_n  = Mf / (Mf[0] + 1e-10)
    below = np.where(Mf_n < amp_th)[0]
    cut   = below[0] if len(below) else len(Mf_n) - 1
    Mf_n, freq = Mf_n[:cut+1], freq[:cut+1]
    dl    = np.sqrt(np.diff(freq / fc)**2 + np.diff(Mf_n)**2)
    return float(-np.sum(dl))


def _velocity_peaks(vel, fs, min_dist_s=0.1):
    """Inversioni di velocità — più picchi = più scattoso."""
    min_dist = max(1, int(min_dist_s * fs))
    peaks, _ = find_peaks(vel, distance=min_dist, prominence=0.01)
    return int(len(peaks)), peaks


def _to_pct_time(time_cum, signal, n_pts=200):
    """Ricampiona un segnale su asse 0-100% del task."""
    pct         = time_cum / time_cum.iloc[-1] * 100
    pct_uniform = np.linspace(0, 100, n_pts)
    return pct_uniform, np.interp(pct_uniform, pct, signal)


def _dwell_intervals(time_cum, vel_avg, threshold=DWELL_THRESHOLD, min_dur=0.1):
    """
    Estrae gli intervalli di tempo in cui la velocità media bilaterale
    è sotto la soglia dwell per almeno min_dur secondi.

    Restituisce un DataFrame con colonne:
      t_start, t_end, duration_s, mean_vel, rula_mean (se disponibile)
    """
    below = vel_avg < threshold
    intervals = []
    in_dwell  = False
    t0 = 0.0

    t = time_cum.values
    b = below

    for i in range(len(t)):
        if b[i] and not in_dwell:
            in_dwell = True
            t0 = t[i]
        elif not b[i] and in_dwell:
            dur = t[i] - t0
            if dur >= min_dur:
                intervals.append({'t_start': round(t0, 3),
                                   't_end':   round(t[i], 3),
                                   'duration_s': round(dur, 3)})
            in_dwell = False

    # Chiudi eventuale intervallo aperto alla fine
    if in_dwell:
        dur = t[-1] - t0
        if dur >= min_dur:
            intervals.append({'t_start': round(t0, 3),
                               't_end':   round(t[-1], 3),
                               'duration_s': round(dur, 3)})

    return pd.DataFrame(intervals) if intervals else pd.DataFrame(
        columns=['t_start', 't_end', 'duration_s'])


# --- 3. ANALISI PER SESSIONE ---

def analyze_task_data(file_path):
    df = _read_csv(file_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%H:%M:%S.%f')
    df = df.sort_values('Timestamp').reset_index(drop=True)
    df['dt'] = df['Timestamp'].diff().dt.total_seconds()
    df['dt'] = df['dt'].fillna(df['dt'].median()).replace(0, 0.033)
    df['time_cum'] = df['dt'].cumsum()

    fs  = 1.0 / df['dt'].median()
    nyq = 0.5 * fs
    b, a = butter(2, min(1.5, nyq * 0.8) / nyq, btype='low')

    # Polso DESTRO
    xR = _lowpass(df, 'R_Wrist_X', b, a)
    yR = _lowpass(df, 'R_Wrist_Y', b, a)
    zR = _lowpass(df, 'R_Wrist_Z', b, a)
    vel_R = _velocity(xR, yR, zR, fs)
    ldj_R, jerk_R = _log_dim_jerk(xR, yR, zR, fs)

    # Polso SINISTRO
    xL = _lowpass(df, 'L_Wrist_X', b, a)
    yL = _lowpass(df, 'L_Wrist_Y', b, a)
    zL = _lowpass(df, 'L_Wrist_Z', b, a)
    vel_L = _velocity(xL, yL, zL, fs)
    ldj_L, jerk_L = _log_dim_jerk(xL, yL, zL, fs)

    duration  = df['time_cum'].iloc[-1]
    # Dwell calcolato separatamente per polso DX e SX
    dwell_pct_R = np.sum(df['dt'][vel_R < DWELL_THRESHOLD]) / duration * 100
    dwell_pct_L = np.sum(df['dt'][vel_L < DWELL_THRESHOLD]) / duration * 100
    dwell_intervals_R = _dwell_intervals(df['time_cum'], vel_R)
    dwell_intervals_L = _dwell_intervals(df['time_cum'], vel_L)
    mean_rula = df['RULA_C'].mean()

    sparc_R             = _sparc(vel_R, fs)
    sparc_L             = _sparc(vel_L, fs)
    n_peaks_R, pk_idx_R = _velocity_peaks(vel_R, fs)
    n_peaks_L, pk_idx_L = _velocity_peaks(vel_L, fs)

    # Colonne angolari — compatibilità CSV vecchi e nuovi
    if 'UA_Sag_R' in df.columns:
        ua_col_R, ua_col_L = 'UA_Sag_R', 'UA_Sag_L'
    elif 'UA_Ang_R' in df.columns:
        ua_col_R, ua_col_L = 'UA_Ang_R', 'UA_Ang_L'
    else:
        ua_col_R, ua_col_L = 'UA_Ang', 'UA_Ang'

    la_col_R = 'LA_Ang_R' if 'LA_Ang_R' in df.columns else None
    la_col_L = 'LA_Ang_L' if 'LA_Ang_L' in df.columns else la_col_R

    # Colonne polso (presenti solo nei CSV prodotti dalla versione aggiornata)
    has_wrist = all(c in df.columns for c in
                    ['Wr_Flex_R', 'Wr_Dev_R', 'Wr_Flex_L', 'Wr_Dev_L'])

    return {
        'df'          : df,
        'duration'    : duration,
        'fs'          : fs,
        'vel_R'       : vel_R,
        'vel_L'       : vel_L,
        'jerk_R'      : jerk_R,
        'jerk_L'      : jerk_L,
        'ldj_R'       : ldj_R,
        'ldj_L'       : ldj_L,
        '_xR': xR, '_yR': yR, '_zR': zR,
        '_xL': xL, '_yL': yL, '_zL': zL,
        'path_len_R'  : _path_length(xR, yR, zR),
        'path_len_L'  : _path_length(xL, yL, zL),
        'dwell_pct_R'       : dwell_pct_R,
        'dwell_pct_L'       : dwell_pct_L,
        'dwell_intervals_R' : dwell_intervals_R,
        'dwell_intervals_L' : dwell_intervals_L,
        'rula'        : mean_rula,
        'sparc_R'     : sparc_R,
        'sparc_L'     : sparc_L,
        'n_peaks_R'   : n_peaks_R,
        'n_peaks_L'   : n_peaks_L,
        'pk_idx_R'    : pk_idx_R,
        'pk_idx_L'    : pk_idx_L,
        'ua_std_R'    : df[ua_col_R].std(),
        'ua_std_L'    : df[ua_col_L].std(),
        'la_std_R'    : df[la_col_R].std() if la_col_R else float('nan'),
        'la_std_L'    : df[la_col_L].std() if la_col_L else float('nan'),
        '_ang_cols'   : (ua_col_R, ua_col_L),
        'has_wrist'   : has_wrist,
        'wr_flex_std_R': df['Wr_Flex_R'].std() if has_wrist else float('nan'),
        'wr_dev_std_R' : df['Wr_Dev_R'].std()  if has_wrist else float('nan'),
        'wr_flex_std_L': df['Wr_Flex_L'].std() if has_wrist else float('nan'),
        'wr_dev_std_L' : df['Wr_Dev_L'].std()  if has_wrist else float('nan'),
    }


# --- 4. ANALISI PRINCIPALE ---

def _dtw_path_3d(xb, yb, zb, xw, yw, zw):
    """DTW 3D — distanza media normalizzata in metri."""
    a = np.vstack([xb, yb, zb]).T
    b = np.vstack([xw, yw, zw]).T
    n, m = len(a), len(b)
    dtw  = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = float(np.linalg.norm(a[i-1] - b[j-1]))
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    return dtw[n, m] / (n + m)


def run_analysis(file_base, file_workload):

    # ── Percorsi di output ────────────────────────────────────────────────────
    base_name     = os.path.splitext(os.path.basename(file_base))[0]
    workload_name = os.path.splitext(os.path.basename(file_workload))[0]
    session_name  = f"{base_name}_vs_{workload_name}"

    session_folder = os.path.join(FOLDER_REPORTS, session_name)
    os.makedirs(session_folder, exist_ok=True)

    report_image_1 = os.path.join(session_folder, f"{session_name}_Cinematica.png")
    report_image_2 = os.path.join(session_folder, f"{session_name}_Angoli_Jerk.png")
    report_table   = os.path.join(session_folder, f"{session_name}_Comparazione.csv")
    dwell_csv      = os.path.join(session_folder, f"{session_name}_DwellIntervals.csv")

    res_b = analyze_task_data(file_base)
    res_w = analyze_task_data(file_workload)

    print("Calcolo DTW traiettorie 3D...")
    dtw_R = _dtw_path_3d(res_b['_xR'], res_b['_yR'], res_b['_zR'],
                          res_w['_xR'], res_w['_yR'], res_w['_zR'])
    dtw_L = _dtw_path_3d(res_b['_xL'], res_b['_yL'], res_b['_zL'],
                          res_w['_xL'], res_w['_yL'], res_w['_zL'])
    print(f"  DTW DX: {dtw_R:.4f} m  |  DTW SX: {dtw_L:.4f} m")

    pct_axis = np.linspace(0, 100, 200)

    # ── Tabella ───────────────────────────────────────────────────────────────
    metrics = {
        'Metrica': [
            'Time to Complete (s)',
            'SPARC DX (fluido=meno negativo)',
            'SPARC SX (fluido=meno negativo)',
            'Log-dim Jerk DX (fluido=piu negativo)',
            'Log-dim Jerk SX (fluido=piu negativo)',
            'Inversioni velocita DX (n)',
            'Inversioni velocita SX (n)',
            'DTW traiettoria DX (m)',
            'DTW traiettoria SX (m)',
            'Dwell Time DX (% task)',
            'Dwell Time SX (% task)',
            'RULA Medio (worst side)',
            'SD Upper Arm DX (deg)',
            'SD Upper Arm SX (deg)',
            'SD Lower Arm DX (deg)',
            'SD Lower Arm SX (deg)',
        ],
        'Base': [
            res_b['duration'],
            res_b['sparc_R'],  res_b['sparc_L'],
            res_b['ldj_R'],    res_b['ldj_L'],
            res_b['n_peaks_R'], res_b['n_peaks_L'],
            0.0, 0.0,
            res_b['dwell_pct_R'], res_b['dwell_pct_L'],
            res_b['rula'],
            res_b['ua_std_R'], res_b['ua_std_L'],
            res_b['la_std_R'], res_b['la_std_L'],
        ],
        'Workload': [
            res_w['duration'],
            res_w['sparc_R'],  res_w['sparc_L'],
            res_w['ldj_R'],    res_w['ldj_L'],
            res_w['n_peaks_R'], res_w['n_peaks_L'],
            dtw_R, dtw_L,
            res_w['dwell_pct_R'], res_w['dwell_pct_L'],
            res_w['rula'],
            res_w['ua_std_R'], res_w['ua_std_L'],
            res_w['la_std_R'], res_w['la_std_L'],
        ],
    }

    results_df = pd.DataFrame(metrics)
    results_df['Variazione %'] = results_df.apply(
        lambda r: ((r['Workload'] - r['Base']) / abs(r['Base']) * 100)
                  if r['Base'] not in (0, 0.0) else float('inf'),
        axis=1
    )

    results_df.to_csv(report_table, index=False)
    print(f"\nTabella salvata in: {report_table}")
    print(results_df.to_string(index=False))

    # ── Intervalli Dwell — separati per polso DX e SX ──────────────────────
    def _enrich(dw, res, sessione, lato):
        """Aggiunge sessione, lato e RULA medio a ogni intervallo dwell."""
        if dw.empty:
            return dw
        out = dw.copy()
        for idx, row in out.iterrows():
            mask = ((res['df']['time_cum'] >= row['t_start']) &
                    (res['df']['time_cum'] <= row['t_end']))
            out.loc[idx, 'rula_medio'] = res['df'].loc[mask, 'RULA_C'].mean()
        out.insert(0, 'lato',     lato)
        out.insert(0, 'sessione', sessione)
        return out

    dwell_all = pd.concat([
        _enrich(res_b['dwell_intervals_R'], res_b, 'Base',     'DX'),
        _enrich(res_b['dwell_intervals_L'], res_b, 'Base',     'SX'),
        _enrich(res_w['dwell_intervals_R'], res_w, 'Workload', 'DX'),
        _enrich(res_w['dwell_intervals_L'], res_w, 'Workload', 'SX'),
    ], ignore_index=True)

    dwell_all.to_csv(dwell_csv, index=False)

    print(f"\n=== INTERVALLI DWELL (soglia {DWELL_THRESHOLD} m/s, durata min 0.1s) ===")
    if dwell_all.empty:
        print("  Nessun intervallo rilevato — abbassa DWELL_THRESHOLD o min_dur.")
    else:
        print(dwell_all.to_string(index=False))
    print(f"\nIntervalli dwell salvati in: {dwell_csv}")

    # ── Helper asse % tempo ───────────────────────────────────────────────────
    tc_b_pct = res_b['df']['time_cum'] / res_b['df']['time_cum'].iloc[-1] * 100
    tc_w_pct = res_w['df']['time_cum'] / res_w['df']['time_cum'].iloc[-1] * 100

    col_R_b, col_L_b = res_b['_ang_cols']
    col_R_w, col_L_w = res_w['_ang_cols']
    label_ua = 'UA Sag' if 'Sag' in col_R_b else 'UA Ang'

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURA 1 — Cinematica: RULA · SPARC · Velocità · Stabilità UA · Var%
    # ══════════════════════════════════════════════════════════════════════════
    fig1, ax1 = plt.subplots(3, 2, figsize=(18, 17))
    plt.subplots_adjust(hspace=0.45, wspace=0.35)

    # A. RULA nel tempo (asse reale)
    ax1[0, 0].step(res_b['df']['time_cum'], res_b['df']['RULA_C'],
                   label=f"Base ({res_b['duration']:.1f}s)")
    ax1[0, 0].step(res_w['df']['time_cum'], res_w['df']['RULA_C'],
                   label=f"Workload ({res_w['duration']:.1f}s)", alpha=0.7)
    ax1[0, 0].axhline(4, color='gray', ls='--', lw=0.8, label='Soglia rischio (4)')
    ax1[0, 0].set_title('Evoluzione RULA e Tempo di Completamento')
    ax1[0, 0].set_ylabel('RULA score')
    ax1[0, 0].set_ylim(0, 8)
    ax1[0, 0].legend()

    # B. SPARC — barre affiancate DX/SX
    x_pos = np.arange(2)
    w = 0.35
    bars_R = ax1[0, 1].bar(x_pos - w/2,
                            [res_b['sparc_R'], res_w['sparc_R']], w,
                            color=['#2a6fa8', '#c0392b'])
    bars_L = ax1[0, 1].bar(x_pos + w/2,
                            [res_b['sparc_L'], res_w['sparc_L']], w,
                            color=['#1a9e75', '#8e44ad'])
    ax1[0, 1].set_xticks(x_pos)
    ax1[0, 1].set_xticklabels(['Base', 'Workload'])
    ax1[0, 1].set_title('SPARC — Fluidità del movimento\n(più negativo = più scattoso)')
    ax1[0, 1].set_ylabel('SPARC')
    ax1[0, 1].legend([bars_R[0], bars_L[0]], ['Polso DX', 'Polso SX'], title='Lato')
    for bar in list(bars_R) + list(bars_L):
        v = bar.get_height()
        ax1[0, 1].text(bar.get_x() + bar.get_width()/2,
                       v - 0.05, f'{v:.2f}',
                       ha='center', va='top', fontsize=8,
                       color='white' if v < -0.5 else 'black')

    # C. Inversioni velocità — polso SX (asse % tempo)
    _, vL_b = _to_pct_time(res_b['df']['time_cum'], res_b['vel_L'])
    _, vL_w = _to_pct_time(res_w['df']['time_cum'], res_w['vel_L'])
    pk_b_x  = tc_b_pct.iloc[res_b['pk_idx_L']].values
    pk_w_x  = tc_w_pct.iloc[res_w['pk_idx_L']].values

    ax1[1, 0].plot(pct_axis, vL_b, color='#2a6fa8', alpha=0.8,
                   label=f"Base ({res_b['n_peaks_L']} picchi)")
    ax1[1, 0].plot(pct_axis, vL_w, color='#c0392b', alpha=0.8,
                   label=f"Workload ({res_w['n_peaks_L']} picchi)")
    ax1[1, 0].plot(pk_b_x, res_b['vel_L'][res_b['pk_idx_L']], 'v', color='#2a6fa8', ms=7)
    ax1[1, 0].plot(pk_w_x, res_w['vel_L'][res_w['pk_idx_L']], 'v', color='#c0392b', ms=7)
    ax1[1, 0].axhline(DWELL_THRESHOLD, color='gray', ls='--', lw=0.8,
                      label=f'Soglia dwell ({DWELL_THRESHOLD} m/s)')
    ax1[1, 0].set_title('Inversioni velocità — Polso SINISTRO\n(triangoli = picchi locali)')
    ax1[1, 0].set_xlabel('Avanzamento task (%)')
    ax1[1, 0].set_ylabel('Velocità (m/s)')
    ax1[1, 0].legend(fontsize=8)

    # D. Inversioni velocità — polso DX (asse % tempo)
    _, vR_b = _to_pct_time(res_b['df']['time_cum'], res_b['vel_R'])
    _, vR_w = _to_pct_time(res_w['df']['time_cum'], res_w['vel_R'])
    pk_b_x_R = tc_b_pct.iloc[res_b['pk_idx_R']].values
    pk_w_x_R = tc_w_pct.iloc[res_w['pk_idx_R']].values

    ax1[1, 1].plot(pct_axis, vR_b, color='#2a6fa8', alpha=0.8,
                   label=f"Base ({res_b['n_peaks_R']} picchi)")
    ax1[1, 1].plot(pct_axis, vR_w, color='#c0392b', alpha=0.8,
                   label=f"Workload ({res_w['n_peaks_R']} picchi)")
    ax1[1, 1].plot(pk_b_x_R, res_b['vel_R'][res_b['pk_idx_R']], 'v', color='#2a6fa8', ms=7)
    ax1[1, 1].plot(pk_w_x_R, res_w['vel_R'][res_w['pk_idx_R']], 'v', color='#c0392b', ms=7)
    ax1[1, 1].axhline(DWELL_THRESHOLD, color='gray', ls='--', lw=0.8,
                      label=f'Soglia dwell ({DWELL_THRESHOLD} m/s)')
    ax1[1, 1].set_title('Inversioni velocità — Polso DESTRO\n(triangoli = picchi locali)')
    ax1[1, 1].set_xlabel('Avanzamento task (%)')
    ax1[1, 1].set_ylabel('Velocità (m/s)')
    ax1[1, 1].legend(fontsize=8)

    # E. Stabilità Angolare Upper Arm — asse % tempo (normalizzato)
    _, ua_R_b_pct = _to_pct_time(res_b['df']['time_cum'], res_b['df'][col_R_b].values)
    _, ua_R_w_pct = _to_pct_time(res_w['df']['time_cum'], res_w['df'][col_R_w].values)
    _, ua_L_b_pct = _to_pct_time(res_b['df']['time_cum'], res_b['df'][col_L_b].values)
    _, ua_L_w_pct = _to_pct_time(res_w['df']['time_cum'], res_w['df'][col_L_w].values)

    ax1[2, 0].plot(pct_axis, ua_R_b_pct,
                   label=f"Base DX (SD={res_b['ua_std_R']:.1f}°)", color='#2a6fa8')
    ax1[2, 0].plot(pct_axis, ua_R_w_pct,
                   label=f"Workload DX (SD={res_w['ua_std_R']:.1f}°)", color='#c0392b', alpha=0.7)
    ax1[2, 0].plot(pct_axis, ua_L_b_pct,
                   label=f"Base SX (SD={res_b['ua_std_L']:.1f}°)", color='#1a9e75', ls='--')
    ax1[2, 0].plot(pct_axis, ua_L_w_pct,
                   label=f"Workload SX (SD={res_w['ua_std_L']:.1f}°)", color='#8e44ad', ls='--', alpha=0.7)
    ax1[2, 0].set_title(f'Stabilità Angolare {label_ua} — tempo normalizzato')
    ax1[2, 0].set_xlabel('Avanzamento task (%)')
    ax1[2, 0].set_ylabel('Angolo (°)')
    ax1[2, 0].legend(fontsize=8)

    # F. Variazione % riepilogo
    plot_df = results_df[~results_df['Variazione %'].isin([float('inf'), float('-inf')])]
    def _bar_color(row):
        if 'SPARC' in row['Metrica'] or 'Jerk' in row['Metrica']:
            return '#c0392b' if row['Variazione %'] < 0 else '#2a6fa8'
        return '#c0392b' if row['Variazione %'] > 0 else '#2a6fa8'
    colors = [_bar_color(row) for _, row in plot_df.iterrows()]
    ax1[2, 1].barh(plot_df['Metrica'], plot_df['Variazione %'], color=colors)
    ax1[2, 1].set_title('Variazione % Workload vs Base\n(rosso=peggioramento, blu=miglioramento)')
    ax1[2, 1].axvline(0, color='black', lw=0.8)
    ax1[2, 1].tick_params(axis='y', labelsize=7)

    fig1.suptitle('Report Analisi Cinematica — Baseline vs Workload',
                  fontsize=14, fontweight='bold')
    fig1.savefig(report_image_1, dpi=150, bbox_inches='tight')
    print(f"\nFigura 1 salvata: {report_image_1}")

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURA 2 — Angoli corporei + Log-dimensionless Jerk
    # ══════════════════════════════════════════════════════════════════════════
    fig2, ax2 = plt.subplots(3, 2, figsize=(18, 17))
    plt.subplots_adjust(hspace=0.45, wspace=0.35)

    df_b = res_b['df']
    df_w = res_w['df']

    # Controlla disponibilità colonne angoli sagittali
    has_neck  = 'Neck_Sag'  in df_b.columns and 'Neck_Sag'  in df_w.columns
    has_trunk = 'Trunk_Sag' in df_b.columns and 'Trunk_Sag' in df_w.columns
    has_ua_sag = 'UA_Sag_R' in df_b.columns and 'UA_Sag_R' in df_w.columns

    # G. Collo e Trunk sagittali (asse % tempo)
    if has_neck or has_trunk:
        if has_neck:
            _, nk_b = _to_pct_time(df_b['time_cum'], df_b['Neck_Sag'].values)
            _, nk_w = _to_pct_time(df_w['time_cum'], df_w['Neck_Sag'].values)
            ax2[0, 0].plot(pct_axis, nk_b, color='#2a6fa8', label='Neck Base')
            ax2[0, 0].plot(pct_axis, nk_w, color='#c0392b', label='Neck Workload', alpha=0.8)
        if has_trunk:
            _, tr_b = _to_pct_time(df_b['time_cum'], df_b['Trunk_Sag'].values)
            _, tr_w = _to_pct_time(df_w['time_cum'], df_w['Trunk_Sag'].values)
            ax2[0, 0].plot(pct_axis, tr_b, color='#1a9e75', ls='--', label='Trunk Base')
            ax2[0, 0].plot(pct_axis, tr_w, color='#8e44ad', ls='--', label='Trunk Workload', alpha=0.8)
        ax2[0, 0].set_title('Angoli Collo e Trunk — piano sagittale\ntempo normalizzato')
        ax2[0, 0].set_xlabel('Avanzamento task (%)')
        ax2[0, 0].set_ylabel('Angolo (°)')
        ax2[0, 0].legend(fontsize=8)
    else:
        ax2[0, 0].text(0.5, 0.5, 'Neck_Sag / Trunk_Sag\nnon disponibili nel CSV',
                       ha='center', va='center', transform=ax2[0, 0].transAxes,
                       color='gray', fontsize=11)
        ax2[0, 0].set_title('Angoli Collo e Trunk')

    # H. Upper Arm DX — sagittale (asse % tempo)
    if has_ua_sag:
        _, ua_sag_r_b = _to_pct_time(df_b['time_cum'], df_b['UA_Sag_R'].values)
        _, ua_sag_r_w = _to_pct_time(df_w['time_cum'], df_w['UA_Sag_R'].values)
        ax2[0, 1].plot(pct_axis, ua_sag_r_b, color='#2a6fa8', label='Base')
        ax2[0, 1].plot(pct_axis, ua_sag_r_w, color='#c0392b', label='Workload', alpha=0.8)
        ax2[0, 1].axhline(0,  color='gray', ls=':', lw=0.8)
        ax2[0, 1].axhline(90, color='gray', ls='--', lw=0.8, label='90°')
        ax2[0, 1].set_title('Upper Arm DX — piano sagittale\ntempo normalizzato')
        ax2[0, 1].set_xlabel('Avanzamento task (%)')
        ax2[0, 1].set_ylabel('Angolo (°)')
        ax2[0, 1].legend(fontsize=8)
    else:
        ax2[0, 1].text(0.5, 0.5, 'UA_Sag_R\nnon disponibile nel CSV',
                       ha='center', va='center', transform=ax2[0, 1].transAxes,
                       color='gray', fontsize=11)
        ax2[0, 1].set_title('Upper Arm DX — sagittale')

    # I. Upper Arm SX — sagittale (asse % tempo)
    if has_ua_sag:
        _, ua_sag_l_b = _to_pct_time(df_b['time_cum'], df_b['UA_Sag_L'].values)
        _, ua_sag_l_w = _to_pct_time(df_w['time_cum'], df_w['UA_Sag_L'].values)
        ax2[1, 0].plot(pct_axis, ua_sag_l_b, color='#1a9e75', label='Base')
        ax2[1, 0].plot(pct_axis, ua_sag_l_w, color='#8e44ad', label='Workload', alpha=0.8)
        ax2[1, 0].axhline(0,  color='gray', ls=':', lw=0.8)
        ax2[1, 0].axhline(90, color='gray', ls='--', lw=0.8, label='90°')
        ax2[1, 0].set_title('Upper Arm SX — piano sagittale\ntempo normalizzato')
        ax2[1, 0].set_xlabel('Avanzamento task (%)')
        ax2[1, 0].set_ylabel('Angolo (°)')
        ax2[1, 0].legend(fontsize=8)
    else:
        ax2[1, 0].text(0.5, 0.5, 'UA_Sag_L\nnon disponibile nel CSV',
                       ha='center', va='center', transform=ax2[1, 0].transAxes,
                       color='gray', fontsize=11)
        ax2[1, 0].set_title('Upper Arm SX — sagittale')

    # J. Angoli polso — flessione DX e SX (asse % tempo)
    has_wr_b = res_b['has_wrist']
    has_wr_w = res_w['has_wrist']
    if has_wr_b and has_wr_w:
        _, wf_r_b = _to_pct_time(df_b['time_cum'], df_b['Wr_Flex_R'].values)
        _, wf_r_w = _to_pct_time(df_w['time_cum'], df_w['Wr_Flex_R'].values)
        _, wf_l_b = _to_pct_time(df_b['time_cum'], df_b['Wr_Flex_L'].values)
        _, wf_l_w = _to_pct_time(df_w['time_cum'], df_w['Wr_Flex_L'].values)
        _, wd_r_b = _to_pct_time(df_b['time_cum'], df_b['Wr_Dev_R'].values)
        _, wd_r_w = _to_pct_time(df_w['time_cum'], df_w['Wr_Dev_R'].values)
        _, wd_l_b = _to_pct_time(df_b['time_cum'], df_b['Wr_Dev_L'].values)
        _, wd_l_w = _to_pct_time(df_w['time_cum'], df_w['Wr_Dev_L'].values)

        # Flessione (linee continue) + deviazione (linee tratteggiate)
        ax2[1, 1].plot(pct_axis, wf_r_b, color='#2a6fa8',
                       label=f"Flex DX Base (SD={res_b['wr_flex_std_R']:.1f}°)")
        ax2[1, 1].plot(pct_axis, wf_r_w, color='#c0392b', alpha=0.8,
                       label=f"Flex DX WL (SD={res_w['wr_flex_std_R']:.1f}°)")
        ax2[1, 1].plot(pct_axis, wf_l_b, color='#1a9e75',
                       label=f"Flex SX Base (SD={res_b['wr_flex_std_L']:.1f}°)")
        ax2[1, 1].plot(pct_axis, wf_l_w, color='#8e44ad', alpha=0.8,
                       label=f"Flex SX WL (SD={res_w['wr_flex_std_L']:.1f}°)")
        ax2[1, 1].plot(pct_axis, wd_r_b, color='#2a6fa8', ls='--', alpha=0.5,
                       label=f"Dev DX Base")
        ax2[1, 1].plot(pct_axis, wd_r_w, color='#c0392b', ls='--', alpha=0.4,
                       label=f"Dev DX WL")
        ax2[1, 1].plot(pct_axis, wd_l_b, color='#1a9e75', ls='--', alpha=0.5,
                       label=f"Dev SX Base")
        ax2[1, 1].plot(pct_axis, wd_l_w, color='#8e44ad', ls='--', alpha=0.4,
                       label=f"Dev SX WL")
        ax2[1, 1].axhline(0, color='gray', ls=':', lw=0.8, label='neutro')
        ax2[1, 1].axhline( 15, color='#c0392b', ls=':', lw=0.6, alpha=0.4)
        ax2[1, 1].axhline(-15, color='#c0392b', ls=':', lw=0.6, alpha=0.4,
                          label='soglia RULA ±15°')
        ax2[1, 1].set_title('Angoli Polso — flessione (continua) e deviazione (tratteggiata)\ntempo normalizzato')
        ax2[1, 1].set_xlabel('Avanzamento task (%)')
        ax2[1, 1].set_ylabel('Angolo (°)')
        ax2[1, 1].legend(fontsize=7, ncol=2)
    else:
        ax2[1, 1].text(0.5, 0.5,
                       'Wr_Flex / Wr_Dev\nnon disponibili nel CSV\n(riprocessare con Processing.py aggiornato)',
                       ha='center', va='center', transform=ax2[1, 1].transAxes,
                       color='gray', fontsize=10)
        ax2[1, 1].set_title('Angoli Polso')

    # K. Log-dimensionless Jerk DX — profilo temporale (asse % tempo, scala log)
    _, jk_R_b = _to_pct_time(res_b['df']['time_cum'],
                              np.nan_to_num(res_b['jerk_R']))
    _, jk_R_w = _to_pct_time(res_w['df']['time_cum'],
                              np.nan_to_num(res_w['jerk_R']))
    jk_R_b = np.clip(jk_R_b, 1e-6, None)
    jk_R_w = np.clip(jk_R_w, 1e-6, None)
    ax2[2, 0].plot(pct_axis, jk_R_b, color='#2a6fa8', alpha=0.8,
                   label=f"Base (LDJ={res_b['ldj_R']:.1f})")
    ax2[2, 0].plot(pct_axis, jk_R_w, color='#c0392b', alpha=0.8,
                   label=f"Workload (LDJ={res_w['ldj_R']:.1f})")
    ax2[2, 0].set_yscale('log')
    ax2[2, 0].set_title('Jerk Polso DESTRO — tempo normalizzato\n(LDJ in legenda)')
    ax2[2, 0].set_xlabel('Avanzamento task (%)')
    ax2[2, 0].set_ylabel('Jerk (m/s³)')
    ax2[2, 0].legend(fontsize=8)

    # L. Log-dimensionless Jerk SX — profilo temporale (asse % tempo, scala log)
    _, jk_L_b = _to_pct_time(res_b['df']['time_cum'],
                              np.nan_to_num(res_b['jerk_L']))
    _, jk_L_w = _to_pct_time(res_w['df']['time_cum'],
                              np.nan_to_num(res_w['jerk_L']))
    jk_L_b = np.clip(jk_L_b, 1e-6, None)
    jk_L_w = np.clip(jk_L_w, 1e-6, None)
    ax2[2, 1].plot(pct_axis, jk_L_b, color='#1a9e75', alpha=0.8,
                   label=f"Base (LDJ={res_b['ldj_L']:.1f})")
    ax2[2, 1].plot(pct_axis, jk_L_w, color='#8e44ad', alpha=0.8,
                   label=f"Workload (LDJ={res_w['ldj_L']:.1f})")
    ax2[2, 1].set_yscale('log')
    ax2[2, 1].set_title('Jerk Polso SINISTRO — tempo normalizzato\n(LDJ in legenda)')
    ax2[2, 1].set_xlabel('Avanzamento task (%)')
    ax2[2, 1].set_ylabel('Jerk (m/s³)')
    ax2[2, 1].legend(fontsize=8)

    fig2.suptitle('Report Angoli, Polso e Jerk — Baseline vs Workload',
                  fontsize=14, fontweight='bold')
    fig2.savefig(report_image_2, dpi=150, bbox_inches='tight')
    print(f"Figura 2 salvata: {report_image_2}")

    plt.show()
    return results_df, res_b, res_w


# Per usare questo modulo, importalo e chiama run_analysis() con i tuoi percorsi:
#
#   from AnalisiDati import run_analysis
#   results_df, res_b, res_w = run_analysis("output_csv/Base.csv", "output_csv/Workload.csv")