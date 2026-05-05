"""
analisi_sessione.py
===================
Analisi ergonomica di una singola sessione (montaggio / smontaggio).
Legge un CSV prodotto da process_pump.py e genera:
  - output_reports/<nome>_cinematica.png   (6 grafici cinematici)
  - output_reports/<nome>_angoli.png       (6 grafici angolari)
  - output_reports/<nome>_metriche.csv     (tabella riepilogativa)
  - output_reports/<nome>_dwell.csv        (intervalli di pausa)

Uso:
  python analisi_sessione.py output_csv/montaggio_001.csv
  python analisi_sessione.py                              # ultimo CSV generato
"""

import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

# =============================================================================
# CONFIGURAZIONE
# =============================================================================

_SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
FOLDER_REPORTS = os.path.join(_SCRIPT_DIR, 'output_reports')
os.makedirs(FOLDER_REPORTS, exist_ok=True)

DWELL_THRESHOLD = 0.05   # m/s — soglia pausa

# =============================================================================
# I/O
# =============================================================================

def _find_latest_csv():
    csvs = sorted(glob.glob(os.path.join(_SCRIPT_DIR, 'output_csv', '*.csv')),
                  key=os.path.getmtime)
    if not csvs:
        raise FileNotFoundError(f'Nessun CSV in {_SCRIPT_DIR}/output_csv/')
    return csvs[-1]


def _read_csv(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as fh:
        first_line = fh.readline()
    sep = ';' if first_line.count(';') > first_line.count(',') else ','
    dec = ',' if sep == ';' else '.'
    return pd.read_csv(file_path, sep=sep, decimal=dec, encoding='utf-8-sig')

# =============================================================================
# METRICHE (stesse formule di AnalisiDati.py)
# =============================================================================

def _lowpass(series, b, a):
    return filtfilt(b, a, series.interpolate().ffill().bfill().values)


def _velocity(xf, yf, zf, fs):
    vx = np.gradient(xf, 1/fs)
    vy = np.gradient(yf, 1/fs)
    vz = np.gradient(zf, 1/fs)
    return np.sqrt(vx**2 + vy**2 + vz**2)


def _log_dim_jerk(xf, yf, zf, fs):
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
    min_dist = max(1, int(min_dist_s * fs))
    peaks, _ = find_peaks(vel, distance=min_dist, prominence=0.01)
    return int(len(peaks)), peaks


def _dwell_intervals(time_cum, vel, threshold=DWELL_THRESHOLD, min_dur=0.1):
    below = vel < threshold
    intervals, in_dwell, t0 = [], False, 0.0
    t = time_cum.values
    for i in range(len(t)):
        if below[i] and not in_dwell:
            in_dwell, t0 = True, t[i]
        elif not below[i] and in_dwell:
            dur = t[i] - t0
            if dur >= min_dur:
                intervals.append({'t_start': round(t0, 3),
                                   't_end':   round(t[i], 3),
                                   'duration_s': round(dur, 3)})
            in_dwell = False
    if in_dwell:
        dur = t[-1] - t0
        if dur >= min_dur:
            intervals.append({'t_start': round(t0, 3),
                               't_end':   round(t[-1], 3),
                               'duration_s': round(dur, 3)})
    return pd.DataFrame(intervals) if intervals else pd.DataFrame(
        columns=['t_start', 't_end', 'duration_s'])

# =============================================================================
# ANALISI
# =============================================================================

def analyze(file_path):
    df = _read_csv(file_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%H:%M:%S.%f')
    df = df.sort_values('Timestamp').reset_index(drop=True)
    df['dt'] = df['Timestamp'].diff().dt.total_seconds()
    df['dt'] = df['dt'].fillna(df['dt'].median()).replace(0, 0.033)
    df['time_cum'] = df['dt'].cumsum()

    fs  = 1.0 / df['dt'].median()
    nyq = 0.5 * fs
    b, a = butter(2, min(1.5, nyq * 0.8) / nyq, btype='low')

    xR = _lowpass(df['R_Wrist_X'], b, a)
    yR = _lowpass(df['R_Wrist_Y'], b, a)
    zR = _lowpass(df['R_Wrist_Z'], b, a)
    xL = _lowpass(df['L_Wrist_X'], b, a)
    yL = _lowpass(df['L_Wrist_Y'], b, a)
    zL = _lowpass(df['L_Wrist_Z'], b, a)

    vel_R = _velocity(xR, yR, zR, fs)
    vel_L = _velocity(xL, yL, zL, fs)
    ldj_R, jerk_R = _log_dim_jerk(xR, yR, zR, fs)
    ldj_L, jerk_L = _log_dim_jerk(xL, yL, zL, fs)

    duration = df['time_cum'].iloc[-1]
    dwell_pct_R = np.sum(df['dt'][vel_R < DWELL_THRESHOLD]) / duration * 100
    dwell_pct_L = np.sum(df['dt'][vel_L < DWELL_THRESHOLD]) / duration * 100
    dwell_int_R = _dwell_intervals(df['time_cum'], vel_R)
    dwell_int_L = _dwell_intervals(df['time_cum'], vel_L)

    sparc_R         = _sparc(vel_R, fs)
    sparc_L         = _sparc(vel_L, fs)
    n_peaks_R, pk_R = _velocity_peaks(vel_R, fs)
    n_peaks_L, pk_L = _velocity_peaks(vel_L, fs)

    ua_col_R = 'UA_Sag_R' if 'UA_Sag_R' in df.columns else 'UA_Ang_R'
    ua_col_L = 'UA_Sag_L' if 'UA_Sag_L' in df.columns else 'UA_Ang_L'
    la_col_R = 'LA_Ang_R' if 'LA_Ang_R' in df.columns else None
    la_col_L = 'LA_Ang_L' if 'LA_Ang_L' in df.columns else None
    has_wrist = all(c in df.columns for c in
                    ['Wr_Flex_R', 'Wr_Dev_R', 'Wr_Flex_L', 'Wr_Dev_L'])
    has_trunk = 'Trunk_Sag' in df.columns
    has_neck  = 'Neck_Sag'  in df.columns

    return {
        'df': df, 'fs': fs, 'duration': duration,
        'vel_R': vel_R, 'vel_L': vel_L,
        'jerk_R': jerk_R, 'jerk_L': jerk_L,
        'ldj_R': ldj_R, 'ldj_L': ldj_L,
        'sparc_R': sparc_R, 'sparc_L': sparc_L,
        'n_peaks_R': n_peaks_R, 'n_peaks_L': n_peaks_L,
        'pk_R': pk_R, 'pk_L': pk_L,
        'path_R': _path_length(xR, yR, zR),
        'path_L': _path_length(xL, yL, zL),
        'dwell_pct_R': dwell_pct_R, 'dwell_pct_L': dwell_pct_L,
        'dwell_int_R': dwell_int_R, 'dwell_int_L': dwell_int_L,
        'ua_col_R': ua_col_R, 'ua_col_L': ua_col_L,
        'la_col_R': la_col_R, 'la_col_L': la_col_L,
        'has_wrist': has_wrist, 'has_trunk': has_trunk, 'has_neck': has_neck,
    }

# =============================================================================
# REPORT
# =============================================================================

def run_analysis(file_path):
    session_name = os.path.splitext(os.path.basename(file_path))[0]
    out_dir      = os.path.join(FOLDER_REPORTS, session_name)
    os.makedirs(out_dir, exist_ok=True)

    path_fig1    = os.path.join(out_dir, f'{session_name}_cinematica.png')
    path_fig2    = os.path.join(out_dir, f'{session_name}_angoli.png')
    path_metrics = os.path.join(out_dir, f'{session_name}_metriche.csv')
    path_dwell   = os.path.join(out_dir, f'{session_name}_dwell.csv')

    print(f'[LOAD] {file_path}')
    r  = analyze(file_path)
    df = r['df']
    t  = df['time_cum']

    # ── Tabella metriche ──────────────────────────────────────────────────────
    metrics = pd.DataFrame({
        'Metrica': [
            'Durata (s)',
            'Frequenza campionamento (Hz)',
            'RULA medio',
            'RULA massimo',
            '% frame RULA ≥ 5 (azione immediata)',
            'SPARC DX (fluido = meno negativo)',
            'SPARC SX',
            'Log-dim Jerk DX (fluido = più negativo)',
            'Log-dim Jerk SX',
            'Inversioni velocità DX (n)',
            'Inversioni velocità SX (n)',
            'Lunghezza traiettoria DX (m)',
            'Lunghezza traiettoria SX (m)',
            'Dwell Time DX (% task)',
            'Dwell Time SX (% task)',
            'Intervalli pausa DX (n)',
            'Intervalli pausa SX (n)',
            'SD Upper Arm DX (°)',
            'SD Upper Arm SX (°)',
        ],
        'Valore': [
            round(r['duration'], 2),
            round(r['fs'], 1),
            round(df['RULA_C'].mean(), 2),
            int(df['RULA_C'].max()),
            round((df['RULA_C'] >= 5).mean() * 100, 1),
            round(r['sparc_R'], 3),
            round(r['sparc_L'], 3),
            round(r['ldj_R'], 3) if not np.isnan(r['ldj_R']) else 'nan',
            round(r['ldj_L'], 3) if not np.isnan(r['ldj_L']) else 'nan',
            r['n_peaks_R'],
            r['n_peaks_L'],
            round(r['path_R'], 3),
            round(r['path_L'], 3),
            round(r['dwell_pct_R'], 1),
            round(r['dwell_pct_L'], 1),
            len(r['dwell_int_R']),
            len(r['dwell_int_L']),
            round(df[r['ua_col_R']].std(), 2),
            round(df[r['ua_col_L']].std(), 2),
        ]
    })
    metrics.to_csv(path_metrics, index=False)
    print(metrics.to_string(index=False))

    # ── Dwell intervals ───────────────────────────────────────────────────────
    dwell_all = pd.concat([
        r['dwell_int_R'].assign(lato='DX'),
        r['dwell_int_L'].assign(lato='SX'),
    ], ignore_index=True)
    dwell_all.to_csv(path_dwell, index=False)

    # =========================================================================
    # FIGURA 1 — Cinematica
    # =========================================================================
    fig1, axes = plt.subplots(3, 2, figsize=(18, 15))
    fig1.suptitle(f'Analisi Cinematica — {session_name}', fontsize=14, fontweight='bold')
    plt.subplots_adjust(hspace=0.45, wspace=0.35)

    C_DX = '#2a6fa8'
    C_SX = '#c0392b'

    # A. RULA nel tempo
    ax = axes[0, 0]
    ax.step(t, df['RULA_C'], color='#444444', linewidth=1.2, label='RULA_C')
    ax.axhline(4, color='orange', ls='--', lw=0.9, label='Soglia 4')
    ax.axhline(5, color='red',    ls='--', lw=0.9, label='Soglia 5')
    ax.fill_between(t, df['RULA_C'], alpha=0.12, color='#444444', step='pre')
    ax.set_title(f'RULA nel tempo  (medio={df["RULA_C"].mean():.2f}  max={df["RULA_C"].max()})')
    ax.set_xlabel('Tempo (s)')
    ax.set_ylabel('RULA score')
    ax.set_ylim(0, 9)
    ax.legend(fontsize=8)

    # B. Distribuzione RULA (istogramma)
    ax = axes[0, 1]
    counts = df['RULA_C'].value_counts().sort_index()
    colors_bar = ['#2ecc71' if v <= 2 else '#f39c12' if v <= 4 else '#e74c3c'
                  for v in counts.index]
    ax.bar(counts.index, counts.values / len(df) * 100, color=colors_bar, edgecolor='white')
    ax.set_title('Distribuzione RULA (% frame)')
    ax.set_xlabel('Score RULA')
    ax.set_ylabel('% frame')
    ax.set_xticks(range(1, 8))
    for v, c in zip(counts.index, counts.values):
        ax.text(v, c / len(df) * 100 + 0.3, f'{c/len(df)*100:.0f}%',
                ha='center', va='bottom', fontsize=8)

    # C. Velocità polso DX
    ax = axes[1, 0]
    ax.plot(t, r['vel_R'], color=C_DX, linewidth=1, alpha=0.85)
    ax.fill_between(t, r['vel_R'], alpha=0.15, color=C_DX)
    pk_t = t.iloc[r['pk_R']] if len(r['pk_R']) else []
    if len(pk_t):
        ax.plot(pk_t, r['vel_R'][r['pk_R']], 'v', color=C_DX, ms=6,
                label=f'{r["n_peaks_R"]} picchi')
    ax.axhline(DWELL_THRESHOLD, color='gray', ls='--', lw=0.8,
               label=f'Soglia dwell ({DWELL_THRESHOLD} m/s)')
    ax.set_title(f'Velocità polso DESTRO  (SPARC={r["sparc_R"]:.2f}  dwell={r["dwell_pct_R"]:.1f}%)')
    ax.set_xlabel('Tempo (s)')
    ax.set_ylabel('Velocità (m/s)')
    ax.legend(fontsize=8)

    # D. Velocità polso SX
    ax = axes[1, 1]
    ax.plot(t, r['vel_L'], color=C_SX, linewidth=1, alpha=0.85)
    ax.fill_between(t, r['vel_L'], alpha=0.15, color=C_SX)
    pk_t = t.iloc[r['pk_L']] if len(r['pk_L']) else []
    if len(pk_t):
        ax.plot(pk_t, r['vel_L'][r['pk_L']], 'v', color=C_SX, ms=6,
                label=f'{r["n_peaks_L"]} picchi')
    ax.axhline(DWELL_THRESHOLD, color='gray', ls='--', lw=0.8,
               label=f'Soglia dwell ({DWELL_THRESHOLD} m/s)')
    ax.set_title(f'Velocità polso SINISTRO  (SPARC={r["sparc_L"]:.2f}  dwell={r["dwell_pct_L"]:.1f}%)')
    ax.set_xlabel('Tempo (s)')
    ax.set_ylabel('Velocità (m/s)')
    ax.legend(fontsize=8)

    # E. Jerk DX
    ax = axes[2, 0]
    jk_R = np.clip(np.nan_to_num(r['jerk_R']), 1e-6, None)
    ax.plot(t, jk_R, color=C_DX, linewidth=0.8, alpha=0.85)
    ax.set_yscale('log')
    ax.set_title(f'Jerk polso DESTRO  (LDJ={r["ldj_R"]:.2f})')
    ax.set_xlabel('Tempo (s)')
    ax.set_ylabel('Jerk (m/s³)')

    # F. Jerk SX
    ax = axes[2, 1]
    jk_L = np.clip(np.nan_to_num(r['jerk_L']), 1e-6, None)
    ax.plot(t, jk_L, color=C_SX, linewidth=0.8, alpha=0.85)
    ax.set_yscale('log')
    ax.set_title(f'Jerk polso SINISTRO  (LDJ={r["ldj_L"]:.2f})')
    ax.set_xlabel('Tempo (s)')
    ax.set_ylabel('Jerk (m/s³)')

    fig1.savefig(path_fig1, dpi=150, bbox_inches='tight')
    print(f'\nFigura 1 salvata: {path_fig1}')

    # =========================================================================
    # FIGURA 2 — Angoli corporei
    # =========================================================================
    fig2, axes2 = plt.subplots(3, 2, figsize=(18, 15))
    fig2.suptitle(f'Angoli Corporei — {session_name}', fontsize=14, fontweight='bold')
    plt.subplots_adjust(hspace=0.45, wspace=0.35)

    # A. Upper Arm sagittale DX e SX
    ax = axes2[0, 0]
    ax.plot(t, df[r['ua_col_R']], color=C_DX, linewidth=1,
            label=f'DX  (SD={df[r["ua_col_R"]].std():.1f}°)')
    ax.plot(t, df[r['ua_col_L']], color=C_SX, linewidth=1, alpha=0.85,
            label=f'SX  (SD={df[r["ua_col_L"]].std():.1f}°)')
    ax.axhline(0,  color='gray', ls=':', lw=0.8)
    for _ang, _lbl in [(-20, '-20°'), (20, '20°'), (45, '45°'), (90, '90°')]:
        ax.axhline(_ang, color='#e67e22', ls='--', lw=0.9, label='_nolegend_')
        ax.text(t.iloc[-1], _ang, f' {_lbl}', va='center', fontsize=7, color='#e67e22')
    ax.set_title('Upper Arm — piano sagittale\n0°=fianco  +90°=avanti orizzontale')
    ax.set_xlabel('Tempo (s)')
    ax.set_ylabel('Angolo (°)')
    ax.legend(fontsize=8)

    # B. Upper Arm frontale DX e SX
    has_fro = 'UA_Fro_R' in df.columns and 'UA_Fro_L' in df.columns
    ax = axes2[0, 1]
    if has_fro:
        ax.plot(t, df['UA_Fro_R'], color=C_DX, linewidth=1,
                label=f'DX  (SD={df["UA_Fro_R"].std():.1f}°)')
        ax.plot(t, df['UA_Fro_L'], color=C_SX, linewidth=1, alpha=0.85,
                label=f'SX  (SD={df["UA_Fro_L"].std():.1f}°)')
        ax.axhline(0, color='gray', ls=':', lw=0.8)
        ax.set_title('Upper Arm — piano frontale\n0°=fianco  +90°=abduzione laterale')
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('Angolo (°)')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'UA frontale\nnon disponibile',
                ha='center', va='center', transform=ax.transAxes, color='gray')
        ax.set_title('Upper Arm — piano frontale')

    # C. Lower Arm (gomito) DX e SX
    ax = axes2[1, 0]
    if r['la_col_R'] and r['la_col_R'] in df.columns:
        ax.plot(t, df[r['la_col_R']], color=C_DX, linewidth=1,
                label=f'DX  (SD={df[r["la_col_R"]].std():.1f}°)')
        ax.plot(t, df[r['la_col_L']], color=C_SX, linewidth=1, alpha=0.85,
                label=f'SX  (SD={df[r["la_col_L"]].std():.1f}°)')
        for _ang, _lbl in [(60, '60°'), (100, '100°')]:
            ax.axhline(_ang, color='#e67e22', ls='--', lw=0.9, label='_nolegend_')
            ax.text(t.iloc[-1], _ang, f' {_lbl}', va='center', fontsize=7, color='#e67e22')
        ax.set_title('Gomito — flessione\n0°=braccio disteso  90°=gomito piegato')
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('Angolo (°)')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'LA angle\nnon disponibile',
                ha='center', va='center', transform=ax.transAxes, color='gray')
        ax.set_title('Gomito — flessione')

    # D. Collo e Tronco sagittali
    ax = axes2[1, 1]
    if r['has_neck']:
        ax.plot(t, df['Neck_Sag'], color='#8e44ad', linewidth=1,
                label=f'Collo  (SD={df["Neck_Sag"].std():.1f}°)')
    if r['has_trunk']:
        ax.plot(t, df['Trunk_Sag'], color='#16a085', linewidth=1, alpha=0.85,
                label=f'Tronco  (SD={df["Trunk_Sag"].std():.1f}°)')
    ax.axhline(0,  color='gray', ls=':', lw=0.8)
    ax.axhline(20, color='orange', ls='--', lw=0.8, label='20° (RULA)')
    ax.set_title('Collo e Tronco — piano sagittale\n0°=eretto  +20°=inclinato avanti')
    ax.set_xlabel('Tempo (s)')
    ax.set_ylabel('Angolo (°)')
    ax.legend(fontsize=8)

    # E. Polso DX — flessione e deviazione
    ax = axes2[2, 0]
    if r['has_wrist']:
        ax.plot(t, df['Wr_Flex_R'], color=C_DX, linewidth=1,
                label=f'Flessione  (SD={df["Wr_Flex_R"].std():.1f}°)')
        ax.plot(t, df['Wr_Dev_R'],  color=C_DX, linewidth=1, ls='--', alpha=0.7,
                label=f'Deviazione  (SD={df["Wr_Dev_R"].std():.1f}°)')
        ax.axhline(0,   color='gray',   ls=':', lw=0.8)
        ax.axhline( 15, color='orange', ls='--', lw=0.6, alpha=0.5)
        ax.axhline(-15, color='orange', ls='--', lw=0.6, alpha=0.5, label='±15° RULA')
        ax.set_title('Polso DESTRO — flessione (continua) e deviazione (tratteggiata)')
    else:
        ax.text(0.5, 0.5, 'Dati polso\nnon disponibili',
                ha='center', va='center', transform=ax.transAxes, color='gray')
        ax.set_title('Polso DESTRO')
    ax.set_xlabel('Tempo (s)')
    ax.set_ylabel('Angolo (°)')
    ax.legend(fontsize=8)

    # F. Polso SX — flessione e deviazione
    ax = axes2[2, 1]
    if r['has_wrist']:
        ax.plot(t, df['Wr_Flex_L'], color=C_SX, linewidth=1,
                label=f'Flessione  (SD={df["Wr_Flex_L"].std():.1f}°)')
        ax.plot(t, df['Wr_Dev_L'],  color=C_SX, linewidth=1, ls='--', alpha=0.7,
                label=f'Deviazione  (SD={df["Wr_Dev_L"].std():.1f}°)')
        ax.axhline(0,   color='gray',   ls=':', lw=0.8)
        ax.axhline( 15, color='orange', ls='--', lw=0.6, alpha=0.5)
        ax.axhline(-15, color='orange', ls='--', lw=0.6, alpha=0.5, label='±15° RULA')
        ax.set_title('Polso SINISTRO — flessione (continua) e deviazione (tratteggiata)')
    else:
        ax.text(0.5, 0.5, 'Dati polso\nnon disponibili',
                ha='center', va='center', transform=ax.transAxes, color='gray')
        ax.set_title('Polso SINISTRO')
    ax.set_xlabel('Tempo (s)')
    ax.set_ylabel('Angolo (°)')
    ax.legend(fontsize=8)

    fig2.savefig(path_fig2, dpi=150, bbox_inches='tight')
    print(f'Figura 2 salvata: {path_fig2}')
    print(f'Metriche  -> {path_metrics}')
    print(f'Dwell     -> {path_dwell}')

    plt.show()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        if not os.path.isabs(csv_path) and not os.path.exists(csv_path):
            csv_path = os.path.join(_SCRIPT_DIR, csv_path)
    else:
        csv_path = _find_latest_csv()
        print(f'[AUTO] Nessun argomento — uso: {csv_path}')

    run_analysis(csv_path)
