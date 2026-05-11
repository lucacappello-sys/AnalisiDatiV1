"""Build analysis_dataset.csv — N=18 participants, updated survey (0-100 scale)."""
import warnings, os, sys
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
warnings.filterwarnings('ignore')

PROJECT_DIR = r'C:\Users\luca2\Desktop\VRula\BodyTracking\separated_scripts'
os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)
os.makedirs('output_analysis', exist_ok=True)

DWELL_THRESHOLD = 0.05  # m/s

# survey_code -> (csv_number, order_group, dominant_hand)
# Order groups for 12-18: assumed alternating from 11 (A->B) to maintain balance
# Final tally: A->B = 9 (1,4,6,9,11,13,15,17), B->A = 9 (2,3,5,7,8,10,12,14,16,18) — NOTE: confirm with PI
PARTICIPANTS = {
    1:  ('01', 'A->B', 'SX'),
    2:  ('02', 'B->A', 'DX'),
    3:  ('03', 'B->A', 'DX'),
    4:  ('04', 'A->B', 'DX'),
    5:  ('05', 'B->A', 'DX'),
    6:  ('06', 'A->B', 'DX'),
    7:  ('07', 'B->A', 'DX'),
    8:  ('08', 'B->A', 'DX'),
    9:  ('09', 'A->B', 'SX'),
    10: ('10', 'B->A', 'DX'),
    11: ('11', 'A->B', 'DX'),
    12: ('12', 'B->A', 'DX'),
    13: ('13', 'A->B', 'DX'),
    14: ('14', 'B->A', 'DX'),
    15: ('15', 'A->B', 'DX'),
    16: ('16', 'B->A', 'DX'),
    17: ('17', 'A->B', 'DX'),
    18: ('18', 'B->A', 'DX'),
}

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def _read_csv(path):
    # Autodetect separator: CSVs italiani usano ';' con decimale ','
    with open(path, 'r', encoding='utf-8-sig') as f:
        first = f.readline()
    sep = ';' if first.count(';') > first.count(',') else ','
    dec = ',' if sep == ';' else '.'
    return pd.read_csv(path, sep=sep, decimal=dec, encoding='utf-8-sig')

def _lowpass(series, b, a):
    return filtfilt(b, a, series.interpolate().ffill().bfill().values)

def _velocity(xf, yf, zf, fs):
    vx = np.gradient(xf, 1/fs); vy = np.gradient(yf, 1/fs); vz = np.gradient(zf, 1/fs)
    return np.sqrt(vx**2 + vy**2 + vz**2)

def _path_length(xf, yf, zf):
    return float(np.sum(np.sqrt(np.diff(xf)**2 + np.diff(yf)**2 + np.diff(zf)**2)))

def _n_peaks(vel, fs):
    # distance=10% fs → picchi distanti almeno 100 ms; height=0.03 m/s filtra microtremori
    peaks, _ = find_peaks(vel, distance=int(0.1 * fs), height=0.03)
    return len(peaks)

def _vel_inversions(vel, fs):
    # prominence=0.01: conta inversioni anche piccole (fluidità del profilo)
    min_dist = max(1, int(0.1 * fs))
    peaks, _ = find_peaks(vel, distance=min_dist, prominence=0.01)
    return int(len(peaks))

def _log_dim_jerk(xf, yf, zf, fs):
    dt = 1.0 / fs
    vx = np.gradient(xf, dt); vy = np.gradient(yf, dt); vz = np.gradient(zf, dt)
    ax = np.gradient(vx, dt); ay = np.gradient(vy, dt); az = np.gradient(vz, dt)
    jx = np.gradient(ax, dt); jy = np.gradient(ay, dt); jz = np.gradient(az, dt)
    jerk = np.nan_to_num(np.sqrt(jx**2 + jy**2 + jz**2))
    T = len(xf) / fs
    A = np.sqrt(np.var(xf) + np.var(yf) + np.var(zf))
    if A < 1e-6 or T < 1e-6:
        return float('nan')
    return float(-np.log((T**5 / A**2) * np.mean(jerk**2) + 1e-10))

def _sparc(vel, fs, fc=20.0, amp_th=0.05):
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

def _dwell_pct(vel, dt_values, duration):
    return float(np.sum(dt_values[vel < DWELL_THRESHOLD]) / duration * 100)


# ─── PER-SESSION METRICS ─────────────────────────────────────────────────────

def compute_session_metrics(csv_path):
    df = _read_csv(csv_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%H:%M:%S.%f')
    df = df.sort_values('Timestamp').reset_index(drop=True)
    df['dt'] = df['Timestamp'].diff().dt.total_seconds()
    # Sostituisce 0 e NaN per evitare divisioni per zero nel calcolo di fs
    df['dt'] = df['dt'].fillna(df['dt'].median()).replace(0, df['dt'].median())
    fs = 1.0 / df['dt'].median()
    nyq = 0.5 * fs
    # Limita a 1.5 Hz (frequenza di taglio gesto umano); il cap a nyq*0.8 evita instabilità del filtro
    b, a = butter(2, min(1.5, nyq * 0.8) / nyq, btype='low')

    xR = _lowpass(df['R_Wrist_X'], b, a); yR = _lowpass(df['R_Wrist_Y'], b, a); zR = _lowpass(df['R_Wrist_Z'], b, a)
    xL = _lowpass(df['L_Wrist_X'], b, a); yL = _lowpass(df['L_Wrist_Y'], b, a); zL = _lowpass(df['L_Wrist_Z'], b, a)

    vel_R  = _velocity(xR, yR, zR, fs); vel_L  = _velocity(xL, yL, zL, fs)
    path_R = _path_length(xR, yR, zR);  path_L = _path_length(xL, yL, zL)

    duration = float(df['dt'].sum())
    dt_vals  = df['dt'].values

    g  = lambda col: float(df[col].mean()) if col in df.columns else float('nan')
    gs = lambda col: float(df[col].std())  if col in df.columns else float('nan')
    ua_col_R = next((c for c in ['UA_Sag_R', 'UA_Ang_R'] if c in df.columns), None)
    ua_col_L = next((c for c in ['UA_Sag_L', 'UA_Ang_L'] if c in df.columns), None)
    la_col_R = next((c for c in ['LA_Ang_R', 'LA_Sag_R'] if c in df.columns), None)
    la_col_L = next((c for c in ['LA_Ang_L', 'LA_Sag_L'] if c in df.columns), None)

    return {
        # kinematics
        'SPARC_DX':  _sparc(vel_R, fs),       'SPARC_SX':  _sparc(vel_L, fs),
        'Jerk_DX':   _log_dim_jerk(xR, yR, zR, fs), 'Jerk_SX': _log_dim_jerk(xL, yL, zL, fs),
        'VelInv_DX': _vel_inversions(vel_R, fs), 'VelInv_SX': _vel_inversions(vel_L, fs),
        'Dwell_DX':  _dwell_pct(vel_R, dt_vals, duration),
        'Dwell_SX':  _dwell_pct(vel_L, dt_vals, duration),
        'Vel_Mean_DX': float(np.mean(vel_R)), 'Vel_Mean_SX': float(np.mean(vel_L)),
        'Vel_Peak_DX': float(np.max(vel_R)),  'Vel_Peak_SX': float(np.max(vel_L)),
        'Vel_Var_DX':   float(np.std(vel_R)),  'Vel_Var_SX':   float(np.std(vel_L)),
        'PathLen_DX':  path_R,                 'PathLen_SX':  path_L,
        'RULA_Mean':   float(df['RULA_C'].mean()),
        'Wr_Flex_Mean_DX': g('Wr_Flex_R'), 'Wr_Flex_Var_DX': gs('Wr_Flex_R'),
        'Wr_Dev_Mean_DX':  g('Wr_Dev_R'),  'Wr_Dev_Var_DX':  gs('Wr_Dev_R'),
        'Wr_Flex_Mean_SX': g('Wr_Flex_L'), 'Wr_Flex_Var_SX': gs('Wr_Flex_L'),
        'Wr_Dev_Mean_SX':  g('Wr_Dev_L'),  'Wr_Dev_Var_SX':  gs('Wr_Dev_L'),
        'UA_Mean_DX':  g(ua_col_R),  'UA_Mean_SX':  g(ua_col_L),
        'Var_UA_DX':    gs(ua_col_R), 'Var_UA_SX':    gs(ua_col_L),
        'LA_Mean_DX':  g(la_col_R),  'LA_Mean_SX':  g(la_col_L),
        'Var_LA_DX':    gs(la_col_R), 'Var_LA_SX':    gs(la_col_L),
        'Neck_Sag_Mean':  g('Neck_Sag'),  'Neck_Sag_Var':  gs('Neck_Sag'),
        'Trunk_Sag_Mean': g('Trunk_Sag'), 'Trunk_Sag_Var': gs('Trunk_Sag'),
    }

# ─── SURVEY ──────────────────────────────────────────────────────────────────

def build_survey_domains(survey_path='output_analysis/SurveysData.csv'):
    df = pd.read_csv(survey_path, sep=',', decimal='.', encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    df = df[pd.to_numeric(df['Participant code'], errors='coerce').notna()].copy()
    df['Participant code'] = df['Participant code'].astype(float).astype(int)
    df['Condition'] = df['Condition'].str.strip().str.lower()
    num_cols = df.columns.drop(['Participant code', 'Condition'])
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

# ─── BUILD ───────────────────────────────────────────────────────────────────
per_session_rows = []
for survey_code, (csv_num, order_group, dom_hand) in sorted(PARTICIPANTS.items(), key=lambda x: x[1][0]):
    for condition in ['baseline', 'workload']:
        prefix   = 'Baseline' if condition == 'baseline' else 'Workload'
        csv_path = f'output_csv/{prefix}{csv_num}.csv'
        print(f'  {prefix}{csv_num} ... ', end='', flush=True)
        try:
            sess = compute_session_metrics(csv_path)
            row  = {
                'Subject_ID':  int(csv_num),
                'Survey_Code': survey_code,
                'Condition':   condition,
                'Order_Group': order_group,
                'Dom_Hand':    dom_hand,
            }
            row.update(sess)
            per_session_rows.append(row)
            print('OK')
        except Exception as e:
            print(f'ERRORE: {e}')

df_all = pd.DataFrame(per_session_rows)
print(f'\nSessioni processate: {len(df_all)}')

# Salva CSV per-sessione
for _, r in df_all.iterrows():
    prefix = 'Baseline' if r['Condition'] == 'baseline' else 'Workload'
    fname  = f"output_analysis/{prefix}{int(r['Subject_ID']):02d}_metrics.csv"
    pd.DataFrame([r]).to_csv(fname, index=False)
print('CSV per-sessione salvati in output_analysis/')

# Merge survey
survey = build_survey_domains('output_analysis/SurveysData.csv')
df_all = df_all.merge(
    survey.rename(columns={'Participant code': 'Survey_Code'}),
    on=['Survey_Code', 'Condition'],
    how='left'
)
missing = df_all['Awareness'].isna().sum()
print(f'Survey merge: {missing} righe mancanti' if missing else 'Survey merge: OK')

# Swap DX↔SX per i mancini in modo che _DX diventi sempre la mano dominante
base_cols = sorted({c.replace('_DX', '') for c in df_all.columns
                    if c.endswith('_DX') and c.replace('_DX', '') + '_SX' in df_all.columns})
left_mask = df_all['Dom_Hand'] == 'SX'
for base in base_cols:
    dx, sx = f'{base}_DX', f'{base}_SX'
    df_all.loc[left_mask, [dx, sx]] = df_all.loc[left_mask, [sx, dx]].values
rename_map = {f'{b}_DX': f'{b}_Dom' for b in base_cols}
rename_map.update({f'{b}_SX': f'{b}_NonDom' for b in base_cols})
df_all = df_all.rename(columns=rename_map)
df_all = df_all.drop(columns=[c for c in df_all.columns if c.endswith('_NonDom')])

df_all = df_all.sort_values(['Subject_ID', 'Condition']).reset_index(drop=True)
df_all.to_csv('output_analysis/analysis_dataset.csv', index=False)
print(f'\nDataset salvato: {df_all.shape[0]} righe x {df_all.shape[1]} colonne')
print(df_all[['Subject_ID','Condition','Order_Group','Dom_Hand','RULA_Mean','Awareness']].to_string(index=False))
