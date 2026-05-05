"""
Processing.py  —  Script 2 di 3
═══════════════════════════════════════════════════════════════════════════════
Processa offline i file registrati da Acquisizione.py.
Legge il video RGB (.avi), la depth stack (.depth.npy) e i timestamp
(.times.npy), esegue il rilevamento della postura con MediaPipe Pose,
calcola tutti gli angoli anatomici e il punteggio RULA frame per frame,
e salva i risultati in un file CSV pronto per AnalisiDati.py.

─── Input (cartella output_raw/) ─────────────────────────────────────────────
  <base>.avi            video RGB (XVID, registrato da Acquisizione.py)
  <base>.depth.npy      depth stack (N, H, W) float32 in metri
  <base>.times.npy      timestamp relativo di ogni frame (secondi)
  output_raw/gravity.npy  vettore gravità IMU (opzionale, fallback [0,1,0])

─── Output (cartella output_csv/) ────────────────────────────────────────────
  <base>.csv            colonne: Timestamp, Frame, RULA_C, Score_A_R,
                        Score_A_L, Score_B, angoli anatomici, landmark 3D

─── Uso da terminale ─────────────────────────────────────────────────────────
  python Processing.py output_raw/Base_Line_2026-03-17_10-00-00
  python Processing.py                  # processa l'ultimo .avi in output_raw/

─── Uso da notebook ──────────────────────────────────────────────────────────
  from processing import process_file, find_latest_base
  csv_path = process_file("output_raw/Base_Line_2026-03-17_10-00-00")

═══════════════════════════════════════════════════════════════════════════════
"""

import csv
import datetime
import glob
import os
import sys

import cv2
import mediapipe as mp
import numpy as np
import rula_logic


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURAZIONE
# ═══════════════════════════════════════════════════════════════════════════════

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
FOLDER_CSV   = os.path.join(_SCRIPT_DIR, 'output_csv')
FOLDER_VIDEO = os.path.join(_SCRIPT_DIR, 'output_video')
os.makedirs(FOLDER_CSV,   exist_ok=True)
os.makedirs(FOLDER_VIDEO, exist_ok=True)

# CSV compatibile con Excel italiano (True) o standard internazionale (False).
# True  → separatore colonne = ;   separatore decimale = ,
# False → separatore colonne = ,   separatore decimale = .
EXCEL_ITALIANO = True

# ── Profilo di performance ─────────────────────────────────────────────────────
# Cambia solo PERFORMANCE_PROFILE. Le costanti derivate si aggiornano in automatico.
#
#  'accuracy'  complexity=2  640×480  static=True   ~10-22 fps
#              Massima accuratezza. Usa per calibrazione o occlusioni frequenti.
#
#  'balanced'  complexity=1  640×480  static=False  ~25-35 fps  <- RACCOMANDATO
#              Buon equilibrio qualita/velocita per task ergonomici standard.
#
#  'fast'      complexity=1  320×240  static=False  ~40-55 fps
#              Alta frequenza. Adatto quando il soggetto e a >1.5 m dalla camera.
#
#  'ultra'     complexity=0  320×240  static=False  ~60-90 fps
#              Massima velocita. Solo per posture lente o benchmark.
#
PERFORMANCE_PROFILE = 'balanced'

_PROFILES = {
    #              complexity  res_w  res_h  static_mode
    'accuracy': (2,            640,   480,   True),
    'balanced': (1,            640,   480,   False),
    'fast':     (1,            320,   240,   False),
    'ultra':    (0,            320,   240,   False),
}
_prof = _PROFILES.get(PERFORMANCE_PROFILE, _PROFILES['balanced'])
MP_COMPLEXITY, PROC_W, PROC_H, MP_STATIC = _prof
print(f'[PROFILE] {PERFORMANCE_PROFILE}  ->  complexity={MP_COMPLEXITY}'
      f'  res={PROC_W}x{PROC_H}  static={MP_STATIC}')

# Risoluzione del video AVI originale (non modificare).
# PROC_W/PROC_H e la risoluzione a cui viene ridimensionato il frame
# prima di passarlo a MediaPipe. I landmark vengono poi riproiettati
# sulla depth map a risoluzione piena (RES_W x RES_H).
RES_W, RES_H = 640, 480

# Smoothing esponenziale applicato alle coordinate 3D dei landmark.
# 0 = nessuno smoothing, 1 = massimo (landmark fissi).
ALPHA_SMOOTHING = 0.35

# Nomi MediaPipe Pose per i 33 landmark, nell'ordine restituito dal modello.
MP_NAMES = [
    'Nose',       'L_Eye_In',  'L_Eye',     'L_Eye_Out',
    'R_Eye_In',   'R_Eye',     'R_Eye_Out', 'L_Ear',
    'R_Ear',      'Mouth_L',   'Mouth_R',   'L_Shoulder',
    'R_Shoulder', 'L_Elbow',   'R_Elbow',   'L_Wrist',
    'R_Wrist',    'L_Pinky',   'R_Pinky',   'L_Index',
    'R_Index',    'L_Thumb',   'R_Thumb',   'L_Hip',
    'R_Hip',      'L_Knee',    'R_Knee',    'L_Ankle',
    'R_Ankle',    'L_Heel',    'R_Heel',    'L_Foot_In',
    'R_Foot_In',
]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FUNZIONI DI SUPPORTO — GEOMETRIA E ANGOLI
# ═══════════════════════════════════════════════════════════════════════════════

def build_body_frame(sh_l, sh_r, gravity):
    """
    Costruisce il sistema di riferimento locale del corpo a partire dalle
    spalle e dal vettore gravita misurato dall'IMU.

    Assi risultanti (tutti unitari e ortonormali):
      up      = gravity         (verticale reale, verso la testa)
      right   = sh_r - sh_l      (da spalla sinistra a destra, asse laterale)
      forward = up x right       (anteroposteriore, verso davanti)

    La ri-ortogonalizzazione (Gram-Schmidt) garantisce che right perp up
    anche quando le spalle non sono perfettamente orizzontali.

    Restituisce: (up, right, forward) come np.ndarray [3] unitari.
    """
    g  = np.array(gravity, dtype=float)
    up = g / (np.linalg.norm(g) + 1e-9)

    lat = np.array(sh_r, dtype=float) - np.array(sh_l, dtype=float)
    lat_norm = np.linalg.norm(lat)
    right = lat / lat_norm if lat_norm > 1e-6 else np.array([1.0, 0.0, 0.0])

    # Gram-Schmidt: rimuove la componente di right parallela a up
    right = right - np.dot(right, up) * up
    r_norm = np.linalg.norm(right)
    right = right / r_norm if r_norm > 1e-6 else np.array([1.0, 0.0, 0.0])

    forward = np.cross(up, right)
    forward = forward / (np.linalg.norm(forward) + 1e-9)
    return up, right, forward


def project_angle(p_base, p_tip, plane_axis1, plane_axis2, zero_dir, positive_dir):
    """
    Proietta il vettore (p_base -> p_tip) sul piano definito da (plane_axis1,
    plane_axis2) e restituisce l'angolo con segno rispetto a zero_dir.

    Parametri:
      p_base, p_tip  : punti 3D (lista o array)
      plane_axis1/2  : assi che definiscono il piano di proiezione
      zero_dir       : direzione che corrisponde a 0 gradi (postura neutra)
      positive_dir   : direzione verso cui l'angolo diventa positivo

    Convenzioni di segno per i segmenti principali:
      Upper arm sagittale  ->  zero_dir = -up,  positive_dir = forward
                               0 deg = braccio lungo il fianco
                              +90 deg = braccio orizzontale in avanti
      Upper arm frontale   ->  zero_dir = -up,  positive_dir = right
                               0 deg = braccio lungo il fianco
                              +90 deg = braccio abdotto di lato
      Tronco/Collo sagitt. ->  zero_dir = up,   positive_dir = forward
                               0 deg = postura eretta
                              +20 deg = inclinato in avanti

    Restituisce: angolo in gradi (float), positivo o negativo.
    """
    v = np.array(p_tip, dtype=float) - np.array(p_base, dtype=float)

    # Proiezione sul piano
    v_proj = (np.dot(v, plane_axis1) * np.array(plane_axis1) +
              np.dot(v, plane_axis2) * np.array(plane_axis2))
    norm = np.linalg.norm(v_proj)
    if norm < 1e-6:
        return 0.0
    v_proj = v_proj / norm

    z_norm = np.array(zero_dir)     / (np.linalg.norm(zero_dir)     + 1e-9)
    p_norm = np.array(positive_dir) / (np.linalg.norm(positive_dir) + 1e-9)

    angle = float(np.degrees(np.arccos(np.clip(np.dot(v_proj, z_norm), -1.0, 1.0))))
    if np.dot(v_proj, p_norm) < 0:
        angle = -angle
    return angle


def calculate_flexion(p1, p2, p3):
    """
    Angolo di flessione al giunto p2, formato dai segmenti (p1->p2) e (p3->p2).

    Convenzione: 180 gradi - angolo_interno, in modo che:
      0 deg   = braccio completamente disteso (segmenti allineati)
      90 deg  = gomito piegato a 90 gradi
      180 deg = massima flessione (segmenti sovrapposti — geometricamente impossibile)

    Questa convenzione rende LA_Ang direttamente interpretabile come
    'quanti gradi e piegato il gomito rispetto alla posizione distesa'.
    E coerente con la notazione RULA che usa 0 come riferimento neutro.
    """
    v1 = np.array(p1, dtype=float) - np.array(p2, dtype=float)
    v2 = np.array(p3, dtype=float) - np.array(p2, dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_val = np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0)
    angle_interno = float(np.degrees(np.arccos(cos_val)))
    return 180.0 - angle_interno


def calculate_wrist_angles(wr, el, idx_mcp, pin_mcp):
    """
    Calcola flessione/estensione e deviazione ulnare/radiale del polso.

    Geometria:
      Avambraccio = wr - el  (asse longitudinale)
      Mano        = midpoint(idx_mcp, pin_mcp) - wr

    Flessione/estensione:
      Angolo tra avambraccio e vettore mano nel piano sagittale della mano.
      Positivo = flessione (palmo verso avambraccio)
      Negativo = estensione

    Deviazione:
      Componente laterale del vettore nocche rispetto all'asse avambraccio.
      Positivo = deviazione radiale (verso pollice)
      Negativo = deviazione ulnare (verso mignolo)

    Restituisce: (flex_ext_deg, deviation_deg) — entrambi float.
    Se i landmark sono degeneri restituisce (0.0, 0.0).
    """
    wr      = np.array(wr,      dtype=float)
    el      = np.array(el,      dtype=float)
    idx_mcp = np.array(idx_mcp, dtype=float)
    pin_mcp = np.array(pin_mcp, dtype=float)

    forearm = wr - el
    fn = np.linalg.norm(forearm)
    if fn < 1e-6:
        return 0.0, 0.0
    forearm_u = forearm / fn

    mcp_mid = (idx_mcp + pin_mcp) / 2.0
    hand_v  = mcp_mid - wr
    hn = np.linalg.norm(hand_v)
    if hn < 1e-6:
        return 0.0, 0.0
    hand_u = hand_v / hn

    # Flessione: angolo con segno tra avambraccio e vettore mano
    cos_flex = np.clip(np.dot(forearm_u, hand_u), -1.0, 1.0)
    flex_ext = float(np.degrees(np.arccos(cos_flex)))
    cross = np.cross(forearm_u, hand_u)
    lat   = idx_mcp - pin_mcp
    lat_n = np.linalg.norm(lat)
    if lat_n > 1e-6 and np.dot(cross, lat / lat_n) < 0:
        flex_ext = -flex_ext   # estensione

    # Deviazione: componente di (idx->pin) perpendicolare all'avambraccio
    lat_u = lat / lat_n if lat_n > 1e-6 else np.array([0.0, 0.0, 1.0])
    lat_perp = lat_u - np.dot(lat_u, forearm_u) * forearm_u
    lp_n = np.linalg.norm(lat_perp)
    if lp_n < 1e-6:
        return float(flex_ext), 0.0
    lat_perp = lat_perp / lp_n
    deviation = float(np.degrees(np.arccos(np.clip(np.dot(lat_u, lat_perp), -1.0, 1.0))))
    if np.dot(np.cross(forearm_u, lat_perp), lat_u) < 0:
        deviation = -deviation

    return float(flex_ext), float(deviation)


def deproject_pixel(intrinsics_params, px, py, depth_m):
    """
    Riproietta un pixel (px, py) con la sua profondita in un punto 3D metrico.

    intrinsics_params = (fx, fy, ppx, ppy)
      fx, fy  : lunghezze focali in pixel
      ppx,ppy : coordinate del punto principale (centro ottico)

    Restituisce [X, Y, Z] in metri nel sistema di riferimento camera.
    """
    fx, fy, ppx, ppy = intrinsics_params
    x = (px - ppx) * depth_m / fx
    y = (py - ppy) * depth_m / fy
    return [x, y, depth_m]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FUNZIONI DI SUPPORTO — I/O
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt(v):
    """
    Formatta un float per la scrittura CSV.
    Se EXCEL_ITALIANO=True sostituisce il punto decimale con la virgola.
    """
    if isinstance(v, float):
        s = f'{v:.6f}'
        return s.replace('.', ',') if EXCEL_ITALIANO else s
    return v


def _load_gravity(base_path):
    """
    Carica il vettore gravita cercandolo in quattro percorsi in ordine di priorita:
      1. <base_path>.gravity.npy          (specifico per sessione)
      2. <cartella_del_file>/gravity.npy  (cartella output_raw/)
      3. <script_dir>/output_raw/gravity.npy  (percorso standard assoluto)
      4. <script_dir>/gravity.npy             (fallback nella dir dello script)

    Se nessun file e trovato, restituisce [0.0, 1.0, 0.0]
    (camera orizzontale, asse Y camera = verticale del mondo).

    Restituisce: lista [x, y, z] normalizzata.
    """
    _base_dir = os.path.dirname(os.path.abspath(base_path))
    candidates = [
        base_path + '.gravity.npy',
        os.path.join(_base_dir, 'gravity.npy'),
        os.path.join(_SCRIPT_DIR, 'output_raw', 'gravity.npy'),
        os.path.join(_SCRIPT_DIR, 'gravity.npy'),
    ]
    # Rimuove duplicati mantenendo l'ordine
    seen = set()
    candidates = [p for p in candidates if not (p in seen or seen.add(p))]

    for gpath in candidates:
        if os.path.exists(gpath):
            g = np.load(gpath)
            # Verifica che sia un array 1D di 3 elementi validi (non scalare, non NaN)
            if g.ndim == 1 and len(g) == 3 and not np.any(np.isnan(g)):
                g_list = (g / np.linalg.norm(g)).tolist()
                print(f'[IMU]  gravity caricato da: {gpath}')
                print(f'[IMU]  vettore: {[round(x, 3) for x in g_list]}')
                return g_list
            else:
                print(f'[IMU]  ATTENZIONE: {gpath} contiene un valore non valido '
                      f'(shape={g.shape}) — ignorato.')

    print('[IMU]  ATTENZIONE: gravity.npy non trovato — uso fallback [0, 1, 0].')
    print('       Percorsi cercati:')
    for p in candidates:
        print(f'         {p}')
    print('       Soluzione: esegui crea_gravity.py, oppure avvia Acquisizione.py')
    print('       con la camera ferma per 2 secondi prima di registrare.')
    print('       Il processing continua: gli angoli sono corretti se la camera')
    print('       e tenuta orizzontale e frontale (setup standard).')
    return [0.0, 1.0, 0.0]


def find_latest_base(folder=None):
    """
    Trova il file .avi piu recente nella cartella specificata e restituisce
    il percorso base senza estensione, pronto per process_file().

    Solleva FileNotFoundError se la cartella non contiene file .avi.
    """
    if folder is None:
        folder = os.path.join(_SCRIPT_DIR, 'output_raw')
    avis = glob.glob(os.path.join(folder, '*.avi'))
    if not avis:
        raise FileNotFoundError(f'Nessun file .avi trovato in {folder}')
    latest = max(avis, key=os.path.getmtime)
    return latest.replace('.avi', '')


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FUNZIONE PRINCIPALE — PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def process_file(base_path):
    """
    Processa una sessione di acquisizione e produce il CSV di output.

    Parametri:
      base_path : percorso base senza estensione
                  es. "output_raw/Base_Line_2026-03-17_10-00-00"
                  Cerca automaticamente .avi, .depth.npy, .times.npy

    Restituisce: percorso del file CSV generato (stringa).

    Flusso:
      1. Carica depth stack e timestamp
      2. Carica il vettore gravita IMU (con fallback automatico)
      3. Apre il video RGB con OpenCV
      4. Per ogni frame: rileva la posa con MediaPipe, proietta i landmark
         in 3D usando la depth map, calcola gli angoli anatomici e il RULA
      5. Scrive una riga CSV per ogni frame rilevato correttamente
    """
    avi_path   = f'{base_path}.avi'
    depth_path = f'{base_path}.depth.npy'
    times_path = f'{base_path}.times.npy'

    for p in [avi_path, depth_path, times_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f'File mancante: {p}')

    # ── Caricamento dati ─────────────────────────────────────────────────────
    print(f'[LOAD] {base_path}')
    depth_stack = np.load(depth_path)   # (N, H, W) float32 in metri
    time_stack  = np.load(times_path)   # (N,)       float64 in secondi
    n_frames    = len(time_stack)
    duration    = time_stack[-1]
    fs_est      = n_frames / duration if duration > 0 else 0
    print(f'       {n_frames} frame  |  durata {duration:.1f}s  |  fs~{fs_est:.1f} Hz')

    # ── Parametri intrinseci camera ──────────────────────────────────────────
    # Valori approssimati per Intel RealSense D435 @ 640x480.
    # Per valori esatti: salva gli intrinseci in Acquisizione.py con np.save.
    fx, fy   = 615.0, 615.0
    ppx, ppy = 320.0, 240.0
    intr     = (fx, fy, ppx, ppy)

    # ── Gravita IMU ──────────────────────────────────────────────────────────
    gravity = _load_gravity(base_path)

    # ── MediaPipe Pose ───────────────────────────────────────────────────────
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=MP_STATIC,
        model_complexity=MP_COMPLEXITY,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Stato dei landmark: usato come fallback quando la depth non e valida
    last_valid_3d = {name: [0.0, 0.0, 0.0] for name in MP_NAMES}
    smoothed_3d   = {name: [0.0, 0.0, 0.0] for name in MP_NAMES}

    # ── Output CSV ───────────────────────────────────────────────────────────
    csv_name = os.path.basename(base_path) + '.csv'
    csv_path = os.path.join(FOLDER_CSV, csv_name)
    delim    = ';' if EXCEL_ITALIANO else ','

    cap = cv2.VideoCapture(avi_path)

    # ── Output Video con marker ──────────────────────────────────────────────
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    if fps_video <= 0:
        fps_video = fs_est if fs_est > 0 else 30.0
    video_name = os.path.basename(base_path) + '_markers.mp4'
    video_path = os.path.join(FOLDER_VIDEO, video_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps_video, (RES_W, RES_H))

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f_csv:
        writer = csv.writer(f_csv, delimiter=delim)

        # Intestazione colonne
        headers = [
            'Timestamp', 'Frame', 'RULA_C', 'Score_A_R', 'Score_A_L', 'Score_B',
            # Angoli sagittali principali (usati da RULA)
            'Trunk_Sag', 'Neck_Sag', 'UA_Sag_R', 'LA_Ang_R', 'UA_Sag_L', 'LA_Ang_L',
            # Angoli frontali (analisi supplementare)
            'Trunk_Fro', 'Neck_Fro', 'UA_Fro_R', 'UA_Fro_L',
            # Angoli polso: flessione/estensione e deviazione ulnare/radiale
            'Wr_Flex_R', 'Wr_Dev_R', 'Wr_Flex_L', 'Wr_Dev_L',
        ]
        for name in MP_NAMES:
            headers.extend([f'{name}_X', f'{name}_Y', f'{name}_Z'])
        writer.writerow(headers)

        frame_idx = 0
        processed = 0
        skipped   = 0

        while True:
            ret, img = cap.read()
            if not ret or frame_idx >= n_frames:
                break

            t_elapsed = time_stack[frame_idx]
            # Converti secondi -> HH:MM:SS.mmm (compatibile con AnalisiDati.py)
            ts_dt  = datetime.datetime(2000, 1, 1) + datetime.timedelta(seconds=t_elapsed)
            ts_str = ts_dt.strftime('%H:%M:%S.%f')[:-3]

            depth_map = depth_stack[frame_idx]   # (H, W) float32 in metri

            # Ridimensiona per MediaPipe solo se il profilo usa risoluzione ridotta.
            # I landmark normalizzati [0,1] vengono scalati a RES_W x RES_H
            # per accedere correttamente alla depth map a piena risoluzione.
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_mp  = cv2.resize(img_rgb, (PROC_W, PROC_H)) if (PROC_W != RES_W) else img_rgb
            results = pose.process(img_mp)

            if results.pose_landmarks:
                full_3d = []

                for i, lm in enumerate(results.pose_landmarks.landmark):
                    name = MP_NAMES[i]
                    px   = int(np.clip(lm.x * RES_W, 0, RES_W - 1))
                    py   = int(np.clip(lm.y * RES_H, 0, RES_H - 1))
                    d    = float(depth_map[py, px])

                    if d > 0.1:   # depth valida (> 10 cm — esclude valori degeneri)
                        point = deproject_pixel(intr, px, py, d)
                        last_valid_3d[name] = point
                    else:
                        point = last_valid_3d[name]

                    # Smoothing esponenziale: riduce il rumore tra fotogrammi
                    smoothed_3d[name] = [
                        (1 - ALPHA_SMOOTHING) * smoothed_3d[name][j] + ALPHA_SMOOTHING * point[j]
                        for j in range(3)
                    ]
                    full_3d.append(smoothed_3d[name])

                # ── Landmark principali ──────────────────────────────────────
                sh_r, el_r, wr_r = full_3d[12], full_3d[14], full_3d[16]
                sh_l, el_l, wr_l = full_3d[11], full_3d[13], full_3d[15]
                hip_r, hip_l     = full_3d[24], full_3d[23]
                ear_r            = full_3d[8]
                # Nocche per il calcolo degli angoli del polso
                # Indici MP: Pinky_MCP = 17(SX)/18(DX), Index_MCP = 19(SX)/20(DX)
                pin_r, pin_l = full_3d[18], full_3d[17]
                idx_r, idx_l = full_3d[20], full_3d[19]

                # ── Sistema di riferimento corporeo ──────────────────────────
                up, right, forward = build_body_frame(sh_l, sh_r, gravity)
                down = -up   # verso il basso = postura neutra arti

                # ── Upper Arm (braccio superiore) ────────────────────────────
                # 0 deg = braccio lungo il fianco
                # +90 deg = braccio orizzontale in avanti (sagittale)
                # +90 deg = abduzione laterale (frontale)
                ua_sag_r = project_angle(sh_r, el_r, up, forward,
                                         zero_dir=down, positive_dir=forward)
                ua_sag_l = project_angle(sh_l, el_l, up, forward,
                                         zero_dir=down, positive_dir=forward)
                ua_fro_r = project_angle(sh_r, el_r, up, right,
                                         zero_dir=down, positive_dir=right)
                ua_fro_l = project_angle(sh_l, el_l, up, -right,
                                         zero_dir=down, positive_dir=-right)
                # Per RULA: usa la flessione assoluta (RULA non distingue avanti/indietro)
                ua_ang_r = abs(ua_sag_r)
                ua_ang_l = abs(ua_sag_l)

                # ── Lower Arm (gomito) ───────────────────────────────────────
                # Angolo 3D tra i segmenti braccio e avambraccio
                # 0 deg = braccio disteso | 90 deg = gomito piegato a 90 gradi
                la_ang_r = calculate_flexion(sh_r, el_r, wr_r)
                la_ang_l = calculate_flexion(sh_l, el_l, wr_l)

                # ── Polso ────────────────────────────────────────────────────
                # flex_ext: 0 deg = neutro, + = flessione, - = estensione
                # deviation: 0 deg = neutro, + = radiale, - = ulnare
                wr_flex_r, wr_dev_r = calculate_wrist_angles(wr_r, el_r, idx_r, pin_r)
                wr_flex_l, wr_dev_l = calculate_wrist_angles(wr_l, el_l, idx_l, pin_l)

                # ── Tronco ───────────────────────────────────────────────────
                hip_mid = [(hip_r[i] + hip_l[i]) / 2 for i in range(3)]
                sh_mid  = [(sh_r[i]  + sh_l[i])  / 2 for i in range(3)]
                # 0 deg = tronco eretto | + = inclinato in avanti
                tr_sag = project_angle(hip_mid, sh_mid, up, forward,
                                       zero_dir=up, positive_dir=forward)
                # 0 deg = tronco eretto | + = inclinato verso destra
                tr_fro = project_angle(hip_mid, sh_mid, up, right,
                                       zero_dir=up, positive_dir=right)
                tr_ang = abs(tr_sag)

                # ── Collo ────────────────────────────────────────────────────
                # 0 deg = testa eretta | + = testa inclinata in avanti
                nk_sag = project_angle(sh_mid, ear_r, up, forward,
                                       zero_dir=up, positive_dir=forward)
                # 0 deg = testa eretta | + = orecchio verso spalla destra
                nk_fro = project_angle(sh_mid, ear_r, up, right,
                                       zero_dir=up, positive_dir=right)
                nk_ang = abs(nk_sag)

                # ── RULA ─────────────────────────────────────────────────────
                # Score B: collo + tronco + gambe (gambe = 1, non visibili)
                nk_s   = rula_logic.get_neck_score(nk_ang, 0, 0)
                tr_s   = rula_logic.get_trunk_score(tr_ang, 0, 0)
                sc_b   = rula_logic.get_rula_score_b(nk_s, tr_s, 1)
                # Score A lato destro: upper arm + lower arm + polso
                ua_s_r = rula_logic.get_upper_arm_score(ua_ang_r)
                la_s_r = rula_logic.get_lower_arm_score(la_ang_r)
                wr_s_r = rula_logic.get_wrist_score(wr_flex_r, wr_dev_r)
                sc_a_r = rula_logic.get_rula_score_a(ua_s_r, la_s_r, wr_s_r, 1)
                # Score A lato sinistro
                ua_s_l = rula_logic.get_upper_arm_score(ua_ang_l)
                la_s_l = rula_logic.get_lower_arm_score(la_ang_l)
                wr_s_l = rula_logic.get_wrist_score(wr_flex_l, wr_dev_l)
                sc_a_l = rula_logic.get_rula_score_a(ua_s_l, la_s_l, wr_s_l, 1)
                # Score C finale: worst-side tra DX e SX
                sc_c   = rula_logic.get_rula_score_c(max(sc_a_r, sc_a_l), sc_b)

                # ── Scrittura riga CSV ────────────────────────────────────────
                floats = [
                    tr_sag,    nk_sag,
                    ua_sag_r,  la_ang_r, ua_sag_l, la_ang_l,
                    tr_fro,    nk_fro,   ua_fro_r, ua_fro_l,
                    wr_flex_r, wr_dev_r, wr_flex_l, wr_dev_l,
                ]
                row = [ts_str, processed + 1, sc_c, sc_a_r, sc_a_l, sc_b]
                row += [_fmt(v) for v in floats]
                for pt in full_3d:
                    row.extend([_fmt(v) for v in pt])
                writer.writerow(row)
                processed += 1

                # ── Frame annotato per il video ──────────────────────────────
                frame_out = img.copy()
                mp.solutions.drawing_utils.draw_landmarks(
                    frame_out, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp.solutions.drawing_utils.DrawingSpec(
                        color=(0, 0, 255), thickness=2),
                )
                rula_color = (0, 200, 0) if sc_c <= 2 else (0, 165, 255) if sc_c <= 4 else (0, 0, 255)
                cv2.putText(frame_out, f'RULA: {sc_c}',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, rula_color, 2)
                cv2.putText(frame_out, f'Frame: {frame_idx + 1}',
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                video_writer.write(frame_out)

            else:
                skipped += 1
                video_writer.write(img)

            frame_idx += 1

            # Progress a terminale ogni 30 frame
            if frame_idx % 30 == 0:
                pct = frame_idx / n_frames * 100
                print(f'  {pct:5.1f}%  frame {frame_idx}/{n_frames}'
                      f'  ok={processed}  skip={skipped}', end='\r')

    cap.release()
    video_writer.release()
    pose.close()
    print(f'\n[DONE] {processed} frame processati, {skipped} saltati (no detection)')
    print(f'       CSV   -> {csv_path}')
    print(f'       Video -> {video_path}')
    return csv_path


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if len(sys.argv) > 1:
        base = sys.argv[1].replace('.avi', '')
    else:
        print('[AUTO] Nessun argomento — cerco il file piu recente in output_raw/')
        base = find_latest_base('output_raw')
        print(f'       Trovato: {base}')

    process_file(base)