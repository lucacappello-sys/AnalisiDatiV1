"""
process_pump.py
===============
Analizza i frame della procedura di montaggio pompa con MediaPipe Pose.

Input  (PumpProcedure/montaggio/):
  rgb_frames/frame_XXXXX.png    frame RGB ZED (720x1280)
  npy_frames/frame_XXXXX.npy   point cloud ZED (720x1280x4, float32, XYZ in mm)

Output:
  output_csv/montaggio.csv      CSV compatibile con AnalisiDati.py
  output_video/montaggio_markers.mp4  video annotato

Uso:
  python process_pump.py
  python process_pump.py montaggio          # sottocartella custom
"""

import csv
import datetime
import glob
import os
import sys

import cv2
import mediapipe as mp
import numpy as np

# ── rula_logic dal progetto principale ───────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, '..', 'separated_scripts'))
import rula_logic

# =============================================================================
# CONFIGURAZIONE
# =============================================================================

SUBFOLDER      = sys.argv[1] if len(sys.argv) > 1 else 'montaggio'
INPUT_DIR      = os.path.join(_SCRIPT_DIR, SUBFOLDER)
RGB_DIR        = os.path.join(INPUT_DIR, 'rgb_frames')
NPY_DIR        = os.path.join(INPUT_DIR, 'npy_frames')
OUTPUT_CSV     = os.path.join(_SCRIPT_DIR, 'output_csv', f'{SUBFOLDER}.csv')
OUTPUT_VIDEO   = os.path.join(_SCRIPT_DIR, 'output_video', f'{SUBFOLDER}_markers.mp4')
os.makedirs(os.path.dirname(OUTPUT_CSV),   exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

FPS             = 30.0
ALPHA           = 0.35      # smoothing esponenziale landmark 3D
EXCEL_ITALIANO  = True      # True -> sep=; dec=,  |  False -> sep=, dec=.

# Vettore gravità nel sistema ZED (X=destra, Y=basso, Z=avanti)
# Con camera orizzontale e frontale: gravità punta lungo +Y camera.
GRAVITY = [0.0, 1.0, 0.0]

# Profilo MediaPipe
MP_COMPLEXITY = 1
PROC_W, PROC_H = 640, 480
MP_STATIC = False

# Dimensioni frame originali (per lookup point cloud)
RES_W, RES_H = 1280, 720

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

# =============================================================================
# GEOMETRIA (identiche a processing.py)
# =============================================================================

def build_body_frame(sh_l, sh_r, gravity):
    g  = np.array(gravity, dtype=float)
    up = g / (np.linalg.norm(g) + 1e-9)
    lat = np.array(sh_r, dtype=float) - np.array(sh_l, dtype=float)
    lat_norm = np.linalg.norm(lat)
    right = lat / lat_norm if lat_norm > 1e-6 else np.array([1.0, 0.0, 0.0])
    right = right - np.dot(right, up) * up
    r_norm = np.linalg.norm(right)
    right = right / r_norm if r_norm > 1e-6 else np.array([1.0, 0.0, 0.0])
    forward = np.cross(up, right)
    forward = forward / (np.linalg.norm(forward) + 1e-9)
    return up, right, forward


def project_angle(p_base, p_tip, plane_axis1, plane_axis2, zero_dir, positive_dir):
    v = np.array(p_tip, dtype=float) - np.array(p_base, dtype=float)
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
    v1 = np.array(p1, dtype=float) - np.array(p2, dtype=float)
    v2 = np.array(p3, dtype=float) - np.array(p2, dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_val = np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0)
    return 180.0 - float(np.degrees(np.arccos(cos_val)))


def calculate_wrist_angles(wr, el, idx_mcp, pin_mcp):
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
    cos_flex = np.clip(np.dot(forearm_u, hand_u), -1.0, 1.0)
    flex_ext = float(np.degrees(np.arccos(cos_flex)))
    cross = np.cross(forearm_u, hand_u)
    lat   = idx_mcp - pin_mcp
    lat_n = np.linalg.norm(lat)
    if lat_n > 1e-6 and np.dot(cross, lat / lat_n) < 0:
        flex_ext = -flex_ext
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


def _fmt(v):
    if isinstance(v, float):
        s = f'{v:.6f}'
        return s.replace('.', ',') if EXCEL_ITALIANO else s
    return v

# =============================================================================
# MAIN
# =============================================================================

def process():
    # ── Raccolta frame ────────────────────────────────────────────────────────
    rgb_files = sorted(glob.glob(os.path.join(RGB_DIR, 'frame_*.png')))
    if not rgb_files:
        raise FileNotFoundError(f'Nessun PNG in {RGB_DIR}')

    # Indice dei point cloud disponibili: num_frame -> percorso
    npy_map = {
        int(os.path.basename(p).split('_')[1].split('.')[0]):
        p for p in glob.glob(os.path.join(NPY_DIR, 'frame_*.npy'))
    }

    n_total   = len(rgb_files)
    n_with_pc = sum(1 for p in rgb_files
                    if int(os.path.basename(p).split('_')[1].split('.')[0]) in npy_map)
    print(f'[INFO] {n_total} frame RGB  |  {n_with_pc} con point cloud  |  FPS={FPS}')

    # ── MediaPipe ─────────────────────────────────────────────────────────────
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=MP_STATIC,
        model_complexity=MP_COMPLEXITY,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    last_valid_3d = {name: [0.0, 0.0, 0.0] for name in MP_NAMES}
    smoothed_3d   = {name: [0.0, 0.0, 0.0] for name in MP_NAMES}

    # ── Output video ──────────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (RES_W, RES_H))

    delim = ';' if EXCEL_ITALIANO else ','

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as f_csv:
        writer = csv.writer(f_csv, delimiter=delim)

        headers = [
            'Timestamp', 'Frame', 'RULA_C', 'Score_A_R', 'Score_A_L', 'Score_B',
            'Trunk_Sag', 'Neck_Sag', 'UA_Sag_R', 'LA_Ang_R', 'UA_Sag_L', 'LA_Ang_L',
            'Trunk_Fro', 'Neck_Fro', 'UA_Fro_R', 'UA_Fro_L',
            'Wr_Flex_R', 'Wr_Dev_R', 'Wr_Flex_L', 'Wr_Dev_L',
        ]
        for name in MP_NAMES:
            headers.extend([f'{name}_X', f'{name}_Y', f'{name}_Z'])
        writer.writerow(headers)

        processed = 0
        skipped   = 0

        for rgb_path in rgb_files:
            frame_num = int(os.path.basename(rgb_path).split('_')[1].split('.')[0])
            t_sec     = frame_num / FPS
            ts_dt     = datetime.datetime(2000, 1, 1) + datetime.timedelta(seconds=t_sec)
            ts_str    = ts_dt.strftime('%H:%M:%S.%f')[:-3]

            img_bgr = cv2.imread(rgb_path)
            if img_bgr is None:
                skipped += 1
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_mp  = cv2.resize(img_rgb, (PROC_W, PROC_H))
            results = pose.process(img_mp)

            # Carica point cloud se disponibile
            pc = None
            if frame_num in npy_map:
                pc = np.load(npy_map[frame_num])  # (H, W, 4) float32, XYZ in mm

            if results.pose_landmarks:
                full_3d = []

                # world landmarks come fallback quando manca il point cloud
                wl = results.pose_world_landmarks

                for i, lm in enumerate(results.pose_landmarks.landmark):
                    name = MP_NAMES[i]

                    if pc is not None:
                        px = int(np.clip(lm.x * RES_W, 0, RES_W - 1))
                        py = int(np.clip(lm.y * RES_H, 0, RES_H - 1))
                        xyz_mm = pc[py, px, 0:3]

                        if not np.any(np.isnan(xyz_mm)):
                            point = (xyz_mm / 1000.0).tolist()  # mm → m
                            last_valid_3d[name] = point
                        else:
                            point = last_valid_3d[name]
                    else:
                        # Nessun point cloud: usa world landmarks MediaPipe (metrici)
                        wlm   = wl.landmark[i]
                        point = [wlm.x, wlm.y, wlm.z]
                        last_valid_3d[name] = point

                    smoothed_3d[name] = [
                        (1 - ALPHA) * smoothed_3d[name][j] + ALPHA * point[j]
                        for j in range(3)
                    ]
                    full_3d.append(smoothed_3d[name])

                # ── Landmark principali ───────────────────────────────────────
                sh_r, el_r, wr_r = full_3d[12], full_3d[14], full_3d[16]
                sh_l, el_l, wr_l = full_3d[11], full_3d[13], full_3d[15]
                hip_r, hip_l     = full_3d[24], full_3d[23]
                ear_r            = full_3d[8]
                pin_r, pin_l     = full_3d[18], full_3d[17]
                idx_r, idx_l     = full_3d[20], full_3d[19]

                # ── Sistema di riferimento corporeo ───────────────────────────
                up, right, forward = build_body_frame(sh_l, sh_r, GRAVITY)
                down = -up

                # ── Angoli ────────────────────────────────────────────────────
                ua_sag_r = project_angle(sh_r, el_r, up, forward, down, forward)
                ua_sag_l = project_angle(sh_l, el_l, up, forward, down, forward)
                ua_fro_r = project_angle(sh_r, el_r, up, right,   down, right)
                ua_fro_l = project_angle(sh_l, el_l, up, -right,  down, -right)
                ua_ang_r = abs(ua_sag_r)
                ua_ang_l = abs(ua_sag_l)

                la_ang_r = calculate_flexion(sh_r, el_r, wr_r)
                la_ang_l = calculate_flexion(sh_l, el_l, wr_l)

                wr_flex_r, wr_dev_r = calculate_wrist_angles(wr_r, el_r, idx_r, pin_r)
                wr_flex_l, wr_dev_l = calculate_wrist_angles(wr_l, el_l, idx_l, pin_l)

                hip_mid = [(hip_r[i] + hip_l[i]) / 2 for i in range(3)]
                sh_mid  = [(sh_r[i]  + sh_l[i])  / 2 for i in range(3)]
                tr_sag  = project_angle(hip_mid, sh_mid, up, forward, up, forward)
                tr_fro  = project_angle(hip_mid, sh_mid, up, right,   up, right)
                tr_ang  = abs(tr_sag)

                nk_sag  = project_angle(sh_mid, ear_r, up, forward, up, forward)
                nk_fro  = project_angle(sh_mid, ear_r, up, right,   up, right)
                nk_ang  = abs(nk_sag)

                # ── RULA ──────────────────────────────────────────────────────
                nk_s   = rula_logic.get_neck_score(nk_ang, 0, 0)
                tr_s   = rula_logic.get_trunk_score(tr_ang, 0, 0)
                sc_b   = rula_logic.get_rula_score_b(nk_s, tr_s, 1)
                ua_s_r = rula_logic.get_upper_arm_score(ua_ang_r)
                la_s_r = rula_logic.get_lower_arm_score(la_ang_r)
                wr_s_r = rula_logic.get_wrist_score(wr_flex_r, wr_dev_r)
                sc_a_r = rula_logic.get_rula_score_a(ua_s_r, la_s_r, wr_s_r, 1)
                ua_s_l = rula_logic.get_upper_arm_score(ua_ang_l)
                la_s_l = rula_logic.get_lower_arm_score(la_ang_l)
                wr_s_l = rula_logic.get_wrist_score(wr_flex_l, wr_dev_l)
                sc_a_l = rula_logic.get_rula_score_a(ua_s_l, la_s_l, wr_s_l, 1)
                sc_c   = rula_logic.get_rula_score_c(max(sc_a_r, sc_a_l), sc_b)

                # ── Scrittura CSV ─────────────────────────────────────────────
                floats = [
                    tr_sag, nk_sag,
                    ua_sag_r, la_ang_r, ua_sag_l, la_ang_l,
                    tr_fro,   nk_fro,   ua_fro_r, ua_fro_l,
                    wr_flex_r, wr_dev_r, wr_flex_l, wr_dev_l,
                ]
                row = [ts_str, processed + 1, sc_c, sc_a_r, sc_a_l, sc_b]
                row += [_fmt(v) for v in floats]
                for pt in full_3d:
                    row.extend([_fmt(v) for v in pt])
                writer.writerow(row)
                processed += 1

                # ── Frame annotato ────────────────────────────────────────────
                frame_out = img_bgr.copy()
                mp.solutions.drawing_utils.draw_landmarks(
                    frame_out, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp.solutions.drawing_utils.DrawingSpec(
                        color=(0, 0, 255), thickness=2),
                )
                rula_color = (0, 200, 0) if sc_c <= 2 else (0, 165, 255) if sc_c <= 4 else (0, 0, 255)
                src_label  = 'PC' if pc is not None else 'WL'
                cv2.putText(frame_out, f'RULA: {sc_c}  [{src_label}]',
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, rula_color, 2)
                cv2.putText(frame_out, f'Frame: {frame_num}',
                            (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                video_writer.write(frame_out)

            else:
                skipped += 1
                video_writer.write(img_bgr)

            # Progress ogni 30 frame
            idx = rgb_files.index(rgb_path) + 1
            if idx % 30 == 0 or idx == n_total:
                pct = idx / n_total * 100
                print(f'  {pct:5.1f}%  frame {idx}/{n_total}'
                      f'  ok={processed}  skip={skipped}', end='\r')

    video_writer.release()
    pose.close()
    print(f'\n[DONE] {processed} frame processati, {skipped} saltati')
    print(f'       CSV   -> {OUTPUT_CSV}')
    print(f'       Video -> {OUTPUT_VIDEO}')
    return OUTPUT_CSV


if __name__ == '__main__':
    process()
