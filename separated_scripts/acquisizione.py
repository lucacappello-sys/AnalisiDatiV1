"""
Acquisizione.py  —  Script 1 di 2
──────────────────────────────────
Acquisisce solo video RGB + depth stream sincronizzati a 30fps.
NON fa tracking: zero collo di bottiglia, frame rate stabile.

Output in output_raw/:
  <session>_<ts>.avi          video RGB (XVID, 30fps)
  <session>_<ts>.depth.npy   depth stack (N, H, W) float32 in metri
  <session>_<ts>.times.npy   timestamp relativo di ogni frame (secondi)

Uso:
  python Acquisizione.py
  python Acquisizione.py Base_Line_Prova_2

Controlli:
  Click sinistro  avvia / ferma registrazione
  ESC             esci
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import datetime, os, sys

RES_W, RES_H = 640, 480
FPS_CAMERA   = 30

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDER_RAW  = os.path.join(_SCRIPT_DIR, 'output_raw')
os.makedirs(FOLDER_RAW, exist_ok=True)

session_name = sys.argv[1] if len(sys.argv) > 1 else 'sessione'


def run_acquisition(session_name):
    recording   = False
    rgb_writer  = None
    depth_stack = []
    time_stack  = []
    t_start     = None
    frame_cnt   = 0
    basename    = ''

    def _open_files():
        ts     = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        base   = os.path.join(FOLDER_RAW, f'{session_name}')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(f'{base}.avi', fourcc, FPS_CAMERA, (RES_W, RES_H))
        print(f'[REC START] {base}')
        return writer, base

    def _close_files(writer, base, d_stack, t_stack):
        writer.release()
        np.save(f'{base}.depth.npy', np.array(d_stack, dtype=np.float32))
        np.save(f'{base}.times.npy', np.array(t_stack,  dtype=np.float64))
        print(f'[REC STOP]  {len(t_stack)} frame → {base}.*')

    def on_mouse(event, x, y, flags, param):
        nonlocal recording, rgb_writer, depth_stack, time_stack
        nonlocal t_start, frame_cnt, basename
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if not recording:
            rgb_writer, basename = _open_files()
            depth_stack, time_stack = [], []
            t_start, frame_cnt = None, 0
            recording = True
        else:
            _close_files(rgb_writer, basename, depth_stack, time_stack)
            rgb_writer = None
            depth_stack, time_stack = [], []
            recording = False

    # Setup RealSense
    spatial      = rs.spatial_filter()
    temporal     = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()

    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.depth, RES_W, RES_H, rs.format.z16,  FPS_CAMERA)
    config.enable_stream(rs.stream.color, RES_W, RES_H, rs.format.bgr8, FPS_CAMERA)
    # Gli stream IMU devono essere abilitati PRIMA di pipeline.start()
    # altrimenti RealSense li ignora e accel_samples rimane vuoto.
    imu_disponibile = False
    try:
        config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 100)
        config.enable_stream(rs.stream.gyro,  rs.format.motion_xyz32f, 200)
        imu_disponibile = True
    except Exception as e:
        print(f'[IMU]   stream non abilitati ({e}) — gravity.npy userà fallback')

    try:
        profile = pipeline.start(config)
        align   = rs.align(rs.stream.color)

        # ── Calibrazione IMU — legge la gravità a camera ferma ───────────────
        # Tutti gli stream (depth, color, accel, gyro) sono già attivi.
        # Attendi ~2s con la camera ferma prima di cliccare per registrare.
        if imu_disponibile:
            print('[IMU]   Calibrazione gravità in corso — tenere la camera ferma...')
            accel_samples = []
            for _ in range(60):   # ~2s a 30fps
                frames = pipeline.wait_for_frames()
                accel  = frames.first_or_default(rs.stream.accel)
                if accel:
                    d = accel.as_motion_frame().get_motion_data()
                    accel_samples.append([d.x, d.y, d.z])
            if accel_samples:
                g_cam = np.mean(accel_samples, axis=0)
                norm  = np.linalg.norm(g_cam)
                if norm > 1e-6:
                    g_cam = g_cam / norm
                    np.save(os.path.join(FOLDER_RAW, 'gravity.npy'), g_cam)
                    print(f'[IMU]   gravity salvato: {[round(x,3) for x in g_cam.tolist()]}')
                else:
                    print('[IMU]   ATTENZIONE: norma gravità nulla — IMU non affidabile, nessun file salvato.')
            else:
                print('[IMU]   ATTENZIONE: nessun campione IMU ricevuto — gravity.npy non salvato.')
        else:
            print('[IMU]   Salto calibrazione IMU — stream non disponibili.')
            print('[IMU]   Processing.py userà fallback [0,1,0] (camera orizzontale).')
        # ─────────────────────────────────────────────────────────────────────

        cv2.namedWindow('ACQUISIZIONE')
        cv2.setMouseCallback('ACQUISIZIONE', on_mouse)
        print('[STANDBY] Clicca per avviare. ESC per uscire.')

        while True:
            frames         = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            d_fr = (hole_filling
                    .process(temporal.process(spatial.process(
                        aligned_frames.get_depth_frame())))
                    .as_depth_frame())
            c_fr = aligned_frames.get_color_frame()
            if not d_fr or not c_fr:
                continue

            img = np.asanyarray(c_fr.get_data())

            if recording:
                now = datetime.datetime.now().timestamp()
                if t_start is None:
                    t_start = now
                elapsed = now - t_start

                rgb_writer.write(img)

                # Depth in metri (float32)
                depth_m = (np.asanyarray(d_fr.get_data()).astype(np.float32)
                           * d_fr.get_units())
                depth_stack.append(depth_m)
                time_stack.append(elapsed)
                frame_cnt += 1

                cv2.circle(img, (30, 30), 10, (0, 0, 255), -1)
                cv2.putText(img,
                    f'REC  {frame_cnt} frame  {elapsed:.1f}s  (click per fermare)',
                    (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.circle(img, (30, 30), 10, (180, 180, 180), -1)
                cv2.putText(img, 'STANDBY — clicca per iniziare',
                    (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)

            cv2.imshow('ACQUISIZIONE', img)
            if cv2.waitKey(1) == 27:
                break

    finally:
        if recording and rgb_writer is not None:
            _close_files(rgb_writer, basename, depth_stack, time_stack)
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    run_acquisition(session_name)