import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import separated_scripts.rula_logic as rula_logic
import csv
import datetime
import os

# --- CONFIGURAZIONI ---
RES_W, RES_H = 640, 480
ALPHA_SMOOTHING = 0.35 
FOLDER_CSV = "output_csv"
FOLDER_VIDEO = "output_videos"
MODEL_PATH = "yolo11n-pose.pt" # Verrà scaricato automaticamente al primo avvio

# Creazione cartelle di output se non esistono
os.makedirs(FOLDER_CSV, exist_ok=True)
os.makedirs(FOLDER_VIDEO, exist_ok=True)

# Mappatura YOLO COCO-17
YOLO_NAMES = [
    "Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hip", "R_Hip",
    "L_Knee", "R_Knee", "L_Ankle", "R_Ankle"
]

# --- STATO GLOBALE REGISTRAZIONE ---
app_state = {
    "recording": False, 
    "frame_cnt": 0,
    "video_writer": None,
    "csv_file": None,
    "csv_writer": None,
    "session_name": "Test_YOLO_V11"
}

def start_recording(filename):
    """Inizializza i file di output per la sessione corrente."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    full_filename = f"{filename}_{timestamp}"
    
    # Setup CSV
    csv_path = os.path.join(FOLDER_CSV, f"{full_filename}.csv")
    app_state["csv_file"] = open(csv_path, mode='w', newline='')
    app_state["csv_writer"] = csv.writer(app_state["csv_file"])
    
    headers = ["Timestamp", "Frame", "RULA_C", "Score_A", "Score_B", "Trunk_Ang", "Neck_Ang", "UA_Ang", "LA_Ang"]
    for name in YOLO_NAMES: headers.extend([f"{name}_X", f"{name}_Y", f"{name}_Z"])
    app_state["csv_writer"].writerow(headers)
    
    # Setup Video
    video_path = os.path.join(FOLDER_VIDEO, f"{full_filename}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    app_state["video_writer"] = cv2.VideoWriter(video_path, fourcc, 20.0, (RES_W, RES_H))
    
    app_state["recording"] = True
    print(f"\a >>> REGISTRAZIONE AVVIATA: {full_filename}")

def stop_recording():
    """Chiude i file e ferma la scrittura dei dati."""
    if app_state["csv_file"]: app_state["csv_file"].close()
    if app_state["video_writer"]: app_state["video_writer"].release()
    
    app_state["recording"] = False
    app_state["csv_file"] = None
    app_state["video_writer"] = None
    print("\a <<< REGISTRAZIONE FERMATA.")

def mouse_callback(event, x, y, flags, param):
    """Callback per gestire l'avvio con click del mouse."""
    if event == cv2.EVENT_LBUTTONDOWN:
        if not app_state["recording"]:
            start_recording(app_state["session_name"])
        else:
            stop_recording()

# --- FUNZIONI GEOMETRICHE ---
def calculate_inclination(p1, p2):
    v = np.array(p2) - np.array(p1)
    vertical = np.array([0, -1, 0])
    norm_v = np.linalg.norm(v)
    if norm_v == 0: return 0
    return np.degrees(np.arccos(np.clip(np.dot(v, vertical) / norm_v, -1.0, 1.0)))

def calculate_angle(p1, p2, p3):
    v1, v2 = np.array(p1) - np.array(p2), np.array(p3) - np.array(p2)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0: return 0
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))

# --- LOGICA PRINCIPALE ---
def main():
    # Caricamento modello e camera
    model = YOLO(MODEL_PATH)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, RES_W, RES_H, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, RES_W, RES_H, rs.format.bgr8, 30)
    
    # Interfaccia OpenCV
    cv2.namedWindow('YOLOv11 RULA TRACKER')
    cv2.setMouseCallback('YOLOv11 RULA TRACKER', mouse_callback)

    smoothed_3d = {name: [0.0, 0.0, 0.0] for name in YOLO_NAMES}
    last_valid_3d = {name: [0.0, 0.0, 0.0] for name in YOLO_NAMES}

    try:
        profile = pipeline.start(config)
        intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        align = rs.align(rs.stream.color)
        
        print("\n--- SISTEMA PRONTO ---")
        print("- Clicca col MOUSE o premi 'R' per registrare.")
        print("- Premi 'ESC' per uscire.\n")

        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            d_fr = aligned_frames.get_depth_frame()
            c_fr = aligned_frames.get_color_frame()
            if not d_fr or not c_fr: continue

            img = np.asanyarray(c_fr.get_data())
            app_state["frame_cnt"] += 1
            
            # Inferenza YOLOv11
            results = model(img, verbose=False, conf=0.5)[0]
            
            # Protezione IndexError: verifica se è stata rilevata una persona
            if results.keypoints is not None and len(results.keypoints.data) > 0 and results.keypoints.has_visible:
                kpts = results.keypoints.xy[0].cpu().numpy()
                full_3d = []
                
                for i, kp in enumerate(kpts):
                    name = YOLO_NAMES[i]
                    px, py = int(kp[0]), int(kp[1])
                    
                    # Estrazione Z RealSense
                    d = d_fr.get_distance(px, py) if (0 <= px < RES_W and 0 <= py < RES_H) else 0
                    point = rs.rs2_deproject_pixel_to_point(intrinsics, [px, py], d) if d > 0 else last_valid_3d[name]
                    
                    last_valid_3d[name] = point
                    # Smoothing esponenziale
                    smoothed_3d[name] = [(1 - ALPHA_SMOOTHING) * smoothed_3d[name][j] + ALPHA_SMOOTHING * point[j] for j in range(3)]
                    full_3d.append(smoothed_3d[name])

                # Se in registrazione, calcola angoli e salva
                if app_state["recording"] and len(full_3d) > 12:
                    # Indici YOLO COCO-17 (R_Ear:4, R_Shoulder:6, R_Elbow:8, R_Wrist:10, R_Hip:12)
                    sh, el, wr, hip, ear = full_3d[6], full_3d[8], full_3d[10], full_3d[12], full_3d[4]
                    
                    tr_ang = calculate_inclination(hip, sh)
                    nk_ang = calculate_inclination(sh, ear)
                    ua_ang = calculate_inclination(sh, el)
                    la_ang = calculate_angle(sh, el, wr)
                    
                    sc_a = rula_logic.get_rula_score_a(rula_logic.get_upper_arm_score(ua_ang), rula_logic.get_lower_arm_score(la_ang), 1, 1)
                    sc_b = rula_logic.get_rula_score_b(rula_logic.get_neck_score(nk_ang, 0, 0), rula_logic.get_trunk_score(tr_ang, 0, 0), 1)
                    sc_c = rula_logic.get_rula_score_c(sc_a, sc_b)

                    if app_state["csv_writer"]:
                        row = [datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3], app_state["frame_cnt"], sc_c, sc_a, sc_b, tr_ang, nk_ang, ua_ang, la_ang]
                        for pt in full_3d: row.extend(pt)
                        app_state["csv_writer"].writerow(row)

                img = results.plot() # Sovrapposizione scheletro YOLO
            else:
                cv2.putText(img, "SOGGETTO NON RILEVATO", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Feedback Registrazione
            if app_state["recording"]:
                cv2.circle(img, (30, 30), 12, (0, 0, 255), -1)
                cv2.putText(img, "REC ON", (55, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if app_state["video_writer"]: app_state["video_writer"].write(img)
            else:
                cv2.putText(img, "READY - STANDBY", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('YOLOv11 RULA TRACKER', img)
            
            key = cv2.waitKey(1)
            if key == 27: # ESC
                break
            elif key == ord('r') or key == ord('R'):
                if not app_state["recording"]:
                    start_recording(app_state["session_name"])
                else:
                    stop_recording()

    finally:
        stop_recording()
        pipeline.stop()
        cv2.destroyAllWindows()

# --- ESECUZIONE DIRETTA ---
if __name__ == "__main__":

    main()