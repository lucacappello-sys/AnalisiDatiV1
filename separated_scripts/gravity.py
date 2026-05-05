import pyrealsense2 as rs
import numpy as np
import os
import time

def get_realsense_gravity():
    # Verifica che almeno un dispositivo RealSense sia connesso
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("Errore: nessuna camera RealSense rilevata. Collega la camera e riprova.")
        return
    dev = devices[0]
    print(f"Camera rilevata: {dev.get_info(rs.camera_info.name)} "
          f"(S/N: {dev.get_info(rs.camera_info.serial_number)})")

    pipeline = None
    started = False

    # Prova configurazioni dalla più specifica alla più generica
    configs_to_try = [
        ("250 Hz", lambda c: c.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)),
        ("63 Hz",  lambda c: c.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 63)),
        ("default", lambda c: c.enable_stream(rs.stream.accel)),
    ]

    for label, setup_fn in configs_to_try:
        pipeline = rs.pipeline()
        config = rs.config()
        try:
            setup_fn(config)
            pipeline.start(config)
            started = True
            print(f"Stream accel avviato ({label})")
            break
        except Exception as e:
            print(f"  {label} fallito: {e}")
            try:
                pipeline.stop()
            except Exception:
                pass
            pipeline = None

    if not started or pipeline is None:
        print("Errore: impossibile avviare il pipeline. Verifica che nessun altro processo usi la camera.")
        return

    try:
        print("Calibrazione in corso... Tieni la camera FERMA.")
        time.sleep(2)  # lascia stabilizzare il sensore

        accel_samples = []
        attempts = 0
        while len(accel_samples) < 50 and attempts < 200:
            attempts += 1
            frames = pipeline.wait_for_frames(timeout_ms=10000)
            accel_frame = frames.first_or_default(rs.stream.accel)
            if accel_frame:
                data = accel_frame.as_motion_frame().get_motion_data()
                accel_samples.append([data.x, data.y, data.z])

        if not accel_samples:
            print("Errore: nessun dato ricevuto dall'accelerometro.")
            return

        g_mean = np.mean(accel_samples, axis=0)
        norm = np.linalg.norm(g_mean)
        g_normalized = g_mean / norm if norm > 0 else np.array([0.0, -1.0, 0.0])

        os.makedirs('output_raw', exist_ok=True)
        np.save('output_raw/gravity.npy', g_normalized.astype(np.float64))

        print(f"\nGravita' salvata con successo!")
        print(f"Vettore rilevato: {np.round(g_normalized, 3)}")
        print("Ora puoi avviare l'acquisizione.")

    except Exception as e:
        print(f"Errore durante la lettura: {e}")
    finally:
        if pipeline is not None:
            pipeline.stop()

if __name__ == "__main__":
    get_realsense_gravity()
