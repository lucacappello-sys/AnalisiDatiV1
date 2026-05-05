"""
organizza_smontaggio.py
=======================
Crea la cartella smontaggio/ con due sottocartelle organizzate:
  smontaggio 001/  <- da smontaggio_001/
  smontaggio 002/  <- da smontaggio_002/

I rgb_frames guidano la rinumerazione (partono da 0).
Per ogni rgb frame, si copia il npy con lo stesso numero originale (se esiste).
Le cartelle svo/ e video/ vengono copiate integralmente.
"""

import glob
import os
import shutil

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SOURCES = [
    ('smontaggio_001', 'smontaggio 001'),
    ('smontaggio_002', 'smontaggio 002'),
]
OUT_ROOT = os.path.join(_SCRIPT_DIR, 'smontaggio')


def reorganize_session(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)

    rgb_src = os.path.join(src_dir, 'rgb_frames')
    npy_src = os.path.join(src_dir, 'npy_frames')
    rgb_dst = os.path.join(dst_dir, 'rgb_frames')
    npy_dst = os.path.join(dst_dir, 'npy_frames')
    os.makedirs(rgb_dst, exist_ok=True)
    os.makedirs(npy_dst, exist_ok=True)

    rgb_files = sorted(glob.glob(os.path.join(rgb_src, 'frame_*.png')))
    npy_map = {
        int(os.path.basename(p).split('_')[1].split('.')[0]): p
        for p in glob.glob(os.path.join(npy_src, 'frame_*.npy'))
    }

    n_rgb = 0
    n_npy = 0
    for new_idx, rgb_path in enumerate(rgb_files):
        orig_num = int(os.path.basename(rgb_path).split('_')[1].split('.')[0])
        new_name = f'frame_{new_idx:05d}'

        shutil.copy2(rgb_path, os.path.join(rgb_dst, f'{new_name}.png'))
        n_rgb += 1

        if orig_num in npy_map:
            shutil.copy2(npy_map[orig_num], os.path.join(npy_dst, f'{new_name}.npy'))
            n_npy += 1

    print(f'  rgb_frames: {n_rgb} frame copiati (rinumerati da 0 a {n_rgb - 1})')
    print(f'  npy_frames: {n_npy} frame copiati (su {len(npy_map)} disponibili)')

    for subdir in ['svo', 'video']:
        src_sub = os.path.join(src_dir, subdir)
        dst_sub = os.path.join(dst_dir, subdir)
        if os.path.isdir(src_sub) and os.listdir(src_sub):
            shutil.copytree(src_sub, dst_sub, dirs_exist_ok=True)
            n_files = sum(len(fs) for _, _, fs in os.walk(dst_sub))
            print(f'  {subdir}/: {n_files} file copiati')


def main():
    print(f'Output: {OUT_ROOT}\n')

    for src_name, dst_name in SOURCES:
        src_dir = os.path.join(_SCRIPT_DIR, src_name)
        dst_dir = os.path.join(OUT_ROOT, dst_name)

        if not os.path.isdir(src_dir):
            print(f'[SKIP] {src_name}/ non trovata — salto')
            continue

        if os.path.isdir(dst_dir) and os.listdir(dst_dir):
            print(f'[SKIP] {dst_name}/ esiste già e non è vuota — salto')
            continue

        print(f'{src_name}/ -> smontaggio/{dst_name}/')
        reorganize_session(src_dir, dst_dir)
        print()

    print('[FATTO]')
    print('\nPer processare le sessioni:')
    for _, dst_name in SOURCES:
        print(f'  python process_pump.py "smontaggio/{dst_name}"')


if __name__ == '__main__':
    main()
