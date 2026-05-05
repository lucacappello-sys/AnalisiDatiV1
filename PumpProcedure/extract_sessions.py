"""
extract_sessions.py
===================
Estrae i file zip delle sessioni di montaggio in sottocartelle separate,
rinominando la cartella interna 'montaggio/' -> 'montaggio_001/', 'montaggio_002/', ecc.

Uso:
  python extract_sessions.py                        # cerca gli zip in Downloads
  python extract_sessions.py C:/percorso/custom     # cartella custom con gli zip
"""

import os
import sys
import zipfile

# =============================================================================
# CONFIGURAZIONE
# =============================================================================

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DOWNLOADS    = os.path.join(os.path.expanduser('~'), 'Downloads')
SEARCH_DIR   = sys.argv[1] if len(sys.argv) > 1 else DOWNLOADS

# Prefissi degli zip da cercare (case-insensitive)
ZIP_PREFIXES = ['montaggio', 'smontaggio']

# =============================================================================

def find_zips(folder, prefix):
    """Trova tutti gli zip che iniziano con 'prefix' nel nome, ordinati."""
    zips = [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if f.lower().endswith('.zip') and f.lower().startswith(prefix.lower())
    ]
    return zips


def extract_zip_to_subfolder(zip_path, dest_root, subfolder_name):
    """
    Estrae uno zip in dest_root/subfolder_name/, rimappando il prefisso
    interno 'montaggio/' -> subfolder_name + '/'.
    """
    dest = os.path.join(dest_root, subfolder_name)
    os.makedirs(dest, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        members = zf.namelist()
        print(f'  {len(members)} file da estrarre...')

        for member in members:
            # Rimuovi il prefisso 'montaggio/' e ri-radica in subfolder_name
            # es. 'montaggio/npy_frames/frame_00000.npy' -> 'montaggio_002/npy_frames/frame_00000.npy'
            parts = member.split('/', 1)
            if len(parts) == 2:
                rel_path = parts[1]
            else:
                rel_path = member

            if not rel_path:
                continue  # entry di cartella, salta

            out_path = os.path.join(dest, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            with zf.open(member) as src, open(out_path, 'wb') as dst:
                dst.write(src.read())

    print(f'  -> estratto in: {dest}')
    return dest


def main():
    all_sessions = []

    for prefix in ZIP_PREFIXES:
        zips = find_zips(SEARCH_DIR, prefix)
        for i, zip_path in enumerate(zips, start=1):
            all_sessions.append((prefix, i, zip_path))

    if not all_sessions:
        print(f'[ERRORE] Nessun zip trovato in: {SEARCH_DIR}')
        return

    print(f'Trovati {len(all_sessions)} zip in {SEARCH_DIR}:')
    for prefix, i, zp in all_sessions:
        print(f'  {os.path.basename(zp)}')
    print()

    extracted = []
    for prefix, i, zip_path in all_sessions:
        subfolder = f'{prefix}_{i:03d}'

        dest = os.path.join(_SCRIPT_DIR, subfolder)
        if os.path.isdir(dest) and os.listdir(dest):
            print(f'[SKIP] {subfolder}/ esiste già e non è vuota — salto.')
            extracted.append(subfolder)
            continue

        print(f'{os.path.basename(zip_path)} -> {subfolder}/')
        extract_zip_to_subfolder(zip_path, _SCRIPT_DIR, subfolder)
        extracted.append(subfolder)

    print('\n[FATTO] Per processare ogni sessione:')
    for sf in extracted:
        print(f'  python process_pump.py {sf}')


if __name__ == '__main__':
    main()
