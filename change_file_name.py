import glob
import os
from config import SUBJECT, ACTIVITY, RAW_DIR, N_ROUNDS, DUR, CUT

files = sorted(glob.glob(os.path.join(RAW_DIR, f"{SUBJECT}_{ACTIVITY}*.zip")))
for i, path in enumerate(files, start=1):
    new_path = os.path.join(RAW_DIR, f"{SUBJECT}_{ACTIVITY}_{i}.zip")
    print(os.path.basename(path), "->", os.path.basename(new_path))
    os.rename(path, new_path)