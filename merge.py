import zipfile
import os
import numpy as np
import pandas as pd
import glob
from config import SUBJECT, ACTIVITY, RAW_DIR, N_ROUNDS, DUR, CUT

# ================= ตั้งค่า =================

OUT_FOLDER = f"data/{SUBJECT}_{ACTIVITY}"
ROUNDS = [f"{RAW_DIR}/{SUBJECT}_{ACTIVITY}_{i}.zip" for i in range(1, N_ROUNDS + 1)]
# ได้: raw/upstairs_1, raw/upstairs_2, ... raw/upstairs_12
# ถ้าชื่อโฟลเดอร์ดิบของคุณต่างออกไป แก้แค่บรรทัด ROUNDS ตรงนี้บรรทัดเดียว
# ============================================


def load_round(zip_path):
    """อ่านหนึ่งรอบจากไฟล์ .zip โดยตรง ตัด 3 วินาทีต้นและท้าย"""
    with zipfile.ZipFile(zip_path) as z:
        acc = pd.read_csv(z.open("Accelerometer.csv")).iloc[:, :4]
        gyr = pd.read_csv(z.open("Gyroscope.csv")).iloc[:, :4]
    acc.columns = ["t", "ax", "ay", "az"]
    gyr.columns = ["t", "gx", "gy", "gz"]

    t0 = max(acc.t.min(), gyr.t.min()) + CUT
    t1 = min(acc.t.max(), gyr.t.max()) - CUT
    if t1 <= t0:
        raise ValueError(f"{zip_path} สั้นเกินไป")

    grid = np.arange(t0, t1, 0.02)
    sig = pd.DataFrame({"t": grid - t0})
    for c in ["ax", "ay", "az"]:
        sig[c] = np.interp(grid, acc.t, acc[c])
    for c in ["gx", "gy", "gz"]:
        sig[c] = np.interp(grid, gyr.t, gyr[c])
    return sig


def concat_rounds(folders):
    """ต่อหลายรอบเข้าด้วยกัน โดยเลื่อนเวลาของแต่ละรอบให้ต่อเนื่อง"""
    parts = []
    offset = 0.0
    for f in folders:
        sig = load_round(f)
        sig["t"] = sig["t"] + offset
        offset = sig["t"].iloc[-1] + 0.02
        parts.append(sig)
    return pd.concat(parts, ignore_index=True)


def build_60s(folders):
    sig = concat_rounds(folders)
    total = sig["t"].iloc[-1] + 0.02
    if total < DUR:
        raise ValueError(f"รวมได้ {total:.1f} วินาที ยังไม่ถึง {DUR:.0f} วินาที ต้องเพิ่มรอบ")
    sig = sig[sig["t"] < DUR].reset_index(drop=True)   # ตัดให้เหลือ 60 วินาทีพอดี
    return sig


def write_folder(sig, out_folder):
    """เขียนเป็น Accelerometer.csv และ Gyroscope.csv ในรูปแบบที่โค้ดหลักอ่านได้"""
    os.makedirs(out_folder, exist_ok=True)
    acc = sig[["t", "ax", "ay", "az"]].copy()
    gyr = sig[["t", "gx", "gy", "gz"]].copy()
    acc.columns = ["Time (s)", "X (m/s^2)", "Y (m/s^2)", "Z (m/s^2)"]
    gyr.columns = ["Time (s)", "X (rad/s)", "Y (rad/s)", "Z (rad/s)"]
    acc.to_csv(os.path.join(out_folder, "Accelerometer.csv"), index=False)
    gyr.to_csv(os.path.join(out_folder, "Gyroscope.csv"), index=False)


if __name__ == "__main__":
    sig = build_60s(ROUNDS)
    write_folder(sig, OUT_FOLDER)
    print("เขียนแล้ว:", OUT_FOLDER, "| จำนวนตัวอย่าง:", len(sig),
          "| ความยาว:", round(sig["t"].iloc[-1] + 0.02, 2), "วินาที")