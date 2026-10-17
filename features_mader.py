import glob
import os
import numpy as np
import pandas as pd

CHANS = ["ax", "ay", "az", "gx", "gy", "gz"]
WIN, STEP = 128, 64
FS_DT = 0.02
CUT = 3.0
TRAIN_RATIO = 1.00
FEAT_COLS = [f"{c}_{s}" for c in CHANS for s in ["mean", "std", "min", "max"]]


DUR = 60.0   # ความยาวสัญญาณที่ใช้ (วินาที) หลังตัดขอบ

def load_signal(acc_path, gyr_path):
    acc = pd.read_csv(acc_path).iloc[:, :4]
    acc.columns = ["t", "ax", "ay", "az"]
    gyr = pd.read_csv(gyr_path).iloc[:, :4]
    gyr.columns = ["t", "gx", "gy", "gz"]

    # จุดเริ่มเป็น 3 วินาทีหลังเริ่มบันทึก
    t0 = max(acc.t.min(), gyr.t.min()) + CUT
    t1 = t0 + DUR - FS_DT   # ช่วงสุดท้ายที่ใช้ได้จริงของ 3000 ตัวอย่าง
    if t1 > min(acc.t.max(), gyr.t.max()):
        raise ValueError("ไฟล์นี้สั้นเกินไป")

    grid = np.linspace(t0, t1, int(round(DUR / FS_DT)) + 1)[:-1]  # 3000 จุดพอดี

    sig = pd.DataFrame({"t": grid})
    for c in ["ax", "ay", "az"]:
        sig[c] = np.interp(grid, acc.t, acc[c])
    for c in ["gx", "gy", "gz"]:
        sig[c] = np.interp(grid, gyr.t, gyr[c])
    return sig


def extract_features(sig):
    rows = []
    for s in range(0, len(sig) - WIN + 1, STEP):
        w = sig[CHANS].iloc[s:s + WIN]
        f = {}
        for c in CHANS:
            f[f"{c}_mean"] = w[c].mean()
            f[f"{c}_std"] = w[c].std()
            f[f"{c}_min"] = w[c].min()
            f[f"{c}_max"] = w[c].max()
        rows.append(f)
    return pd.DataFrame(rows)


def process_folder(folder):
    name = os.path.basename(folder.rstrip("/\\"))
    subject, activity = name.split("_", 1)
    sig = load_signal(
        os.path.join(folder, "Accelerometer.csv"),
        os.path.join(folder, "Gyroscope.csv"),
    )
    feat = extract_features(sig)
    feat["subject"] = subject
    feat["activity"] = activity.lower()
    feat["order"] = np.arange(len(feat))   # ลำดับเวลาภายในไฟล์
    return feat


def split_70_30(all_feat):
    """แบ่งต่อ (subject, activity) ตามลำดับเวลา: 70% แรก = train, 30% หลัง = test"""
    parts = []
    for _, g in all_feat.groupby(["subject", "activity"]):
        g = g.sort_values("order")
        n_train = int(len(g) * TRAIN_RATIO)
        g = g.copy()
        g["split"] = ["train"] * n_train + ["test"] * (len(g) - n_train)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def scale_minus1_1(df):
    """หา min/max จาก train เท่านั้น แล้วใช้กับ train และ test"""
    train = df[df.split == "train"]
    lo = train[FEAT_COLS].min()
    hi = train[FEAT_COLS].max()
    rng = (hi - lo).replace(0, 1)   # กันหารศูนย์ถ้าฟีเจอร์ไหนคงที่
    df = df.copy()
    df[FEAT_COLS] = 2 * (df[FEAT_COLS] - lo) / rng - 1
    return df, lo, hi


if __name__ == "__main__":
    folders = sorted(glob.glob("data/*_*/"))
    all_feat = pd.concat([process_folder(f) for f in folders], ignore_index=True)

    all_feat = split_70_30(all_feat)
    all_feat, lo, hi = scale_minus1_1(all_feat)

    all_feat.drop(columns=["order"]).to_csv("features_ready.csv", index=False)

    print("ทั้งหมด:", all_feat.shape)
    print(all_feat.groupby(["split", "activity"]).size())

    pd.DataFrame({"min": lo, "max": hi}).rename_axis("feature").to_csv("scale_params.csv")