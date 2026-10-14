"""Chay toan bo test case + ablation, in bang ket qua (dua vao bao cao)."""
import glob, json, os, time
import cv2
from pipeline import segment_page
from evaluate import prf

CONFIGS = {
    "Day du":            {},
    "Khong Gauss":       dict(use_blur=False),
    "Khong Closing":     dict(use_closing=False),
    "Khong loc dien tich": dict(use_area_filter=False),
    "Closing 5px":       dict(close_px=5),
    "Closing 40px":      dict(close_px=40),
}

if __name__ == "__main__":
    files = sorted(glob.glob("test_cases/*.png"))
    print(f"{'Cau hinh':22s}{'Case':8s}{'P':>6s}{'R':>6s}{'F1':>6s}{'F1@0.8':>8s}{'t(s)':>7s}")
    for name, cfg in CONFIGS.items():
        for case in ("case1", "case2", "case3"):
            Ps, Rs, Fs, F8, T = [], [], [], [], []
            for f in [x for x in files if os.path.basename(x).startswith(case)]:
                gt = json.load(open(f.replace(".png", ".json")))
                img = cv2.imread(f)
                t0 = time.time(); pred, _ = segment_page(img, **cfg); T.append(time.time() - t0)
                p, r, fs = prf(pred, gt, 0.5); Ps.append(p); Rs.append(r); Fs.append(fs)
                F8.append(prf(pred, gt, 0.8)[2])
            m = lambda a: sum(a) / len(a)
            print(f"{name:22s}{case:8s}{m(Ps):6.2f}{m(Rs):6.2f}{m(Fs):6.2f}{m(F8):8.2f}{m(T):7.2f}")
