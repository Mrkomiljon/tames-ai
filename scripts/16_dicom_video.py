#!/usr/bin/env python
"""Step 16 — turn an ultrasound DICOM study folder into an MP4 (H.264, plays in any player or phone):
every still image for a few seconds, cine loops at their recorded frame rate.

    python scripts/16_dicom_video.py STUDY_FOLDER [--region] [--seconds 2] [--cine-loops 3] [--out video.mp4]

--region keeps only the scanner's ultrasound region (drops the burned-in header with ID, date, institution).
Without it the full screen frame is used, which contains that header: treat the file as patient data.
The output goes next to the study folder by default; writing inside the repository is refused.
"""
import argparse, glob, subprocess, sys
from pathlib import Path
import cv2, numpy as np, pydicom


def frames_of(ds):
    n = int(ds.get("NumberOfFrames", 1) or 1); a = ds.pixel_array; ph = str(ds.get("PhotometricInterpretation", ""))
    def rgb(x):
        if x.ndim == 2:
            x = (x.max() - x) if ph == "MONOCHROME1" else x
            x = np.repeat(x[..., None], 3, axis=2)
        if x.dtype != np.uint8:
            f = x.astype(np.float32); x = (255 * (f - f.min()) / max(float(f.max() - f.min()), 1.0)).astype(np.uint8)
        return x
    return [rgb(a[i]) for i in range(n)] if n > 1 else [rgb(a)]


def region(ds):
    reg = ds.get("SequenceOfUltrasoundRegions")
    if not reg:
        return None
    r = max(reg, key=lambda r: (r.RegionLocationMaxX1 - r.RegionLocationMinX0) * (r.RegionLocationMaxY1 - r.RegionLocationMinY0))
    return int(r.RegionLocationMinX0), int(r.RegionLocationMinY0), int(r.RegionLocationMaxX1), int(r.RegionLocationMaxY1)


def letterbox(img, W, H):
    h, w = img.shape[:2]; s = min(W / w, H / h); nw, nh = max(2, round(w * s)), max(2, round(h * s))
    out = np.zeros((H, W, 3), np.uint8); y, x = (H - nh) // 2, (W - nw) // 2
    out[y:y + nh, x:x + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    return out


def label(img, text):
    cv2.rectangle(img, (0, img.shape[0] - 30), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
    cv2.putText(img, text, (12, img.shape[0] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 210, 220), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("study"); ap.add_argument("--region", action="store_true")
    ap.add_argument("--seconds", type=float, default=2.0); ap.add_argument("--cine-loops", type=int, default=3); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    study = Path(a.study).resolve()
    out = Path(a.out).resolve() if a.out else study.parent / f"{study.name[:24]}_{'region' if a.region else 'full'}.mp4"
    if Path(__file__).resolve().parents[1] in out.parents:
        sys.exit("refusing to write patient images inside the repository; choose an --out outside it")
    files = sorted(glob.glob(str(study / "*.dcm")))
    clips = []
    for f in files:
        ds = pydicom.dcmread(f); fr = frames_of(ds); box = region(ds)
        if a.region and box:
            x0, y0, x1, y1 = box; fr = [x[y0:y1 + 1, x0:x1 + 1] for x in fr]
        fps = float(ds.get("RecommendedDisplayFrameRate", 0) or ds.get("CineRate", 0) or 0) or 19.0
        clips.append((fr, fps))
    W = max(c[0][0].shape[1] for c in clips); H = max(c[0][0].shape[0] for c in clips); W, H = W + W % 2, H + H % 2 + 30
    FPS = 19
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                           "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    n = len(clips); total = 0
    for k, (fr, fps) in enumerate(clips):
        if len(fr) == 1:
            img = label(np.vstack([letterbox(cv2.cvtColor(fr[0], cv2.COLOR_RGB2BGR), W, H - 30), np.zeros((30, W, 3), np.uint8)]), f"{k + 1}/{n}")
            for _ in range(round(a.seconds * FPS)):
                ff.stdin.write(img.tobytes()); total += 1
        else:
            rep = max(1, round(FPS / fps))
            for loop in range(a.cine_loops):
                for j, x in enumerate(fr):
                    img = label(np.vstack([letterbox(cv2.cvtColor(x, cv2.COLOR_RGB2BGR), W, H - 30), np.zeros((30, W, 3), np.uint8)]), f"{k + 1}/{n}  cine {j + 1}/{len(fr)}")
                    for _ in range(rep):
                        ff.stdin.write(img.tobytes()); total += 1
    ff.stdin.close(); ff.wait()
    print(f"wrote {out}: {W}x{H}, {total / FPS:.0f} s, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
