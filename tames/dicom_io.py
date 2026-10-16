"""Read ultrasound DICOM files for single-image inference.

Returns the image cropped to the scanner's ultrasound region (Sequence of Ultrasound Regions), so that the
burned-in header (patient name, ID, date) and the scanner's on-screen controls never reach the models or the
page. Nothing patient-identifying is returned; only technical metadata.

    frame, meta = read_ultrasound(path_or_bytes, frame=None)   # frame: RGB uint8 (H, W, 3); meta: dict
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pydicom


def is_dicom(data: bytes) -> bool:
    return len(data) > 132 and data[128:132] == b"DICM"


def _region(ds) -> tuple[int, int, int, int] | None:
    regs = ds.get("SequenceOfUltrasoundRegions")
    if not regs:
        return None
    # largest 2-D tissue region (RegionSpatialFormat 1 = 2-D); fall back to the largest of any kind
    cand = [r for r in regs if int(r.get("RegionSpatialFormat", 1)) == 1] or list(regs)
    r = max(cand, key=lambda r: (r.RegionLocationMaxX1 - r.RegionLocationMinX0) * (r.RegionLocationMaxY1 - r.RegionLocationMinY0))
    return int(r.RegionLocationMinX0), int(r.RegionLocationMinY0), int(r.RegionLocationMaxX1), int(r.RegionLocationMaxY1), r


def read_ultrasound(src: str | Path | bytes, frame: int | None = None) -> tuple[np.ndarray, dict]:
    ds = pydicom.dcmread(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else str(src))
    n = int(ds.get("NumberOfFrames", 1) or 1)
    arr = ds.pixel_array                      # pydicom 3 already returns RGB for YBR-encoded colour data
    if n > 1:
        frame = n // 2 if frame is None else int(np.clip(frame, 0, n - 1))
        arr = arr[frame]
    else:
        frame = 0
    if arr.ndim == 2:                         # MONOCHROME
        if ds.get("PhotometricInterpretation") == "MONOCHROME1":
            arr = arr.max() - arr
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.dtype != np.uint8:
        a = arr.astype(np.float32); arr = (255 * (a - a.min()) / max(float(a.max() - a.min()), 1.0)).astype(np.uint8)
    reg = _region(ds)
    spacing = None
    if reg:
        x0, y0, x1, y1, r = reg
        arr = arr[y0:y1 + 1, x0:x1 + 1]
        if int(r.get("PhysicalUnitsXDirection", 0)) == 3:   # 3 = cm
            spacing = round(float(r.PhysicalDeltaX) * 10.0, 4)
    meta = {"frames": n, "frame": frame, "cropped_to_region": bool(reg), "pixel_spacing_mm": spacing,
            "manufacturer": str(ds.get("Manufacturer", "")), "model": str(ds.get("ManufacturerModelName", "")),
            "photometric": str(ds.get("PhotometricInterpretation", "")), "size": [int(arr.shape[0]), int(arr.shape[1])]}
    return np.ascontiguousarray(arr), meta
