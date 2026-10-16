#!/usr/bin/env python
"""Step 15 — local DICOM study viewer: builds a self-contained HTML page (slideshow of all images, cine loops
played at their recorded frame rate, header table, English/Uzbek) from a folder of ultrasound DICOM files.

    python scripts/15_dicom_viewer.py STUDY_FOLDER [--out viewer.html] [--open]

The page embeds patient data. Keep it on the local machine: the default output is next to the study folder,
never inside the repository, and nothing is uploaded.
"""
import argparse, base64, glob, json, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2, numpy as np, pydicom

HEADER = [("PatientID", "Patient ID", "Bemor ID"), ("PatientAge", "Age", "Yoshi"), ("PatientName", "Name", "Ismi"), ("PatientBirthDate", "Birth date", "Tug‘ilgan sana"),
          ("PatientSex", "Sex", "Jinsi"), ("PatientWeight", "Weight", "Vazni"), ("PatientSize", "Height", "Bo‘yi"), ("InstitutionName", "Institution", "Muassasa"),
          ("InstitutionalDepartmentName", "Department", "Bo‘lim"), ("StudyDate", "Study date", "Tekshiruv sanasi"), ("StudyTime", "Study time", "Tekshiruv vaqti"),
          ("StudyID", "Study ID", "Tekshiruv raqami"), ("AccessionNumber", "Accession number", "Accession raqami"), ("ReferringPhysicianName", "Referring physician", "Yo‘llagan shifokor"),
          ("Manufacturer", "Manufacturer", "Ishlab chiqaruvchi"), ("ManufacturerModelName", "Model", "Model"), ("DeviceSerialNumber", "Serial number", "Seriya raqami"),
          ("SoftwareVersions", "Software", "Dastur versiyasi"), ("ProtocolName", "Protocol", "Protokol")]


def jpg(rgb, q=84, max_w=1280):
    if rgb.shape[1] > max_w:
        rgb = cv2.resize(rgb, (max_w, round(rgb.shape[0] * max_w / rgb.shape[1])), interpolation=cv2.INTER_AREA)
    ok, b = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(b.tobytes()).decode()


def rgb_of(a, photometric):
    if a.ndim == 2:
        if photometric == "MONOCHROME1":
            a = a.max() - a
        a = np.repeat(a[..., None], 3, axis=2)
    if a.dtype != np.uint8:
        f = a.astype(np.float32); a = (255 * (f - f.min()) / max(float(f.max() - f.min()), 1.0)).astype(np.uint8)
    return a


def fmt(k, v):
    v = str(v).strip()
    if not v or v in ("0", "000M", "000Y", "^"):
        return ""
    if k in ("StudyDate", "PatientBirthDate") and len(v) == 8:
        return f"{v[:4]}-{v[4:6]}-{v[6:]}"
    if k == "StudyTime" and len(v) >= 6:
        return f"{v[:2]}:{v[2:4]}:{v[4:6]}"
    return v


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("study"); ap.add_argument("--out", default=None); ap.add_argument("--open", action="store_true"); a = ap.parse_args()
    study = Path(a.study).resolve()
    out = Path(a.out) if a.out else study.parent / f"{study.name[:24]}_viewer.html"
    if Path(__file__).resolve().parents[1] in out.resolve().parents:
        sys.exit("refusing to write patient images inside the repository; choose an --out outside it")
    items, header = [], None
    for f in sorted(glob.glob(str(study / "*.dcm"))):
        ds = pydicom.dcmread(f); n = int(ds.get("NumberOfFrames", 1) or 1); ph = str(ds.get("PhotometricInterpretation", ""))
        arr = ds.pixel_array
        reg = ds.get("SequenceOfUltrasoundRegions")
        box = None
        if reg:
            r = max(reg, key=lambda r: (r.RegionLocationMaxX1 - r.RegionLocationMinX0) * (r.RegionLocationMaxY1 - r.RegionLocationMinY0))
            box = [int(r.RegionLocationMinX0), int(r.RegionLocationMinY0), int(r.RegionLocationMaxX1), int(r.RegionLocationMaxY1)]
        frames = [rgb_of(arr[i], ph) for i in range(n)] if n > 1 else [rgb_of(arr, ph)]
        H, W = frames[0].shape[:2]
        items.append({"name": Path(f).name, "frames": [jpg(fr, 80 if n > 1 else 86) for fr in frames], "fps": float(ds.get("RecommendedDisplayFrameRate", 0) or ds.get("CineRate", 0) or 0) or 19.0,
                      "w": W, "h": H, "box": box, "sop": str(ds.SOPClassUID.name)})
        if header is None:
            header = [{"key": k, "en": en, "uz": uz, "value": fmt(k, ds.get(k, ""))} for k, en, uz in HEADER]
        print(f"{Path(f).name[:40]:40s} {W}x{H} frames {n}")
    data = json.dumps({"study": study.name, "header": header, "items": items}, separators=(",", ":")).replace("</", "<\\/")
    out.write_text(PAGE.replace("__DATA__", data))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    if a.open:
        subprocess.run(["open", str(out)])


PAGE = r"""<!doctype html><html lang="uz"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ultrasound Study Viewer</title>
<style>
  /* Layout: dark reading room. Film strip on the left, the current image large in the centre, study header on the right. */
  :root { color-scheme: dark; --bg:#0b0e11; --panel:#12171c; --panel2:#182027; --line:#26303a; --ink:#e8ecef; --ink2:#c6ced5; --muted:#93a0ac; --faint:#5f6c78; --accent:#56b4e9; --amber:#e6b84f;
          --mono:"SF Mono",Menlo,Consolas,monospace; --sans:-apple-system,"Helvetica Neue",Arial,sans-serif; }
  * { box-sizing:border-box; } html,body { height:100%; margin:0; } body { background:var(--bg); color:var(--ink); font:14px/1.5 var(--sans); }
  .app { display:grid; grid-template-columns:180px minmax(0,1fr) 300px; grid-template-rows:auto minmax(0,1fr) auto; height:100%; gap:0; }
  header { grid-column:1/-1; display:flex; flex-wrap:wrap; align-items:center; gap:10px 18px; padding:12px 18px; border-bottom:1px solid var(--line); }
  header h1 { margin:0; font-size:1.05rem; } header .sub { color:var(--muted); font-size:.82rem; font-family:var(--mono); }
  .sp { flex:1; }
  .seg { display:inline-flex; border:1px solid var(--line); border-radius:999px; padding:3px; background:var(--panel); }
  .seg button { border:0; background:transparent; color:var(--muted); font:inherit; font-size:.8rem; padding:4px 12px; border-radius:999px; cursor:pointer; }
  .seg button[aria-pressed=true] { background:var(--accent); color:#06121b; font-weight:700; }
  .strip { overflow-y:auto; border-right:1px solid var(--line); padding:10px; display:flex; flex-direction:column; gap:8px; min-height:0; }
  .strip .thumb { flex:none; }
  .thumb { position:relative; border:1px solid var(--line); border-radius:6px; overflow:hidden; background:#000; cursor:pointer; padding:0; display:block; width:100%; }
  .thumb img { display:block; width:100%; height:auto; } .thumb[aria-current=true] { outline:2px solid var(--accent); outline-offset:1px; }
  .thumb span { position:absolute; left:4px; top:3px; font:11px var(--mono); color:#fff; background:rgba(0,0,0,.6); padding:0 4px; border-radius:3px; }
  .thumb .cine { left:auto; right:4px; background:var(--amber); color:#000; }
  .stage { position:relative; display:grid; place-items:center; min-width:0; min-height:0; padding:14px; }
  .view { position:relative; max-width:100%; max-height:100%; }
  .view img { display:block; max-width:100%; max-height:calc(100vh - 170px); border-radius:4px; animation:fade .5s ease both; }
  @keyframes fade { from { opacity:0; transform:scale(.985); } to { opacity:1; transform:none; } }
  .badge { position:absolute; left:10px; top:10px; font:12px var(--mono); background:rgba(0,0,0,.65); padding:2px 8px; border-radius:4px; color:var(--ink2); }
  aside { border-left:1px solid var(--line); overflow-y:auto; padding:14px 16px; }
  aside h2 { margin:0 0 8px; font-size:.75rem; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); font-family:var(--mono); }
  dl { display:grid; grid-template-columns:auto minmax(0,1fr); gap:6px 12px; margin:0 0 14px; font-size:.84rem; }
  dt { color:var(--muted); } dd { margin:0; font-family:var(--mono); overflow-wrap:anywhere; } dd.empty { color:var(--faint); font-family:var(--sans); font-style:italic; }
  .note { color:var(--faint); font-size:.78rem; margin:0 0 10px; }
  footer { grid-column:1/-1; display:flex; flex-wrap:wrap; align-items:center; gap:8px 14px; padding:10px 18px; border-top:1px solid var(--line); }
  .btn { border:1px solid var(--line); background:var(--panel2); color:var(--ink); font:inherit; font-size:.85rem; padding:6px 12px; border-radius:8px; cursor:pointer; }
  .btn.primary { background:var(--accent); border-color:var(--accent); color:#06121b; font-weight:700; }
  input[type=range] { flex:1; min-width:120px; accent-color:var(--accent); }
  .pos { font:12px var(--mono); color:var(--muted); min-width:9em; text-align:right; }
  button:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
  .warn { color:var(--amber); font-size:.78rem; }
  @media (max-width:900px) { .app { grid-template-columns:minmax(0,1fr); grid-template-rows:auto auto minmax(0,1fr) auto auto; height:auto; }
    .strip { flex-direction:row; overflow-x:auto; border-right:0; border-bottom:1px solid var(--line); } .strip .thumb { width:110px; } aside { border-left:0; border-top:1px solid var(--line); } }
  @media (prefers-reduced-motion:reduce) { .view img { animation:none; } }
</style>
<div class="app">
  <header><h1 id="t-title"></h1><span class="sub" id="study"></span><span class="sp"></span>
    <span class="warn" id="t-local"></span>
    <div class="seg" role="group" aria-label="View"><button id="v-full" aria-pressed="true"></button><button id="v-crop" aria-pressed="false"></button></div>
    <div class="seg" role="group" aria-label="Language"><button id="l-en" aria-pressed="false">EN</button><button id="l-uz" aria-pressed="true">UZ</button></div></header>
  <nav class="strip" id="strip"></nav>
  <main class="stage"><div class="view" id="view"></div></main>
  <aside><h2 id="t-hdr"></h2><dl id="hdr"></dl><p class="note" id="t-hnote"></p><h2 id="t-img"></h2><dl id="imeta"></dl></aside>
  <footer><button class="btn" id="prev">◀</button><button class="btn primary" id="play"></button><button class="btn" id="next">▶</button>
    <input type="range" id="scrub" min="0" value="0" aria-label="Position"><span class="pos" id="pos"></span>
    <label class="note" style="margin:0"><span id="t-speed"></span> <select id="speed"><option value="3000">3 s</option><option value="2000" selected>2 s</option><option value="1200">1.2 s</option></select></label></footer>
</div>
<script id="d" type="application/json">__DATA__</script>
<script>
(() => {
  const D = JSON.parse(document.getElementById("d").textContent), $ = (i) => document.getElementById(i);
  const T = { en: { title: "Ultrasound study viewer", local: "Patient data: this file stays on this computer.", full: "Full screen frame", crop: "Ultrasound region only", hdr: "Study header", img: "This image",
      hnote: "Empty fields are shown as not filled in. The full-frame view also shows the text the scanner burned into the screen.", play: "Play", pause: "Pause", speed: "Seconds per image", empty: "not filled in",
      file: "File", size: "Size", frames: "Frames", type: "Object", pos: (i, n, f, m) => `image ${i} / ${n}` + (m > 1 ? ` · frame ${f} / ${m}` : ""), cine: "cine" },
    uz: { title: "UTT tekshiruvini ko‘rish", local: "Bemor ma’lumotlari: bu fayl faqat shu kompyuterda.", full: "To‘liq ekran kadri", crop: "Faqat UTT sohasi", hdr: "Tekshiruv ma’lumotlari", img: "Shu rasm",
      hnote: "Bo‘sh maydonlar «kiritilmagan» deb ko‘rsatilgan. To‘liq kadrda apparat ekranga yozgan matn ham ko‘rinadi.", play: "Ijro etish", pause: "To‘xtatish", speed: "Bir rasmga soniya", empty: "kiritilmagan",
      file: "Fayl", size: "O‘lcham", frames: "Kadrlar", type: "Obyekt", pos: (i, n, f, m) => `${n} rasmdan ${i}-rasm` + (m > 1 ? ` · ${m} kadrdan ${f}-kadr` : ""), cine: "cine" } };
  let lang = "uz", crop = false, i = 0, f = 0, playing = !matchMedia("(prefers-reduced-motion: reduce)").matches, timer = null;
  const thumbs = D.items.map((it) => it.frames[Math.floor(it.frames.length / 2)]);
  function frameSrc(it, k) { return it.frames[k]; }
  function draw() {
    const t = T[lang], it = D.items[i], m = it.frames.length;
    document.documentElement.lang = lang; $("t-title").textContent = t.title; $("t-local").textContent = t.local; $("v-full").textContent = t.full; $("v-crop").textContent = t.crop;
    $("t-hdr").textContent = t.hdr; $("t-hnote").textContent = t.hnote; $("t-img").textContent = t.img; $("t-speed").textContent = t.speed; $("play").textContent = playing ? t.pause : t.play;
    $("study").textContent = D.study; $("v-full").setAttribute("aria-pressed", !crop); $("v-crop").setAttribute("aria-pressed", crop);
    $("l-en").setAttribute("aria-pressed", lang === "en"); $("l-uz").setAttribute("aria-pressed", lang === "uz");
    $("hdr").innerHTML = D.header.map((h) => `<dt>${h[lang]}</dt><dd class="${h.value ? "" : "empty"}">${h.value ? h.value.replace(/</g, "&lt;") : t.empty}</dd>`).join("");
    $("imeta").innerHTML = `<dt>${t.file}</dt><dd>${it.name}</dd><dt>${t.size}</dt><dd>${it.w} × ${it.h}</dd><dt>${t.frames}</dt><dd>${m}${m > 1 ? ` · ${it.fps.toFixed(0)} fps` : ""}</dd><dt>${t.type}</dt><dd>${it.sop}</dd>`;
    const img = new Image(); img.src = frameSrc(it, f); img.alt = it.name;
    if (crop && it.box) {
      const [x0, y0, x1, y1] = it.box, cw = x1 - x0 + 1, ch = y1 - y0 + 1, c = document.createElement("canvas"); c.width = cw; c.height = ch;
      img.onload = () => { const sx = img.naturalWidth / it.w; c.getContext("2d").drawImage(img, x0 * sx, y0 * sx, cw * sx, ch * sx, 0, 0, cw, ch); const o = new Image(); o.src = c.toDataURL("image/jpeg", 0.9); o.alt = it.name; mount(o); };
    } else { mount(img); }
    $("scrub").max = D.items.length - 1; $("scrub").value = i; $("pos").textContent = t.pos(i + 1, D.items.length, f + 1, m);
    $("strip").querySelectorAll(".thumb").forEach((b, k) => b.setAttribute("aria-current", k === i));
  }
  function mount(img) { const v = $("view"); if (f > 0 && v.firstChild) { img.style.animation = "none"; } v.replaceChildren(img); const it = D.items[i]; if (it.frames.length > 1) { const b = document.createElement("span"); b.className = "badge"; b.textContent = `${T[lang].cine} ${f + 1}/${it.frames.length}`; v.appendChild(b); } }
  function schedule() {
    clearTimeout(timer); if (!playing) return; const it = D.items[i];
    if (it.frames.length > 1 && f < it.frames.length - 1) { timer = setTimeout(() => { f++; draw(); schedule(); }, 1000 / it.fps); return; }
    timer = setTimeout(() => { i = (i + 1) % D.items.length; f = 0; draw(); schedule(); }, it.frames.length > 1 ? 600 : +$("speed").value);
  }
  $("strip").innerHTML = D.items.map((it, k) => `<button class="thumb" data-k="${k}"><img src="${thumbs[k]}" alt=""><span>${String(k + 1).padStart(2, "0")}</span>${it.frames.length > 1 ? `<span class="cine">cine</span>` : ""}</button>`).join("");
  $("strip").querySelectorAll(".thumb").forEach((b) => (b.onclick = () => { i = +b.dataset.k; f = 0; draw(); schedule(); }));
  $("prev").onclick = () => { i = (i - 1 + D.items.length) % D.items.length; f = 0; draw(); schedule(); };
  $("next").onclick = () => { i = (i + 1) % D.items.length; f = 0; draw(); schedule(); };
  $("play").onclick = () => { playing = !playing; draw(); schedule(); };
  $("scrub").oninput = (e) => { i = +e.target.value; f = 0; draw(); schedule(); };
  $("v-full").onclick = () => { crop = false; draw(); }; $("v-crop").onclick = () => { crop = true; draw(); };
  $("l-en").onclick = () => { lang = "en"; draw(); }; $("l-uz").onclick = () => { lang = "uz"; draw(); };
  document.addEventListener("keydown", (e) => { if (e.key === "ArrowRight") $("next").click(); if (e.key === "ArrowLeft") $("prev").click(); if (e.key === " ") { e.preventDefault(); $("play").click(); } });
  draw(); schedule();
})();
</script></html>"""

if __name__ == "__main__":
    main()
