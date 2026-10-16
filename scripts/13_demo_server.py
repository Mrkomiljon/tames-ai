#!/usr/bin/env python
"""Step 13 — live walkthrough UI: serves the walkthrough page and runs the trained models on uploaded images.

    python scripts/13_demo_server.py [--port 8765] [--host 127.0.0.1]

Open the printed address, use "Try your own image". With a delineation mask the expert-delineation models
(runs/tn3k_fold0) are used, without one the fully automatic models (runs/tn3k_automask). Local use only.
"""
import argparse, base64, json, sys, time, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2, numpy as np
from tames.inference import TamesPredictor
from tames.ood import OODGate
from demo_common import ROOT, descriptor_order, encode, fit, render, uri

PRED, ORDER, PAGE, GATE = {}, None, None, None


def predictor(name):
    if name not in PRED:
        PRED[name] = TamesPredictor(automatic=(name == "automatic"))
    return PRED[name]


def decode(b64, flag=cv2.IMREAD_GRAYSCALE):
    img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), flag)
    if img is None:
        raise ValueError("the file is not a readable image")
    return img


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code); self.send_header("Content-Type", ctype + "; charset=utf-8"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html")
        elif self.path.startswith("/api/health"):
            self._send(200, json.dumps({"ok": True, "models": 20}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if not self.path.startswith("/api/predict"):
            return self._send(404, json.dumps({"error": "not found"}))
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            colour = decode(body["image"], cv2.IMREAD_COLOR)
            img = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)
            mask = decode(body["mask"]) > 127 if body.get("mask") else None
            ood = GATE.check(colour)
            name_id = Path(body.get("name") or "upload").stem[:24]
            if ood["status"] == "reject":
                print(f"[predict] {name_id}: rejected by the OOD gate, distance {ood['distance']:.3f} > {ood['reject']:.3f}, {ood['reasons']}", flush=True)
                thumb = cv2.resize(colour, (132, max(1, round(132 * colour.shape[0] / colour.shape[1]))), interpolation=cv2.INTER_AREA)
                return self._send(200, json.dumps({"id": name_id, "truth": None, "rejected": True, "ood": ood, "thumb": uri(thumb, 78), "image": uri(fit(colour, 520))}))
            if mask is not None and mask.shape != img.shape:
                mask = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            name = "expert" if mask is not None else "automatic"
            t = time.time(); P = predictor(name); out = P.predict(img, mask)
            res = {"id": name_id, "truth": None, "ood": ood, "modes": {name: encode(out, ORDER, P.desc_stats)}}
            print(f"[predict] {res['id']}: {name} p_final {out['p_final']:.3f} ({out['label']}) in {time.time() - t:.1f}s; OOD {ood['status']} distance {ood['distance']:.3f}", flush=True)
            self._send(200, json.dumps(res))
        except Exception as e:  # report the reason to the page
            traceback.print_exc(); self._send(400, json.dumps({"error": str(e)}))

    def log_message(self, *a):
        pass


def main():
    global ORDER, PAGE
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8765); ap.add_argument("--host", default="127.0.0.1"); a = ap.parse_args()
    ORDER = descriptor_order()
    built = ROOT / "demo" / "tames_walkthrough.html"
    PAGE = "<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>" + (built.read_text() if built.exists() else render({"cases": []}))
    global GATE
    print("loading models…", flush=True); predictor("expert"); predictor("automatic"); GATE = OODGate.load()
    try:
        srv = ThreadingHTTPServer((a.host, a.port), Handler)
    except OSError:
        srv = ThreadingHTTPServer((a.host, 0), Handler)
        print(f"port {a.port} is in use; using a free port instead")
    print(f"TAMES AI walkthrough at http://{a.host}:{srv.server_address[1]}  (Ctrl+C to stop)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
