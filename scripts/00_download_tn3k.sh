#!/usr/bin/env bash
# Downloads TN3K (images + masks + fold JSONs) from the TRFE-Net repository and
# extracts it into data/raw/tn3k. Needs: git, unrar (brew install unrar | apt install unrar).
#
# Labels (0 = benign, 1 = malignant) come from the ACL release (Gong et al., MICCAI 2022):
#   https://github.com/chenghui-666/ACL  -> Google Drive / Baidu: label4trainval.csv, label4test.csv
# data/labels/ already holds a copy rebuilt from the CC-BY-4.0 manifest of
#   https://github.com/b-kiani/Case-level-audit-of-thyroid-ultrasound-and-cytology-deep-learning
# Before publishing, download the official CSVs and run:  python scripts/01_prepare_data.py --verify-labels <dir>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw/tn3k"
TMP="$ROOT/data/raw/_trfe"
mkdir -p "$RAW"
# Homebrew no longer ships unrar; `unar` (brew install unar) is accepted as a fallback.
if command -v unrar >/dev/null; then
  extract() { unrar x -y -idq "$1"; }
elif command -v unar >/dev/null; then
  extract() { unar -q -f -o . "$1"; }
else
  echo "neither unrar nor unar found: brew install unar  (macOS)  |  apt install unrar  (Debian/Ubuntu)"; exit 1
fi
[ -d "$TMP" ] || git clone --depth 1 https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation.git "$TMP"
cd "$RAW"
for f in trainval-image.part1.rar test-image.part1.rar trainval-mask.rar test-mask.rar; do
  extract "$TMP/picture/$f"
done
cp "$TMP"/picture/tn3k-trainval-fold*.json .
for d in trainval-image test-image trainval-mask test-mask; do echo "$d: $(ls "$d" | wc -l) files"; done
echo "Expected: 2879 trainval, 614 test."
echo "Cite: Gong et al., ISBI 2021 (10.1109/ISBI48211.2021.9434087); Gong et al., CBM 2022; Gong et al., MICCAI 2022 (labels)."
