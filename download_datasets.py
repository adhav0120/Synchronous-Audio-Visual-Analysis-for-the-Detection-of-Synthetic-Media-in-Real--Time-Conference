"""
download_datasets.py
====================
Downloads the two datasets needed for calibrated audio model retraining:

  1. LibriSpeech  dev-clean  (~337 MB) — real-world genuine speech (bonafide negatives)
     Downloaded via torchaudio.datasets.LIBRISPEECH (handles redirects automatically)

  2. ASVspoof 2019 LA full bundle (~4.7 GB zipped)
     Tried from multiple mirrors in order.

Run from the project root:
    .venv\Scripts\python.exe download_datasets.py
"""

import os
import sys
import tarfile
import zipfile
import shutil
import urllib.request
import urllib.error
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Progress reporter for urllib
# ─────────────────────────────────────────────────────────────────────────────
def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct  = min(100.0, downloaded * 100.0 / total_size)
        done = int(pct / 2)
        bar  = "█" * done + "░" * (50 - done)
        mb   = downloaded / 1_048_576
        tot  = total_size  / 1_048_576
        print(f"\r  [{bar}] {pct:5.1f}%  {mb:.1f}/{tot:.1f} MB ", end="", flush=True)
    else:
        mb = downloaded / 1_048_576
        print(f"\r  {mb:.1f} MB downloaded…", end="", flush=True)


def download_url(url: str, dest: Path, label: str) -> bool:
    """Download url → dest. Returns True on success."""
    print(f"  Trying: {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    _progress(downloaded // 65536, 65536, total)
        print()
        print(f"  ✓ Downloaded {dest.stat().st_size / 1_048_576:.1f} MB  →  {dest.name}")
        return True
    except Exception as e:
        print(f"\n  ✗ Failed ({e})")
        if dest.exists():
            dest.unlink()
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 1. LibriSpeech dev-clean  →  via torchaudio (most reliable)
# ─────────────────────────────────────────────────────────────────────────────
LIBRI_CHECK = DATA_DIR / "LibriSpeech" / "dev-clean"

def download_librispeech():
    if LIBRI_CHECK.exists():
        n = len(list(LIBRI_CHECK.rglob("*.flac")))
        print(f"  ✓ LibriSpeech dev-clean already present  ({n} flac files)")
        return True

    print("\n⬇  LibriSpeech dev-clean via torchaudio…")
    try:
        import torchaudio
        torchaudio.datasets.LIBRISPEECH(
            root=str(DATA_DIR),
            url="dev-clean",
            download=True
        )
        n = len(list(LIBRI_CHECK.rglob("*.flac")))
        print(f"  ✓ LibriSpeech dev-clean ready  ({n} clips)")
        return True
    except Exception as e:
        print(f"  ✗ torchaudio downloader failed: {e}")
        print("    Trying direct URL fallback…")

    # Fallback mirrors
    mirrors = [
        "https://openslr.magicdatatech.com/resources/12/dev-clean.tar.gz",
        "http://www.openslr.org/resources/12/dev-clean.tar.gz",
        "https://us.openslr.org/resources/12/dev-clean.tar.gz",
    ]
    archive = DATA_DIR / "dev-clean.tar.gz"
    for url in mirrors:
        if download_url(url, archive, "LibriSpeech dev-clean"):
            break
    else:
        print("\n  ✗ All LibriSpeech mirrors failed.")
        print("    Manual download: https://www.openslr.org/12")
        print("    Place dev-clean.tar.gz in  data/  and re-run.")
        return False

    # Extract
    print("  Extracting dev-clean.tar.gz…")
    out = DATA_DIR / "LibriSpeech"
    out.mkdir(exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=out)
    archive.unlink()
    n = len(list(LIBRI_CHECK.rglob("*.flac")))
    print(f"  ✓ Extracted  ({n} clips)")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 2. ASVspoof 2019 LA  →  tried from multiple mirrors
# ─────────────────────────────────────────────────────────────────────────────
ASVSPOOF_CHECK = DATA_DIR / "ASVspoof2019_LA_train"
ASVSPOOF_PROTO = DATA_DIR / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt"

ASVSPOOF_MIRRORS = [
    # Edinburgh DataShare (primary — sometimes accessible without auth)
    "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?sequence=3&isAllowed=y",
    "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?sequence=3&isAllowed=y",
]

def restructure_asvspoof():
    """If zip unpacked into data/LA/, move contents up one level."""
    la_dir = DATA_DIR / "LA"
    if not la_dir.exists():
        return
    print("  ℹ  Restructuring data/LA/ → data/ …")
    for item in list(la_dir.iterdir()):
        dest = DATA_DIR / item.name
        if not dest.exists():
            shutil.move(str(item), str(dest))
            print(f"     Moved: {item.name}")
    try:
        la_dir.rmdir()
    except OSError:
        pass
    print("  ✓ Restructured")

def download_asvspoof():
    if ASVSPOOF_CHECK.exists() and ASVSPOOF_PROTO.exists():
        n = len(list(ASVSPOOF_CHECK.rglob("*.flac")))
        print(f"  ✓ ASVspoof 2019 LA train already present  ({n} flac files)")
        return True

    archive = DATA_DIR / "LA.zip"

    # Try each mirror
    if not archive.exists():
        print(f"\n⬇  ASVspoof 2019 LA  (~4.7 GB zipped)")
        for url in ASVSPOOF_MIRRORS:
            if download_url(url, archive, "ASVspoof 2019 LA"):
                break
        else:
            print("\n  ✗ Could not auto-download ASVspoof 2019 LA.")
            print()
            print("  ──────────────────────────────────────────────────────")
            print("  MANUAL DOWNLOAD REQUIRED (one-time, ~4.7 GB)")
            print("  ──────────────────────────────────────────────────────")
            print()
            print("  1. Open this URL in your browser:")
            print("     https://datashare.ed.ac.uk/handle/10283/3336")
            print()
            print("  2. Click  'LA.zip'  and agree to the license")
            print()
            print("  3. Save the file to:")
            print(f"     {archive}")
            print()
            print("  4. Re-run this script — it will auto-extract.")
            print()
            return False

    # Extract ZIP
    print(f"\n📦 Extracting LA.zip  ({archive.stat().st_size/1_048_576:.0f} MB)…")
    with zipfile.ZipFile(archive, "r") as z:
        names = z.namelist()
        for i, name in enumerate(names, 1):
            z.extract(name, path=DATA_DIR)
            if i % 1000 == 0 or i == len(names):
                pct = i * 100 // len(names)
                print(f"\r  Extracting… {pct}% ({i}/{len(names)} files)", end="", flush=True)
    print()

    restructure_asvspoof()

    n = len(list(ASVSPOOF_CHECK.rglob("*.flac"))) if ASVSPOOF_CHECK.exists() else 0
    print(f"  ✓ ASVspoof 2019 LA extracted  ({n} audio files)")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 62)
    print("  SHIELD — Dataset Downloader for Audio Model Retraining")
    print("=" * 62)
    print(f"  Data dir: {DATA_DIR}\n")

    ok_libri    = False
    ok_asvspoof = False

    # ── 1. LibriSpeech dev-clean ─────────────────────────────────
    print("━" * 62)
    print("  [1 / 2]  LibriSpeech dev-clean  (~337 MB, ~2700 clips)")
    print("           Genuine real-world speech  →  bonafide labels")
    print("━" * 62)
    ok_libri = download_librispeech()

    # ── 2. ASVspoof 2019 LA ──────────────────────────────────────
    print()
    print("━" * 62)
    print("  [2 / 2]  ASVspoof 2019 LA  (~4.7 GB zipped)")
    print("           Spoof + bonafide studio speech  →  both labels")
    print("━" * 62)
    ok_asvspoof = download_asvspoof()

    # ── Summary ──────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("  DOWNLOAD SUMMARY")
    print("=" * 62)

    def status(ok, label):
        icon = "✅" if ok else "❌"
        print(f"  {icon}  {label}")

    status(ok_libri,    "LibriSpeech dev-clean   (bonafide negatives)")
    status(ok_asvspoof, "ASVspoof 2019 LA train  (spoof + bonafide)")

    if ok_libri and ok_asvspoof:
        print()
        print("  🎉 Both datasets ready!")
        print("  Next step:")
        print("    .venv\\Scripts\\python.exe src/retrain_calibrated.py")
    elif ok_libri and not ok_asvspoof:
        print()
        print("  ⚠  LibriSpeech ready but ASVspoof needs manual download.")
        print("  Once LA.zip is in data\\, re-run this script.")
    else:
        print()
        print("  ❌ Please resolve download errors above and re-run.")
    print("=" * 62)


if __name__ == "__main__":
    main()
