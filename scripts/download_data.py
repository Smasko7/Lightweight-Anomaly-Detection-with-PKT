"""Download ADBench .npz files to data/raw/."""
import urllib.request
import os
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/Minqi824/ADBench/main/adbench/datasets/Classical"
FILES = [
    "26_optdigits.npz",
    "36_speech.npz",
    "24_mnist.npz",
    "19_landsat.npz",
    "3_backdoor.npz",
]

def download():
    out_dir = Path(__file__).parent.parent / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    for fname in FILES:
        dest = out_dir / fname
        if dest.exists():
            print(f"  already exists: {fname}")
            continue
        url = f"{BASE_URL}/{fname}"
        print(f"  downloading {fname} ...")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"  saved to {dest}")
        except Exception as e:
            print(f"  FAILED: {e}")
            print(f"  Manual download: {url}")
            print(f"  Save to: {dest}")

if __name__ == "__main__":
    download()
