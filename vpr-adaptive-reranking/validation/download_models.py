"""
validation/download_models.py — Download the trained regressors (model JSON
files) from Google Drive into validation/<subdir>/.

The JSON files are produced by training and are NOT tracked in the repository
(see .gitignore). They are shipped as ONE zip on Google Drive
(validation_models.zip) whose internal layout is:

    logistic_hard/model_<model>_<matcher>.json
    logistic_help/model_<model>_<matcher>.json
    logistic_cost_sensitive/model_logistic_cost_sensitive_<model>_<matcher>.json
    su/model_su_<model>_<matcher>.json
    su_inliers/model_su_num_inliers_<model>_<matcher>.json
    sequential/seq_model_continue_{1,5,10}_*<model>*<matcher>*.json

with <model> in {cosplace, megaloc} and <matcher> in {superpoint-lg, loftr}.
A leading "validation/" folder inside the zip is tolerated.

Usage (Colab cell):
    !pip -q install gdown
    !python VPR-Adaptive-ReRanking/validation/download_models.py --url "<Google Drive share link>"
    # or, if the zip is already on your mounted Drive:
    !python VPR-Adaptive-ReRanking/validation/download_models.py --zip /content/drive/MyDrive/VPR/validation_models.zip
    # only check which JSON files are present:
    !python VPR-Adaptive-ReRanking/validation/download_models.py --check

Set MODELS_URL below once, and --url becomes optional.
"""
import argparse
import sys
import zipfile
from glob import glob
from pathlib import Path, PurePosixPath

# ---> paste here the Google Drive share link of validation_models.zip <---
MODELS_URL = ""

_HERE = Path(__file__).resolve().parent
_DOWNLOAD_DIR = _HERE / "_downloads"

MODELS   = ("cosplace", "megaloc")
MATCHERS = ("superpoint-lg", "loftr")
MATCHER_FILE_TOKENS = {"superpoint-lg": ("superpoint-lg", "sp-lg"), "loftr": ("loftr",)}

# subdir -> list of glob templates ({model}, {matcher})
EXPECTED = {
    "logistic_hard":           ["model_{model}_{matcher}.json"],
    "logistic_help":           ["model_{model}_{matcher}.json"],
    "logistic_cost_sensitive": ["model_logistic_cost_sensitive_{model}_{matcher}.json"],
    "su":                      ["model_su_{model}_{matcher}.json"],
    "su_inliers":              ["model_su_num_inliers_{model}_{matcher}.json"],
    "sequential":              ["*continue_1*{model}*{matcher}*.json",
                                "*continue_5*{model}*{matcher}*.json",
                                "*continue_10*{model}*{matcher}*.json"],
}


def download(url=None, file_id=None):
    try:
        import gdown
    except ImportError:
        sys.exit("gdown is not installed: run   pip install gdown   and retry.")
    _DOWNLOAD_DIR.mkdir(exist_ok=True)
    out = _DOWNLOAD_DIR / "validation_models.zip"
    if file_id:
        url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading {url}\n  -> {out}")
    got = gdown.download(url, str(out), quiet=False, fuzzy=True)
    if not got or not out.exists() or out.stat().st_size == 0:
        sys.exit("Download failed. Check that the link is shared as 'Anyone with the link'.")
    return out


def install_zip(zip_path):
    """Extract only the .json members into validation/<subdir>/ (safe paths only)."""
    zip_path = Path(zip_path)
    if not zip_path.exists():
        sys.exit(f"zip not found: {zip_path}")
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            p = PurePosixPath(member)
            if p.suffix.lower() != ".json" or p.name.startswith(".") or "__MACOSX" in p.parts:
                continue
            parts = [x for x in p.parts if x not in ("", ".")]
            if parts and parts[0] == "validation":
                parts = parts[1:]
            if any(x == ".." for x in parts) or len(parts) != 2 or parts[0] not in EXPECTED:
                print(f"  [skip] unexpected path inside zip: {member}")
                continue
            dest = _HERE / parts[0] / parts[1]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(member))
            n += 1
    print(f"Installed {n} JSON files into {_HERE}")


def check():
    """Print which expected JSON files are present/missing. Returns #missing."""
    missing = 0
    print(f"\nModel JSON files in {_HERE}:")
    for subdir, templates in EXPECTED.items():
        for model in MODELS:
            for matcher in MATCHERS:
                for tmpl in templates:
                    hits = []
                    for tok in MATCHER_FILE_TOKENS[matcher]:
                        hits += glob(str(_HERE / subdir / tmpl.format(model=model, matcher=tok)))
                    label = f"{subdir}/{tmpl.format(model=model, matcher=matcher)}"
                    if hits:
                        print(f"  [ok]      {label}")
                    else:
                        print(f"  [MISSING] {label}")
                        missing += 1
    print(f"\n{missing} missing file(s)." if missing else "\nAll expected model JSON files are present.")
    return missing


def parse_args():
    p = argparse.ArgumentParser(description="Download the trained regressors (JSON) from Google Drive")
    p.add_argument("--url",     default=None, help="Google Drive share link of validation_models.zip (default: MODELS_URL)")
    p.add_argument("--file-id", default=None, help="alternatively, the Google Drive file id")
    p.add_argument("--zip",     default=None, help="install from a local zip instead of downloading")
    p.add_argument("--check",   action="store_true", help="only list present/missing files")
    return p.parse_args()


def main():
    a = parse_args()
    if a.check:
        sys.exit(1 if check() else 0)
    if a.zip:
        install_zip(a.zip)
    else:
        url = a.url or (None if a.file_id else MODELS_URL)
        if not url and not a.file_id:
            sys.exit("No link: pass --url '<Google Drive share link>' (or set MODELS_URL in this file).")
        install_zip(download(url=url, file_id=a.file_id))
    sys.exit(1 if check() else 0)


if __name__ == "__main__":
    main()
