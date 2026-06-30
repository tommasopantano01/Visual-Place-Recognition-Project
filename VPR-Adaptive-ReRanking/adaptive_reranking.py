"""
adaptive_reranking.py — Orchestratore: sceglie il tipo di threshold
(--threshold) e richiama lo script corrispondente in methods/.

NOTA (fix): la versione precedente cercava ogni script in una sottocartella
omonima (es. su/su.py), ma tutti gli script dei metodi vivono in methods/.
Questa versione punta a methods/<script> per ogni metodo.

Uso:
    python VPR-Adaptive-ReRanking/adaptive_reranking.py --threshold su \
        --preds-dir preds/ --z-data z_data.torch --matcher superpoint-lg \
        --output-dir out/ --criterion "P(help)-aP(hurts)"
"""
import argparse
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_METHODS_DIR = _HERE / "methods"

# tipo di threshold -> script in methods/
_METHODS = {
    "youden":                  "youden.py",
    "best_r1":                 "best_r1.py",
    "efficiency":              "efficiency.py",
    "local":                   "local.py",
    "logistic_hard":           "logistic_hard.py",
    "logistic_help":           "logistic_help.py",
    "logistic_cost_sensitive": "logistic_cost_sensitive.py",
    "su":                      "su.py",
    "su_inliers":              "su_inliers.py",
    "sequential":              "sequential.py",
}


def parse_known():
    parser = argparse.ArgumentParser(
        description="Adaptive reranking — orchestratore",
        epilog=f"Tipi di threshold disponibili: {', '.join(_METHODS)}",
    )
    parser.add_argument("--threshold", required=True, choices=list(_METHODS.keys()))
    args, rest = parser.parse_known_args()
    return args, rest


def main():
    args, rest = parse_known()
    script_path = _METHODS_DIR / _METHODS[args.threshold]

    if not script_path.exists():
        raise FileNotFoundError(f"Script non trovato: {script_path}")

    cmd = [sys.executable, str(script_path)] + rest
    print(f"[adaptive_reranking] threshold='{args.threshold}' -> {script_path}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
