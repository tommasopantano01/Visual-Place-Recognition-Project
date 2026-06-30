"""
adaptive_reranking.py — Orchestratore: sceglie il tipo di threshold
(--threshold) e richiama lo script corrispondente nella sua sottocartella.

Non duplica nessuna logica: ogni metodo resta un .py self-contained nella
propria cartella (youden/, logistic_hard/, su/, sequential/, ecc.), con i
propri threshold.csv / model.json gia' calibrati. Questo file e' solo il
punto d'ingresso unico, cosi' l'utente non deve ricordarsi il path esatto
di ogni sottocartella.

Uso:
    python VPR-adaptive-re-ranking/adaptive_reranking.py --threshold youden \
        --preds-dir preds/ --matcher superpoint-lg --output-dir out/

    python VPR-adaptive-re-ranking/adaptive_reranking.py --threshold su \
        --preds-dir preds/ --z-data z_data.torch --matcher superpoint-lg \
        --output-dir out/

    python VPR-adaptive-re-ranking/adaptive_reranking.py --threshold sequential \
        --preds-dir preds/ --matcher superpoint-lg --output-dir out/

Ogni metodo ha argomenti leggermente diversi (es. su/su_inliers vogliono
anche --z-data): --threshold seleziona solo QUALE script lanciare, gli
argomenti successivi vengono passati cosi' come sono allo script scelto
(passthrough, nessuna validazione qui — ogni metodo valida i propri).
"""
import argparse
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# tipo di threshold -> (sottocartella, script)
_METHODS = {
    "youden":                 ("youden", "youden.py"),
    "best_r1":                ("best_r1", "best_r1.py"),
    "efficiency":              ("efficiency", "efficiency.py"),
    "local":                   ("local", "local.py"),
    "logistic_hard":           ("logistic_hard", "logistic_hard.py"),
    "logistic_help":           ("logistic_help", "logistic_help.py"),
    "logistic_cost_sensitive": ("logistic_cost_sensitive", "logistic_cost_sensitive.py"),
    "su":                      ("su", "su.py"),
    "su_inliers":              ("su_inliers", "su_inliers.py"),
    "sequential":              ("sequential", "sequential.py"),
}


def parse_known():
    parser = argparse.ArgumentParser(
        description="Adaptive reranking — orchestratore",
        epilog=f"Tipi di threshold disponibili: {', '.join(_METHODS)}",
    )
    parser.add_argument("--threshold", required=True, choices=list(_METHODS.keys()))
    # tutto il resto viene passato cosi' com'e' allo script del metodo scelto
    args, rest = parser.parse_known_args()
    return args, rest


def main():
    args, rest = parse_known()
    subfolder, script_name = _METHODS[args.threshold]
    script_path = _HERE / subfolder / script_name

    if not script_path.exists():
        raise FileNotFoundError(f"Script non trovato: {script_path}")

    cmd = [sys.executable, str(script_path)] + rest
    print(f"[adaptive_reranking] threshold='{args.threshold}' -> {script_path}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
