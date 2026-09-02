"""
adaptive_reranking.py — Test orchestrator.

One single command for every method: --threshold picks the method, all other
arguments are forwarded to the corresponding family script in methods/.
"""
import argparse
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_METHODS_DIR = _HERE / "methods"

METHODS = {
    "youden":                  ("Thresholds.py", ["--method", "youden"]),
    "best_r1":                 ("Thresholds.py", ["--method", "best_r1"]),
    "efficiency":              ("Thresholds.py", ["--method", "efficiency"]),
    "logistic_hard":           ("logistic.py",   ["--method", "hard"]),
    "logistic_help":           ("logistic.py",   ["--method", "help"]),
    "logistic_cost_sensitive": ("logistic.py",   ["--method", "cost_sensitive"]),
    "su":                      ("su.py",         ["--features", "su"]),
    "su_inliers":              ("su.py",         ["--features", "su_inliers"]),
    "sequential":              ("sequential.py", []),
}


def build_command(method, extra_args):
    script_name, fixed_args = METHODS[method]
    script_path = _METHODS_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    return [sys.executable, str(script_path)] + fixed_args + list(extra_args)


def parse_known():
    parser = argparse.ArgumentParser(
        description="Adaptive reranking — deploy orchestrator",
        epilog=f"Available methods: {', '.join(METHODS)}",
    )
    parser.add_argument("--threshold", required=True, choices=list(METHODS.keys()),
                        help="decision method to use")
    return parser.parse_known_args()


def main():
    args, rest = parse_known()
    cmd = build_command(args.threshold, rest)
    print(f"[adaptive_reranking] '{args.threshold}' -> {' '.join(cmd[1:])}")
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
