import argparse
import csv
import subprocess
import sys
import time
import traceback
from glob import glob
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.append(str(_HERE))
from adaptive_reranking import METHODS as METHOD_DISPATCH, build_command
from check_performance import evaluate

MODELS = ("cosplace", "megaloc")
MATCHERS = ("superpoint-lg", "loftr")
DEFAULT_METHODS = [m for m in METHOD_DISPATCH if m != "local"]
SU_METHODS = ("su", "su_inliers")

SUMMARY_COLUMNS = [
    "method", "model", "matcher", "n_queries",
    "base_r1_pct", "adaptive_r1_pct", "delta_r1",
    "adaptive_r5_pct", "adaptive_r10_pct", "adaptive_r20_pct",
    "matches_per_query", "saving_pct", "seconds", "status",
]


def resolve(template, **kw):
    """Template -> path esistente (glob ammesso). None se manca o e' ambiguo."""
    if not template:
        return None
    pattern = template.format(**kw)
    if any(ch in pattern for ch in "*?["):
        hits = sorted(glob(pattern))
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            print(f"  [ambiguo] {pattern} -> {len(hits)} risultati")
        return None
    return pattern if Path(pattern).exists() else None


def run_one(method, model, matcher, paths, args):
    """Esegue deploy + valutazione per una combinazione. Ritorna una riga dict."""
    out_dir = Path(args.output_root) / method / f"{model}_{matcher}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cli = [
        "--preds-dir", paths["preds"],
        "--model", model, "--matcher", matcher,
        "--output-dir", str(out_dir),
        "--num-preds", str(args.num_preds),
        "--device", args.device,
    ]
    if paths.get("inliers"):
        cli += ["--inliers-dir", paths["inliers"]]
    if method in SU_METHODS:
        cli += ["--z-data", paths["z_data"]]
    if method == "sequential":
        cli += ["--models-dir", str(_HERE / "validation" / "sequential")]

    t0 = time.time()
    cmd = build_command(method, cli)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        err = (res.stderr.strip().splitlines() or ["errore sconosciuto"])[-1]
        raise RuntimeError(err)

    ev = evaluate(paths["preds"], out_dir, num_preds=args.num_preds,
                  recall_values=tuple(args.recall_values),
                  positive_dist_threshold=args.positive_dist_threshold)
    row = {
        "method": method, "model": model, "matcher": matcher,
        "n_queries": ev["n_queries"],
        "base_r1_pct": round(ev["base_recall"][1], 2),
        "adaptive_r1_pct": round(ev["recall"][1], 2),
        "delta_r1": round(ev["recall"][1] - ev["base_recall"][1], 2),
        "matches_per_query": round(ev["matches_per_query"], 2),
        "saving_pct": round(ev["saving_pct"], 2),
        "seconds": round(time.time() - t0, 1),
        "status": "ok",
    }
    for n in (5, 10, 20):
        key = f"adaptive_r{n}_pct"
        row[key] = round(ev["recall"][n], 2) if n in ev["recall"] else ""
    return row


def parse_args():
    p = argparse.ArgumentParser(description="Testa tutti i metodi su tutte le coppie (model, matcher)")
    p.add_argument("--preds-dir-template", required=True,
                   help="cartella dei .txt di retrieval; placeholder {model} {matcher} {dataset}")
    p.add_argument("--inliers-dir-template", default=None,
                   help="OFFLINE: cartella dei .torch top-20 gia' calcolati (consigliato)")
    p.add_argument("--z-data-template", default=None,
                   help="z_data.torch, necessario solo per su / su_inliers")
    p.add_argument("--dataset", default="", help="valore per il placeholder {dataset}")
    p.add_argument("--output-root", required=True)
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                   choices=list(METHOD_DISPATCH.keys()))
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--matchers", nargs="+", default=list(MATCHERS))
    p.add_argument("--num-preds", type=int, default=20)
    p.add_argument("--recall-values", type=int, nargs="+", default=[1, 5, 10, 20])
    p.add_argument("--positive-dist-threshold", type=int, default=25)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    a = parse_args()
    rows, status = [], []

    for model in a.models:
        for matcher in a.matchers:
            fmt = dict(model=model, matcher=matcher, dataset=a.dataset)
            paths = {
                "preds":   resolve(a.preds_dir_template, **fmt),
                "inliers": resolve(a.inliers_dir_template, **fmt),
                "z_data":  resolve(a.z_data_template, **fmt),
            }
            print("\n" + "=" * 78)
            print(f"{model} / {matcher}")
            print(f"  preds   : {paths['preds']}")
            print(f"  inliers : {paths['inliers'] or '(modalita LIVE: matching eseguito ora)'}")
            print(f"  z_data  : {paths['z_data'] or '-'}")
            print("=" * 78)

            if paths["preds"] is None:
                for m in a.methods:
                    status.append((m, model, matcher, "saltato: preds-dir non trovata"))
                continue

            for method in a.methods:
                if method in SU_METHODS and not paths["z_data"]:
                    status.append((method, model, matcher, "saltato: z_data non trovato"))
                    print(f"  [saltato] {method}: z_data non trovato")
                    continue
                print(f"\n--- {method} ---")
                try:
                    rows.append(run_one(method, model, matcher, paths, a))
                    r = rows[-1]
                    print(f"  R@1 {r['adaptive_r1_pct']:.2f}% (base {r['base_r1_pct']:.2f}%, "
                          f"delta {r['delta_r1']:+.2f})  IM/query {r['matches_per_query']:.2f}  "
                          f"saving {r['saving_pct']:.1f}%  [{r['seconds']}s]")
                    status.append((method, model, matcher, "ok"))
                except Exception as e:
                    msg = str(e).splitlines()[0] if str(e) else type(e).__name__
                    kind = "saltato" if "non trovat" in msg or "not found" in msg else "FALLITO"
                    status.append((method, model, matcher, f"{kind}: {msg}"))
                    print(f"  [{kind}] {msg}")
                    if kind == "FALLITO":
                        traceback.print_exc()

    summary_csv = Path(a.output_root) / "summary_deploy.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["model"], r["matcher"], r["method"])):
            w.writerow(r)

    print("\n" + "=" * 78 + "\nRIEPILOGO\n" + "=" * 78)
    if rows:
        print(f"{'method':<24}{'model':<10}{'matcher':<15}{'R@1':>7}{'base':>8}"
              f"{'IM/q':>7}{'saving':>8}")
        for r in sorted(rows, key=lambda r: (r["model"], r["matcher"], -r["adaptive_r1_pct"])):
            print(f"{r['method']:<24}{r['model']:<10}{r['matcher']:<15}"
                  f"{r['adaptive_r1_pct']:>7.2f}{r['base_r1_pct']:>8.2f}"
                  f"{r['matches_per_query']:>7.2f}{r['saving_pct']:>7.1f}%")
    n_ok = sum(1 for s in status if s[3] == "ok")
    n_fail = sum(1 for s in status if s[3].startswith("FALLITO"))
    n_skip = len(status) - n_ok - n_fail
    if n_skip or n_fail:
        print("\nNon eseguiti:")
        for method, model, matcher, st in status:
            if st != "ok":
                print(f"  {method:<24}{model:<10}{matcher:<15}{st}")
    print(f"\n{n_ok} ok, {n_skip} saltati, {n_fail} falliti.")
    print(f"Tabella riassuntiva: {summary_csv}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
