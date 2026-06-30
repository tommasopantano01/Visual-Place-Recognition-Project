"""
helps_estimator.py — Stima la threshold per il re-ranking adattivo.

Implementa le due strategie descritte nel report in "Local probability and
logistic policies", entrambe basate SOLO su num_inliers_top1:

  local    → stima non parametrica di P(helps | I_1(q))
             finestra adattiva centrata su I_1(q), espansa finche' non
             contiene almeno K_min campioni di train.

  logistic → utility-based logistic regressor policy:
             due regressori p_help(I_1) e p_hurt(I_1), combinati come
             S(q) = p_help(q) - lambda * p_hurt(q), rerank se S(q) > tau.
             lambda e tau sono scelti su validation via grid search.

Entrambe le policy sono funzione del solo I_1(q): l'output e' sempre una
soglia intera N* su num_inliers, salvata in extension/thresholds_computed.json
({"type": "threshold", "value": N*}) e usata automaticamente da
match_queries_preds.py — nessuna feature aggiuntiva, nessuna modifica
necessaria all'inference.

Uso:
    # metodo local — Luca
    python extension/helps_estimator.py \\
        --train-csv train.csv --val-csv val.csv \\
        --vpr-method megaloc --matcher superpoint-lg

    # metodo logistic (cost-sensitive) — utility-based logistic regressor
    python extension/helps_estimator.py \\
        --method logistic \\
        --train-csv train.csv --val-csv val.csv \\
        --vpr-method megaloc --matcher superpoint-lg

I CSV in input sono candidate-level (una riga per coppia query-candidato):
    query_id, candidate_path, l2_distance, retrieval_rank,
    num_inliers, rerank_rank_topK, is_positive, K
csv_utils.load_query_level() li converte automaticamente in query-level.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from csv_utils import load_query_level

_JSON_PATH = Path(__file__).parent / "thresholds_computed.json"
# local (Luca) → metodo4 | logistic (utility-based, p_help - lambda*p_hurt) → metodo5
_THRESHOLD_TYPE_BY_METHOD = {"local": "metodo4", "logistic": "metodo5"}


# ============================================================
# UTILS COMUNI
# ============================================================

def compute_metrics(rerank_mask, correct_0, correct_20, K=20):
    """R@1 adattivo, match medi per query e risparmio vs full reranking."""
    adaptive_correct = np.where(rerank_mask, correct_20, correct_0)
    rerank_rate = rerank_mask.mean()
    avg_matches = 1 + (K - 1) * rerank_rate
    return {
        "adaptive_R@1": adaptive_correct.mean(),
        "avg_matches":  avg_matches,
        "savings_%":    100 * (1 - avg_matches / K),
    }


# ============================================================
# METODO LOCAL — stima non parametrica di P(helps | I_1(q))
# ============================================================

def estimate_p_help_local(values, train_x, train_y, initial_window=1, min_k=30):
    """
    Per ogni valore N di num_inliers, stima P(helps) come media locale sul train.
    Espande la finestra finche' non trova almeno min_k campioni.
    """
    max_x = train_x.max()
    p_help_list = []
    for N in tqdm(values, desc=f"Stima P(helps) locale (min_k={min_k})", leave=False):
        N = float(N)
        window = float(initial_window)
        while True:
            mask = (train_x >= N - window) & (train_x <= N + window)
            local_y = train_y[mask]
            if len(local_y) >= min_k or window >= max_x:
                break
            window += 1.0
        p_help_list.append(0.0 if len(local_y) == 0 else float(local_y.mean()))
    return np.array(p_help_list)


def find_N_star_local(train_x, train_y, best_min_k, best_tau, initial_window=1):
    """
    N* = ultimo intero per cui P(helps|N) > best_tau.
    Sotto N*: P(helps) alta → re-rank. Sopra N*: P(helps) bassa → salta.
    """
    x_range = np.arange(int(train_x.min()), int(train_x.max()) + 1)
    p_help = estimate_p_help_local(
        x_range.astype(float), train_x, train_y, initial_window, best_min_k
    )
    above = x_range[p_help > best_tau]
    return int(above.max()) if len(above) > 0 else int(train_x.min())


def run_local(train_df, val_df, args):
    train_x    = train_df["num_inliers_top1"].values
    train_y    = train_df["helps_20"].values
    val_x      = val_df["num_inliers_top1"].values
    correct_0  = val_df["correct_0"].values
    correct_20 = val_df["correct_20"].values
    tau_values = np.round(np.arange(0.05, 1.0, args.tau_step), 3)

    # Grid search su (min_k, tau): trova il miglior R@1 adattivo
    best_r1, best_record = -1.0, None
    for min_k in tqdm(args.min_k_values, desc="Grid search min_k"):
        p_help = estimate_p_help_local(val_x, train_x, train_y, args.initial_window, min_k)
        for tau in tau_values:
            m = compute_metrics(p_help > tau, correct_0, correct_20)
            if m["adaptive_R@1"] > best_r1:
                best_r1     = m["adaptive_R@1"]
                best_record = {"min_k": min_k, "tau": tau}

    # Policy piu' efficiente entro max_drop_pp dal best R@1
    target_r1, best_efficient, best_avg = best_r1 - args.max_drop_pp / 100.0, None, float("inf")
    for min_k in args.min_k_values:
        p_help = estimate_p_help_local(val_x, train_x, train_y, args.initial_window, min_k)
        for tau in tau_values:
            m = compute_metrics(p_help > tau, correct_0, correct_20)
            if m["adaptive_R@1"] >= target_r1 and m["avg_matches"] < best_avg:
                best_avg       = m["avg_matches"]
                best_efficient = {"min_k": min_k, "tau": tau, "metrics": m}

    chosen = best_efficient if best_efficient else {
        **best_record,
        "metrics": compute_metrics(
            estimate_p_help_local(val_x, train_x, train_y,
                                   args.initial_window, best_record["min_k"]) > best_record["tau"],
            correct_0, correct_20
        )
    }
    N_star = find_N_star_local(train_x, train_y, chosen["min_k"], chosen["tau"], args.initial_window)

    return {"type": "threshold", "value": N_star}, chosen["metrics"]


# ============================================================
# METODO LOGISTIC — utility-based logistic regressor policy
#
# y_help(q) = 1{c_0(q)=0 AND c_20(q)=1}   (il rerank ha corretto)
# y_hurt(q) = 1{c_0(q)=1 AND c_20(q)=0}   (il rerank ha rotto)
#
# p_help(q) = P(y_help=1 | I_1(q))   p_hurt(q) = P(y_hurt=1 | I_1(q))
# S(q) = p_help(q) - lambda * p_hurt(q)
# rerank se S(q) > tau
# ============================================================

def fit_logistic(X, y):
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)),
    ])
    clf.fit(X, y)
    return clf


def find_N_star_cost_sensitive(train_x, clf_help, clf_hurt, best_lambda, best_tau):
    """
    N* = ultimo intero per cui S(N) = p_help(N) - lambda*p_hurt(N) > best_tau.
    Scan numerico (non c'e' inversione analitica: S e' la differenza di due
    sigmoidi distinte, non garantita monotona).
    """
    x_range = np.arange(int(train_x.min()), int(train_x.max()) + 1).astype(float).reshape(-1, 1)
    p_help  = clf_help.predict_proba(x_range)[:, 1]
    p_hurt  = clf_hurt.predict_proba(x_range)[:, 1]
    S       = p_help - best_lambda * p_hurt
    above   = x_range.flatten()[S > best_tau]
    return int(above.max()) if len(above) > 0 else int(train_x.min())


def run_logistic(train_df, val_df, args):
    X_train = train_df[["num_inliers_top1"]].values
    y_help  = train_df["helps_20"].values
    y_hurt  = train_df["hurts_20"].values

    X_val      = val_df[["num_inliers_top1"]].values
    correct_0  = val_df["correct_0"].values
    correct_20 = val_df["correct_20"].values

    clf_help = fit_logistic(X_train, y_help)
    clf_hurt = fit_logistic(X_train, y_hurt)

    p_help_val = clf_help.predict_proba(X_val)[:, 1]
    p_hurt_val = clf_hurt.predict_proba(X_val)[:, 1]

    lambda_values = np.round(np.arange(0.0, args.lambda_max + args.lambda_step, args.lambda_step), 3)
    tau_values    = np.round(np.arange(args.tau_min, args.tau_max + args.tau_step, args.tau_step), 3)

    # Grid search su (lambda, tau): max R@1 adattivo, a parita' preferisci piu' saving
    best_r1, best_record = -1.0, None
    for lam in tqdm(lambda_values, desc="Grid search lambda"):
        S_val = p_help_val - lam * p_hurt_val
        for tau in tau_values:
            m = compute_metrics(S_val > tau, correct_0, correct_20)
            better_r1  = m["adaptive_R@1"] > best_r1
            same_r1_more_saving = (
                best_record is not None
                and m["adaptive_R@1"] == best_r1
                and m["avg_matches"] < best_record["metrics"]["avg_matches"]
            )
            if better_r1 or same_r1_more_saving:
                best_r1     = m["adaptive_R@1"]
                best_record = {"lam": lam, "tau": tau, "metrics": m}

    train_x = train_df["num_inliers_top1"].values
    N_star  = find_N_star_cost_sensitive(train_x, clf_help, clf_hurt, best_record["lam"], best_record["tau"])

    print(f"  lambda* = {best_record['lam']}  tau* = {best_record['tau']}")
    return {"type": "threshold", "value": N_star}, best_record["metrics"]


# ============================================================
# ARGOMENTI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Stima threshold P(helps) — local/logistic")
    parser.add_argument("--train-csv",      required=True)
    parser.add_argument("--val-csv",        required=True)
    parser.add_argument("--vpr-method",     required=True,
                        help="metodo VPR usato (es. megaloc, cosplace)")
    parser.add_argument("--matcher",        required=True,
                        help="matcher usato (es. superpoint-lg, loftr)")
    parser.add_argument("--method",         default="local",
                        choices=["local", "logistic"],
                        help="local = non parametrico | logistic = utility-based (p_help - lambda*p_hurt)")
    # Iperparametri metodo local
    parser.add_argument("--min-k-values",   nargs="+", type=int,
                        default=[10, 20, 30, 50, 75, 100])
    parser.add_argument("--initial-window", type=int, default=1)
    # Iperparametri metodo logistic
    parser.add_argument("--lambda-max",     type=float, default=3.0)
    parser.add_argument("--lambda-step",    type=float, default=0.1)
    parser.add_argument("--tau-min",        type=float, default=-1.0)
    parser.add_argument("--tau-max",        type=float, default=1.0)
    # Iperparametri comuni
    parser.add_argument("--tau-step",       type=float, default=0.05)
    parser.add_argument("--max-drop-pp",    type=float, default=0.1)
    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main(args):
    train_df = load_query_level(args.train_csv)
    val_df   = load_query_level(args.val_csv)

    print(f"Train: {len(train_df)} query | Val: {len(val_df)} query")
    print(f"Retrieval R@1 (val):   {val_df['correct_0'].mean():.4f}")
    print(f"Full rerank R@1 (val): {val_df['correct_20'].mean():.4f}")

    if args.method == "local":
        output_info, m = run_local(train_df, val_df, args)
    else:
        output_info, m = run_logistic(train_df, val_df, args)

    threshold_type = _THRESHOLD_TYPE_BY_METHOD[args.method]
    print("\n" + "=" * 60)
    print(f"RISULTATO — {threshold_type} [{args.method}] | {args.vpr_method} | {args.matcher}")
    print(f"  Adaptive R@1 = {m['adaptive_R@1']:.4f}")
    print(f"  Savings      = {m['savings_%']:.1f}%")
    print(f"  → N* (num_inliers threshold) = {output_info['value']}")
    print("=" * 60)

    save_to_json(threshold_type, args.vpr_method, args.matcher, output_info, _JSON_PATH)
    print("match_queries_preds.py userà automaticamente questo valore.")


def save_to_json(threshold_type, vpr_method, matcher, output_info, json_path):
    """Salva nel JSON condiviso con match_queries_preds.py. Scrittura atomica via .tmp."""
    json_path = Path(json_path)
    data = {}
    if json_path.exists():
        with open(json_path) as f:
            content = f.read().strip()
            if content:
                data = json.loads(content)

    data.setdefault(threshold_type, {}).setdefault(vpr_method, {})[matcher] = output_info

    tmp_path = json_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(json_path)


if __name__ == "__main__":
    main(parse_args())
