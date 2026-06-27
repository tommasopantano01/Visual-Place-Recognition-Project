"""
helps_estimator.py — Stima la threshold per il re-ranking adattivo (metodo4).

Supporta due metodi:
  local    → stima non parametrica di P(helps | num_inliers_top1)
  logistic → regressione logistica P(helps | features)

Feature disponibili per il metodo logistic:
  inliers        → solo num_inliers_top1 (segnale geometrico IM)
  SU             → similarity score top-1 = 1/distanza_L2 retrieval
  inliers+SU     → entrambi combinati

Output salvato in extension/thresholds_computed.json:
  - features=inliers (local o logistic): {"type": "threshold", "value": N*}
    → match_queries_preds.py confronta num_inliers con N*, nessuna modifica.
  - features=SU o inliers+SU: {"type": "logistic", "features": [...], params...}
    → match_queries_preds.py applica il modello a runtime con --su-csv.

Uso:
    # metodo local — Luca (default)
    python extension/helps_estimator.py \\
        --train-csv train.csv --val-csv val.csv \\
        --vpr-method megaloc --matcher superpoint-lg

    # metodo logistic con SU — Rocco
    python extension/helps_estimator.py \\
        --method logistic --features SU \\
        --train-csv train.csv --val-csv val.csv \\
        --vpr-method megaloc --matcher superpoint-lg

Colonne richieste nel train CSV: num_inliers_top1, helps_20, correct_0, correct_20
Colonne richieste nel val CSV:   num_inliers_top1, correct_0, correct_20
Colonna aggiuntiva per SU:       SU (in entrambi i CSV)
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.special import logit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

_JSON_PATH = Path(__file__).parent / "thresholds_computed.json"
_THRESHOLD_TYPE = "metodo4"


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
# METODO LOCAL (Luca) — stima non parametrica di P(helps)
# ============================================================

def estimate_p_help_local(values, train_x, train_y, initial_window=1, min_k=30):
    """
    Per ogni valore N di num_inliers, stima P(helps) come media locale sul train.
    Espande la finestra finché non trova almeno min_k campioni.
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

    # Policy più efficiente entro max_drop_pp dal best R@1
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

    # Output: soglia semplice su num_inliers
    return {"type": "threshold", "value": N_star}, chosen["metrics"]


# ============================================================
# METODO LOGISTIC (Rocco) — regressione logistica P(helps)
# ============================================================

def fit_logistic(X, y):
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)),
    ])
    clf.fit(X, y)
    return clf


def run_logistic(train_df, val_df, args):
    # Mappa feature name → colonna CSV
    feat_map = {
        "inliers":    ["num_inliers_top1"],
        "SU":         ["SU"],
        "inliers+SU": ["num_inliers_top1", "SU"],
    }
    feat_cols = feat_map[args.features]

    # Verifica che le colonne esistano
    for col in feat_cols:
        for df, name in [(train_df, "train"), (val_df, "val")]:
            if col not in df.columns:
                raise ValueError(f"Colonna '{col}' mancante nel {name} CSV.")

    X_train    = train_df[feat_cols].values
    X_val      = val_df[feat_cols].values
    y_train    = train_df["helps_20"].values   # 1 se re-ranking ha aiutato
    correct_0  = val_df["correct_0"].values
    correct_20 = val_df["correct_20"].values

    # Training
    clf = fit_logistic(X_train, y_train)
    p_help_val = clf.predict_proba(X_val)[:, 1]

    # Grid search su tau
    tau_values = np.round(np.arange(0.0, 1.0 + args.tau_step, args.tau_step), 4)
    best_r1, best_tau_r1 = -1.0, None
    for tau in tau_values:
        m = compute_metrics(p_help_val > tau, correct_0, correct_20)
        if m["adaptive_R@1"] > best_r1:
            best_r1, best_tau_r1 = m["adaptive_R@1"], tau

    target_r1, best_tau_eff, best_avg = best_r1 - args.max_drop_pp / 100.0, None, float("inf")
    for tau in tau_values:
        m = compute_metrics(p_help_val > tau, correct_0, correct_20)
        if m["adaptive_R@1"] >= target_r1 and m["avg_matches"] < best_avg:
            best_avg, best_tau_eff = m["avg_matches"], tau

    BEST_TAU = best_tau_eff if best_tau_eff is not None else best_tau_r1
    m_final  = compute_metrics(p_help_val > BEST_TAU, correct_0, correct_20)

    logreg = clf.named_steps["logreg"]
    scaler = clf.named_steps["scaler"]

    if args.features == "inliers":
        # Caso inliers only: N* ricavato analiticamente invertendo la sigmoide
        w, b   = logreg.coef_[0][0], logreg.intercept_[0]
        mean, std = scaler.mean_[0], scaler.scale_[0]
        N_star = int(round((logit(BEST_TAU) - b) / w * std + mean))
        output = {"type": "threshold", "value": N_star}
    else:
        # Caso SU o inliers+SU: salva i parametri del modello.
        # match_queries_preds.py applicherà il modello a runtime usando il CSV con SU.
        output = {
            "type":         "logistic",
            "features":     feat_cols,          # lista colonne usate come input
            "scaler_mean":  scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "coef":         logreg.coef_.tolist(),
            "intercept":    float(logreg.intercept_[0]),
            "tau":          float(BEST_TAU),    # re-rank se P(helps) > tau
        }

    return output, m_final


# ============================================================
# SALVATAGGIO ATOMICO IN thresholds_computed.json
# ============================================================

def save_to_json(threshold_type, vpr_method, matcher, output_info, json_path):
    """
    Salva il risultato nel JSON condiviso con match_queries_preds.py.
    output_info può essere:
      {"type": "threshold", "value": N_star}
      {"type": "logistic", "features": [...], "scaler_mean": [...], ...}
    Scrittura atomica via .tmp per evitare JSON corrotto in caso di crash.
    """
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


# ============================================================
# ARGOMENTI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Stima threshold P(helps) — metodo4")
    parser.add_argument("--train-csv",      required=True)
    parser.add_argument("--val-csv",        required=True)
    parser.add_argument("--vpr-method",     required=True,
                        help="metodo VPR usato (es. megaloc, cosplace)")
    parser.add_argument("--matcher",        required=True,
                        help="matcher usato (es. superpoint-lg, loftr)")
    parser.add_argument("--method",         default="local",
                        choices=["local", "logistic"],
                        help="local = non parametrico (Luca) | logistic = regressione (Rocco)")
    parser.add_argument("--features",       default="inliers",
                        choices=["inliers", "SU", "inliers+SU"],
                        help="feature per il metodo logistic (ignorato se --method local)")
    # Iperparametri metodo local
    parser.add_argument("--min-k-values",   nargs="+", type=int,
                        default=[10, 20, 30, 50, 75, 100])
    parser.add_argument("--initial-window", type=int, default=1)
    # Iperparametri comuni
    parser.add_argument("--tau-step",       type=float, default=0.05)
    parser.add_argument("--max-drop-pp",    type=float, default=0.1)
    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main(args):
    required_train = ["num_inliers_top1", "helps_20", "correct_0", "correct_20"]
    required_val   = ["num_inliers_top1", "correct_0", "correct_20"]

    train_df = pd.read_csv(args.train_csv).dropna(subset=required_train)
    val_df   = pd.read_csv(args.val_csv).dropna(subset=required_val)

    for df in [train_df, val_df]:
        df["num_inliers_top1"] = df["num_inliers_top1"].astype(float)
        df["correct_0"]        = df["correct_0"].astype(int)
        df["correct_20"]       = df["correct_20"].astype(int)
    train_df["helps_20"] = train_df["helps_20"].astype(int)

    print(f"Train: {len(train_df)} query | Val: {len(val_df)} query")
    print(f"Retrieval R@1 (val):   {val_df['correct_0'].mean():.4f}")
    print(f"Full rerank R@1 (val): {val_df['correct_20'].mean():.4f}")

    if args.method == "local":
        output_info, m = run_local(train_df, val_df, args)
    else:
        output_info, m = run_logistic(train_df, val_df, args)

    label = args.method if args.method == "local" else f"logistic/{args.features}"
    print("\n" + "=" * 60)
    print(f"RISULTATO — metodo4 [{label}] | {args.vpr_method} | {args.matcher}")
    print(f"  Adaptive R@1 = {m['adaptive_R@1']:.4f}")
    print(f"  Savings      = {m['savings_%']:.1f}%")
    if output_info["type"] == "threshold":
        print(f"  → N* (num_inliers threshold) = {output_info['value']}")
    else:
        print(f"  → modello logistico salvato (features: {output_info['features']}, tau: {output_info['tau']:.2f})")
    print("=" * 60)

    save_to_json(_THRESHOLD_TYPE, args.vpr_method, args.matcher, output_info, _JSON_PATH)
    print("match_queries_preds.py userà automaticamente questo valore.")


if __name__ == "__main__":
    main(parse_args())
