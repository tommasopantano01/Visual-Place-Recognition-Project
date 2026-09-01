"""
_common.py — Shared engine for all adaptive reranking methods.

OUTPUT CONVENTION: one folder per budget inside --output-dir.
  output-dir/top1/<id>.torch    queries stopped at top-1 (skip rerank)
  output-dir/top5/<id>.torch    sequential intermediate stop (5 candidates)
  output-dir/top10/<id>.torch   sequential intermediate stop (10 candidates)
  output-dir/top20/<id>.torch   full rerank (20 candidates)

Each query ends up in EXACTLY one topK folder, where K = how many candidates
were passed to image matching for that query. The .torch file contains K real
IM results. check_performance.py counts the files in each topK to work out
how many queries stopped at each budget.

Exception: methods that decide to skip WITHOUT doing any IM (only `su`, which
decides from the retrieval L2 distances alone) save the retrieval .txt in
top0/ — budget 0 because for those queries NO image matching was done.
check_performance.py handles both cases: .torch -> ranking by inliers, .txt ->
retrieval order.

TWO EXECUTION MODES
  LIVE     actually runs the matcher (GPU + image-matching-models + images).
  OFFLINE  --inliers-dir <folder of already computed top-20 .torch files>: no
           matching, the first K candidates are read from the existing files.
"""

import os
import sys
import csv
import json
import shutil
from glob import glob
from pathlib import Path
from copy import deepcopy

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_REPO_ROOT))
sys.path.append(str(_REPO_ROOT / "image-matching-models"))

try:
    import torch
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

    class _TorchStub:
        def __getattr__(self, name):
            raise ImportError(
                "torch not available: needed for .torch I/O and image matching."
            )
    torch = _TorchStub()

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *a, **k):   # iterable no-op if tqdm is missing
        return x

try:
    from util import read_file_preds
    from matching import get_matcher
    _MATCHER_OK = True
except ImportError as _e:
    _MATCHER_IMPORT_ERROR = _e
    _MATCHER_OK = False

    def _missing_matcher(*_a, **_k):
        raise ImportError(
            "This function requires image-matching-models (util, matching), "
            f"not available: {_MATCHER_IMPORT_ERROR}. Training/validation/SU "
            "decision do not need it (real matching goes through match_queries_preds.py)."
        )
    get_matcher = _missing_matcher
    read_file_preds = _missing_matcher

# compatibility: some functions might check _IM_DEPS_OK
_IM_DEPS_OK = _TORCH_OK and _MATCHER_OK


# ============================================================
# QUERY AND PARAMETER READING
# ============================================================

def get_query_ids(preds_dir):
    """List of query ids (file names without extension) in preds_dir, sorted."""
    txt_files = glob(os.path.join(preds_dir, "*.txt"))
    return sorted((Path(f).stem for f in txt_files), key=int)


def load_threshold_csv(path):
    """Reads threshold_<model>_<matcher>.csv in two formats:
      1) Flat CSV  -> one row, numeric columns (e.g. youden: 'threshold').
      2) Nested JSON (validation SU/logistic) -> extracts the parameters of
         the calibrated criterion (tau, and possibly alpha) and flattens them.
    Always returns a dict {name: float}.
    """
    with open(path) as f:
        head = f.read(1).lstrip()
        f.seek(0)

        # --- JSON format (validation SU/logistic) ---
        if head == "{":
            data = json.load(f)
            fs = data["feature_sets"]
            fs_name = next(iter(fs))
            crit = fs[fs_name]["criteria"]
            if not crit:
                raise ValueError(
                    f"{path}: no calibrated criterion in the threshold JSON "
                    "(validation skipped every criterion)."
                )
            crit_name = next(iter(crit))
            params = crit[crit_name]
            out = {}
            if "tau" in params:
                out["tau"] = float(params["tau"])
            if "alpha" in params:
                out["alpha"] = float(params["alpha"])
            if "threshold" in params:
                out["threshold"] = float(params["threshold"])
            if not out:
                raise ValueError(
                    f"{path}: criterion '{crit_name}' has no recognised numeric "
                    f"parameters ({list(params.keys())})."
                )
            return out

        # --- flat CSV format (one row) ---
        row = next(csv.DictReader(f))
        return {k: float(v) for k, v in row.items()}


def load_model_json(path):
    """Reads model.json (coefficients of one or more logistic regressors)."""
    with open(path) as f:
        return json.load(f)


def budget_folder(output_dir, budget):
    """Returns output_dir/top{budget}/ (creating it if missing). budget = the
    number of candidates actually matched for those queries (1/5/10/20)."""
    d = os.path.join(output_dir, f"top{budget}")
    os.makedirs(d, exist_ok=True)
    return d


# ============================================================
# IM BACKEND — two modes
#
#   LIVE    (default): actually runs the matcher. Needs GPU +
#                      image-matching-models + the dataset images.
#   OFFLINE (--inliers-dir): does NOT run anything, reads the already
#                      computed top-20 .torch files and takes the first K.
#                      Identical numbers, seconds instead of hours, runs on CPU.
#
# In OFFLINE the saved results are "slim" ({'num_inliers': int}): it is the
# only field check_performance.py / reranking.py read, and it keeps the
# output small (the full .torch files contain keypoints and descriptors).
# ============================================================

def slim_result(result):
    """Reduces an IM result to just the field used by the reranking."""
    return {"num_inliers": int(result["num_inliers"])}


def load_precomputed_results(inliers_dir, query_id, k=None):
    """Reads inliers_dir/<query_id>.torch (already computed top-20 IM) and
    returns the slim list of the first k results. None if the file does not
    exist."""
    fp = os.path.join(str(inliers_dir), f"{query_id}.torch")
    if not os.path.exists(fp):
        return None
    data = torch.load(fp, map_location="cpu", weights_only=False)
    results = [slim_result(r) for r in data]
    return results if k is None else results[:k]


def warn_missing(missing_ids, inliers_dir, what="query"):
    """A single readable warning if precomputed .torch files are missing."""
    if not missing_ids:
        return
    sample = ", ".join(missing_ids[:5]) + (" ..." if len(missing_ids) > 5 else "")
    print(f"  [warning] {len(missing_ids)} {what} without .torch in {inliers_dir} "
          f"(excluded): {sample}")


# ============================================================
# IMAGE MATCHING
# ============================================================

def run_im_top1_all(preds_dir, matcher_name, device="cpu", im_size=512):
    """IM between query and top-1 candidate for EVERY query. Returns {id: num_inliers}."""
    matcher = get_matcher(matcher_name, device=device)
    inliers_by_query = {}
    for q_id in tqdm(get_query_ids(preds_dir), desc="IM top-1 on all queries"):
        txt_file = os.path.join(preds_dir, f"{q_id}.txt")
        q_path, pred_paths = read_file_preds(txt_file)
        img0 = matcher.load_image(q_path, resize=im_size)
        img1 = matcher.load_image(pred_paths[0], resize=im_size)
        inliers_by_query[q_id] = matcher(deepcopy(img0), img1)["num_inliers"]
    return inliers_by_query


def run_im_top1_with_results(preds_dir, matcher_name, device="cpu", im_size=512,
                              query_ids=None, inliers_dir=None):
    """Like run_im_top1_all, but returns the full IM result (not just
    num_inliers): needed both to save the .torch of skipped queries and for
    the sequential method to accumulate. If query_ids is given, only that
    subset. Returns {id: result_dict}.

    inliers_dir: OFFLINE mode, reads the top-1 from the precomputed .torch files."""
    if query_ids is None:
        query_ids = get_query_ids(preds_dir)
    if not query_ids:
        return {}

    if inliers_dir is not None:
        results_by_query, missing = {}, []
        for q_id in query_ids:
            r = load_precomputed_results(inliers_dir, q_id, k=1)
            if not r:
                missing.append(q_id)
                continue
            results_by_query[q_id] = r[0]
        warn_missing(missing, inliers_dir)
        print(f"  IM top-1 read from precomputed .torch: {len(results_by_query)} queries")
        return results_by_query

    matcher = get_matcher(matcher_name, device=device)
    results_by_query = {}
    for q_id in tqdm(query_ids, desc="IM top-1 on all queries"):
        txt_file = os.path.join(preds_dir, f"{q_id}.txt")
        q_path, pred_paths = read_file_preds(txt_file)
        img0 = matcher.load_image(q_path, resize=im_size)
        img1 = matcher.load_image(pred_paths[0], resize=im_size)
        result = matcher(deepcopy(img0), img1)
        result["all_desc0"] = result["all_desc1"] = None
        results_by_query[q_id] = result
    return results_by_query


def run_im_extend(preds_dir, query_ids, accumulated_results, start_rank, end_rank,
                   matcher_name, device="cpu", im_size=512, inliers_dir=None):
    """Extends accumulated_results (id -> list of IM results already done) by
    matching candidates from start_rank to end_rank (1-indexed, inclusive)
    ONLY for the queries in query_ids. Modifies accumulated_results in place.

    Returns the list of ids that could NOT be extended (offline mode only:
    missing .torch or with fewer than end_rank candidates). The caller must
    finalize them at the budget already reached."""
    if not query_ids:
        return []

    if inliers_dir is not None:
        failed = []
        for q_id in query_ids:
            res = load_precomputed_results(inliers_dir, q_id, k=end_rank)
            if res is None or len(res) < end_rank:
                failed.append(q_id)
                continue
            accumulated_results[q_id].extend(res[start_rank - 1:end_rank])
        if failed:
            warn_missing(failed, inliers_dir,
                         what=f"queries without candidates up to rank {end_rank}")
        return failed

    matcher = get_matcher(matcher_name, device=device)
    for q_id in tqdm(query_ids, desc=f"IM candidates {start_rank}-{end_rank}"):
        txt_file = os.path.join(preds_dir, f"{q_id}.txt")
        q_path, pred_paths = read_file_preds(txt_file)
        img0 = matcher.load_image(q_path, resize=im_size)
        for rank in range(start_rank, end_rank + 1):
            img1 = matcher.load_image(pred_paths[rank - 1], resize=im_size)
            result = matcher(deepcopy(img0), img1)
            result["all_desc0"] = result["all_desc1"] = None
            accumulated_results[q_id].append(result)
    return []


# ============================================================
# OUTPUT
# ============================================================

def save_results_torch(query_id, results, folder):
    """Saves the list of real IM results (len = budget) as <id>.torch in
    folder."""
    torch.save(list(results), os.path.join(folder, f"{query_id}.torch"))


def save_skipped_as_txt(query_ids, preds_dir, folder):
    """Copies the retrieval .txt of the queries skipped WITHOUT image matching
    (e.g. su/): there are no inliers to invent, the retrieval remains valid."""
    if not query_ids:
        return
    os.makedirs(folder, exist_ok=True)
    for q_id in query_ids:
        shutil.copy2(os.path.join(preds_dir, f"{q_id}.txt"),
                     os.path.join(folder, f"{q_id}.txt"))


def run_im_topN_subset(preds_dir, query_ids, output_dir, budget, matcher_name,
                        device="cpu", im_size=512, inliers_dir=None):
    """IM on the first `budget` candidates for each query in query_ids; saves a
    .torch (len = budget) in output_dir/top{budget}/.

    inliers_dir: OFFLINE mode, trims the precomputed .torch files to the first
    `budget` candidates instead of redoing the matching."""
    if not query_ids:
        return
    folder = budget_folder(output_dir, budget)

    if inliers_dir is not None:
        missing = []
        for q_id in tqdm(query_ids, desc=f"top-{budget} from precomputed .torch"):
            res = load_precomputed_results(inliers_dir, q_id, k=budget)
            if not res:
                missing.append(q_id)
                continue
            save_results_torch(q_id, res, folder)
        warn_missing(missing, inliers_dir)
        return

    matcher = get_matcher(matcher_name, device=device)
    for q_id in tqdm(query_ids, desc=f"IM top-{budget} on the selected queries"):
        out_file = os.path.join(folder, f"{q_id}.torch")
        if os.path.exists(out_file):
            continue
        txt_file = os.path.join(preds_dir, f"{q_id}.txt")
        q_path, pred_paths = read_file_preds(txt_file)
        img0 = matcher.load_image(q_path, resize=im_size)
        results = []
        for pred_path in pred_paths[:budget]:
            img1 = matcher.load_image(pred_path, resize=im_size)
            result = matcher(deepcopy(img0), img1)
            result["all_desc0"] = result["all_desc1"] = None
            results.append(result)
        save_results_torch(q_id, results, folder)


# ============================================================
# RESUME — a query is "done" if already finalized in a topK
# ============================================================

def query_already_done(output_dir, query_id, budgets):
    """True if the query is already saved (.torch or .txt) in one of the topK
    folders from a previous run. Used by the sequential method to resume
    per-query."""
    for b in budgets:
        folder = os.path.join(output_dir, f"top{b}")
        if (os.path.exists(os.path.join(folder, f"{query_id}.torch")) or
                os.path.exists(os.path.join(folder, f"{query_id}.txt"))):
            return True
    return False


# ============================================================
# PARTITIONING
#
# LOW num_inliers = uncertain query -> needs rerank (num_inliers < threshold).
# For the regressors, probability models P(needs rerank), so
# probability > tau -> rerank (never the other way around).
# ============================================================

def partition_by_threshold(num_inliers_by_query, threshold):
    """num_inliers < threshold -> needs rerank. Otherwise skip."""
    rerank_ids = [q for q, n in num_inliers_by_query.items() if n < threshold]
    skip_ids   = [q for q, n in num_inliers_by_query.items() if n >= threshold]
    return rerank_ids, skip_ids


def partition_by_probability(prob_by_query, tau):
    """probability > tau -> needs rerank. Otherwise skip."""
    rerank_ids = [q for q, p in prob_by_query.items() if p > tau]
    skip_ids   = [q for q, p in prob_by_query.items() if p <= tau]
    return rerank_ids, skip_ids


# ============================================================
# LOGISTIC REGRESSION (application, not training)
# ============================================================

def apply_sigmoid(signals_by_query, model_json):
    """Applies scaler + sigmoid to {id: {feature: value, ...}}. Returns {id: prob}.

    Accepts model_json in two formats:
      - FLAT:   {feat_cols, scaler_mean, scaler_scale, coef, intercept, ...}
      - NESTED: {feature_sets: {<fs>: {regressors: {<target>: <flat>}}}}
    For single-feature regressors it does not depend on the feature NAME (the
    runtime signal uses 'inliers', the JSON may have 'feature_0'/'num_inliers_top1')."""
    reg = _extract_flat_regressor(model_json)
    feat_cols = reg["feat_cols"]
    mean  = np.array(reg["scaler_mean"])
    scale = np.array(reg["scaler_scale"])
    w     = np.array(reg["coef"][0])
    b     = reg["intercept"][0]

    prob_by_query = {}
    for q_id, sig in signals_by_query.items():
        if len(feat_cols) == 1:
            x = np.array([next(iter(sig.values()))], dtype=float)
        else:
            x = np.array([sig[c] for c in feat_cols], dtype=float)
        z = (x - mean) / scale
        logit_val = float(np.dot(w, z) + b)
        prob_by_query[q_id] = 1.0 / (1.0 + np.exp(-logit_val))
    return prob_by_query


def _extract_flat_regressor(model_json):
    """Returns the FLAT regressor dict (feat_cols/scaler_*/coef/intercept) from
    a model_json that may already be flat or nested
    (feature_sets -> <fs> -> regressors -> <target>). If nested with more than
    one regressor it is an error: the deploy would not know which one to use."""
    if "coef" in model_json and "intercept" in model_json:
        return model_json                       # already flat
    if "feature_sets" in model_json:
        fs = model_json["feature_sets"]
        regressors = fs[next(iter(fs))]["regressors"]
        if len(regressors) != 1:
            raise ValueError(
                f"model.json nested with more than one regressor {list(regressors)}: "
                "the deploy does not know which one to use (a single target is needed)."
            )
        return next(iter(regressors.values()))
    raise ValueError(
        f"Unrecognised model.json format (keys={list(model_json.keys())})."
    )


# ============================================================
# Score Uncertainty (SU) SIGNAL — only for su/ and su_inliers/
# Computed from the retrieval L2 distances, no image matching.
# ============================================================

def load_z_data_distances(z_data_path):
    """Loads z_data.torch (--save_for_uncertainty) and returns {id: array_L2},
    id = string index (same convention as preds/<idx>.txt)."""
    data = torch.load(z_data_path, map_location="cpu", weights_only=False)
    distances = np.asarray(data["distances"], dtype=float)  # (N, K)
    return {str(i): distances[i] for i in range(distances.shape[0])}


def l2_to_su(l2_distances, k=10, alpha=0.5, eps=1e-12):
    """s_i = 1/(1+L2_i); RS = mean(s_1..k / s_0); SD = median(s)/max;
    SU = alpha*RS + (1-alpha)*SD."""
    d = np.asarray(l2_distances, dtype=float)[:k]
    s = 1.0 / (1.0 + d)
    if len(s) < 2:
        return 0.0
    rs = float(np.mean(s[1:] / (s[0] + eps)))
    sd = float(np.median(s) / (np.max(s) + eps))
    return alpha * rs + (1 - alpha) * sd


# ============================================================
# GENERIC RUNNERS
#   scalar  -> youden / best_r1 / efficiency / local
#   logistic single -> logistic_hard / logistic_help
# ============================================================

def run_scalar_method(preds_dir, threshold, matcher_name, device, im_size,
                       num_preds, output_dir, inliers_dir=None):
    # IM only on the top-1 of every query: it is the decision feature
    results_top1 = run_im_top1_with_results(preds_dir, matcher_name, device, im_size,
                                            inliers_dir=inliers_dir)
    inliers = {q: r["num_inliers"] for q, r in results_top1.items()}

    rerank_ids, skip_ids = partition_by_threshold(inliers, threshold)
    print_summary(rerank_ids, skip_ids)

    # skip -> top1/: saves the IM result of the top-1 only (already computed)
    folder1 = budget_folder(output_dir, 1)
    for q in skip_ids:
        save_results_torch(q, [results_top1[q]], folder1)

    # rerank -> top{num_preds}/: full IM
    run_im_topN_subset(preds_dir, rerank_ids, output_dir, num_preds,
                        matcher_name, device, im_size, inliers_dir=inliers_dir)


def run_logistic_single(preds_dir, tau, model, matcher_name, device, im_size,
                         num_preds, output_dir, inliers_dir=None):
    results_top1 = run_im_top1_with_results(preds_dir, matcher_name, device, im_size,
                                            inliers_dir=inliers_dir)
    signals = {q: {"inliers": r["num_inliers"]} for q, r in results_top1.items()}
    probs   = apply_sigmoid(signals, model)

    rerank_ids, skip_ids = partition_by_probability(probs, tau)
    print_summary(rerank_ids, skip_ids)

    folder1 = budget_folder(output_dir, 1)
    for q in skip_ids:
        save_results_torch(q, [results_top1[q]], folder1)

    run_im_topN_subset(preds_dir, rerank_ids, output_dir, num_preds,
                        matcher_name, device, im_size, inliers_dir=inliers_dir)


# ============================================================
# LOG
# ============================================================

def print_summary(rerank_ids, skip_ids):
    total = len(rerank_ids) + len(skip_ids)
    pct = 100 * len(rerank_ids) / total if total else 0.0
    print(f"\nTotal queries: {total}")
    print(f"  Rerank (top-20): {len(rerank_ids):5d}  ({pct:.1f}%)")
    print(f"  Skip (top-1 only): {len(skip_ids):5d}  ({100 - pct:.1f}%)")


# ============================================================
# TRAINING + VALIDATION of the su/ and su_inliers/ methods
# ============================================================

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Columns required in the candidate-level CSV.

BASE_REQUIRED_COLS = ["query_id", "retrieval_rank",
                      "num_inliers", "is_positive", "rerank_rank_topK"]
SU_REQUIRED_COLS   = BASE_REQUIRED_COLS + ["l2_distance"]

SU_TARGETS  = ("hard", "help", "hurts")
SU_CRITERIA = ("P(hard)", "P(help)", "P(help)-aP(hurts)")

def load_query_level(csv_dir_or_file, k=10, alpha=0.5, needs_l2=True):
    """Reads one or more candidate-level CSVs (dir or single file), collapses
    to query level and computes SU (via l2_to_su), inliers (negated) and the
    hard/helps/hurts labels. Used by training (for X,y) and validation (for X
    + correct_*).

    needs_l2: if True (default) the l2_distance column is REQUIRED (needed to
    compute SU: su/ and su_inliers/ methods). If False (methods that do NOT
    use SU, e.g. logistic_help/logistic_hard/youden/...) L2 is not required:
    if missing it is filled with NaN and the SU column comes out NaN (not
    used by those methods).

    Returned columns: query_id_full, source_file, SU, inliers, num_inliers_top1,
      correct_0, correct_full_rerank, hard, helps, hurts
    """
    if os.path.isdir(csv_dir_or_file):
        files = sorted(glob(os.path.join(csv_dir_or_file, "*.csv")))
    else:
        files = [csv_dir_or_file]
    if not files:
        raise FileNotFoundError(f"No CSV found in {csv_dir_or_file}")

    req = SU_REQUIRED_COLS if needs_l2 else BASE_REQUIRED_COLS

    rows = []
    for fp in files:
        stem = os.path.splitext(os.path.basename(fp))[0]
        df = pd.read_csv(fp)
        missing = [c for c in req if c not in df.columns]
        if missing:
            raise ValueError(f"{fp}: missing columns {missing}")

        if "l2_distance" not in df.columns:
            df["l2_distance"] = np.nan

        for qid, g in df.groupby("query_id", sort=False):
            g = g.sort_values("retrieval_rank")
            if len(g) < k:
                continue  # SU not computable

            l2 = g["l2_distance"].to_numpy(dtype=float)
            su = float("nan") if np.isnan(l2).all() else l2_to_su(l2, k=k, alpha=alpha)

            top1 = g.iloc[0]
            rr_win = g.loc[g["rerank_rank_topK"] == 1]
            if len(rr_win) == 0:
                continue
            rr_win = rr_win.iloc[0]

            correct_0           = int(top1["is_positive"])
            correct_full_rerank = int(rr_win["is_positive"])

            rows.append({
                "query_id_full":       f"{stem}::{qid}",
                "source_file":         stem,
                "SU":                  float(su),
                # SU convention: few inliers -> high feature -> uncertain query
                "inliers":             -float(top1["num_inliers"]),
                # raw (not negated): feature for the logistic methods on num_inliers
                "num_inliers_top1":    float(top1["num_inliers"]),
                "correct_0":           correct_0,
                "correct_full_rerank": correct_full_rerank,
                "hard":  int(correct_0 == 0),
                "helps": int((correct_0 == 0) and (correct_full_rerank == 1)),
                "hurts": int((correct_0 == 1) and (correct_full_rerank == 0)),
            })

    if not rows:
        raise ValueError("No valid query extracted from the CSVs (check k and the columns).")
    return pd.DataFrame(rows)


# --- TRAINING of a single regressor --------------------------------------

def fit_regressor(X, y):
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(class_weight="balanced",
                                      random_state=42, max_iter=1000)),
    ])
    clf.fit(X, y)
    return clf


def regressor_to_dict(clf, feat_cols):
    sc = clf.named_steps["scaler"]
    lr = clf.named_steps["logreg"]
    return {
        "feat_cols":    list(feat_cols),
        "scaler_mean":  sc.mean_.tolist(),
        "scaler_scale": sc.scale_.tolist(),
        "coef":         lr.coef_.tolist(),
        "intercept":    lr.intercept_.tolist(),
        "classes":      lr.classes_.tolist(),
    }


# --- Reconstruction + application of the regressor for the grid-search -------------

def regressor_from_dict(d):
    sc = StandardScaler()
    sc.mean_   = np.array(d["scaler_mean"])
    sc.scale_  = np.array(d["scaler_scale"])
    sc.var_    = sc.scale_ ** 2
    sc.n_features_in_ = len(d["feat_cols"])
    lr = LogisticRegression()
    lr.coef_       = np.array(d["coef"])
    lr.intercept_  = np.array(d["intercept"])
    lr.classes_    = np.array(d["classes"])
    return Pipeline([("scaler", sc), ("logreg", lr)])


def predict_proba_pos(clf, X):
    """P(positive class). Explicit name so it is not confused with
    apply_sigmoid (which operates on dicts for the scalar/logistic methods)."""
    return clf.predict_proba(X)[:, 1]


def clean_scores(scores):
    """NaN/inf -> finite values (median / finite max / finite min). For the grid-search."""
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    if np.all(finite):
        return scores
    fv = np.nanmedian(scores[finite]) if np.any(finite) else 0.0
    return np.nan_to_num(
        scores, nan=fv,
        posinf=np.max(scores[finite]) if np.any(finite) else fv,
        neginf=np.min(scores[finite]) if np.any(finite) else fv,
    )


# ============================================================
# SEQUENTIAL PROGRESSIVE FEATURES
# ============================================================

def _progressive_feats_from_results(results_upto_b):
    """results_upto_b: list of IM results of candidates with retrieval_rank<=b,
    in rank order. Returns (max_inliers, second_max, gap, best_rank,
    top1_is_best): sorts by num_inliers desc, tie on retrieval_rank asc."""
    
    items = [(float(r["num_inliers"]), i + 1) for i, r in enumerate(results_upto_b)]
    items.sort(key=lambda t: (-t[0], t[1]))     # num_inliers desc, rank asc
    max_inl, best_rank = items[0]
    second_max = items[1][0] if len(items) >= 2 else 0.0
    gap = max_inl - second_max
    return max_inl, second_max, gap, int(best_rank), int(best_rank == 1)


def sequential_features(accumulated_q, gate):
    """Builds the feature vector for the gate ('gate1'|'gate5'|'gate10') from
    the IM results accumulated for a query.
    accumulated_q: list of IM results (rank 1..N seen so far).
    Returns a list of floats (positional order = feat_cols of the model.json)."""
    num_inliers_top1 = float(accumulated_q[0]["num_inliers"])
    if gate == "gate1":
        return [num_inliers_top1]

    # top5 block features (first 5 candidates seen)
    up5 = accumulated_q[:5]
    max5, second5, gap5, brank5, isbest5 = _progressive_feats_from_results(up5)

    if gate == "gate5":
        # [num_inliers_top1, max_top5, second_max_top5, gap_top5, best_rank_top5, top1_is_best_top5]
        return [num_inliers_top1, max5, second5, gap5, float(brank5), float(isbest5)]

    if gate == "gate10":
        up10 = accumulated_q[:10]
        max10, second10, gap10, brank10, isbest10 = _progressive_feats_from_results(up10)
        # NB asymmetry: for the top5 block there is NO second_max, only the gap.
        # [num_inliers_top1, max_top5, gap_top5, best_rank_top5, top1_is_best_top5,
        #  max_top10, second_max_top10, gap_top10, best_rank_top10, top1_is_best_top10]
        return [num_inliers_top1, max5, gap5, float(brank5), float(isbest5),
                max10, second10, gap10, float(brank10), float(isbest10)]

    raise ValueError(f"unknown gate: {gate}")


def apply_sigmoid_vector(feature_vector, model_json):
    """Like apply_sigmoid but for A SINGLE feature vector already ordered
    positionally (used by the multi-feature sequential method). Returns P(class 1)."""
    reg = _extract_flat_regressor(model_json)
    mean  = np.array(reg["scaler_mean"], dtype=float)
    scale = np.array(reg["scaler_scale"], dtype=float)
    w     = np.array(reg["coef"][0], dtype=float)
    b     = float(reg["intercept"][0])
    x = np.asarray(feature_vector, dtype=float)
    if x.shape[0] != mean.shape[0]:
        raise ValueError(f"Feature mismatch: vector len {x.shape[0]}, "
                         f"the model wants {mean.shape[0]}.")
    z = (x - mean) / scale
    return 1.0 / (1.0 + np.exp(-(float(np.dot(w, z)) + b)))
