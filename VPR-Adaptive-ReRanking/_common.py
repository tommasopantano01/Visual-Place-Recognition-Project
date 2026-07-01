"""
_common.py — Motore condiviso da tutti i metodi di adaptive reranking.

CONVENZIONE DI OUTPUT (nuova): una cartella per budget dentro --output-dir.
  output-dir/top1/<id>.torch    query fermate al top-1 (skip rerank)
  output-dir/top5/<id>.torch    stop intermedio sequenziale (5 candidati)
  output-dir/top10/<id>.torch   stop intermedio sequenziale (10 candidati)
  output-dir/top20/<id>.torch   rerank completo (20 candidati)

Ogni query finisce in ESATTAMENTE una cartella topK, dove K = quanti
candidati sono stati passati all'image matching per quella query. Il .torch
contiene K risultati IM reali (formato per-elemento identico a
match_queries_preds.py). check_performance.py conta i file in ogni topK per
ricavare quante query si sono fermate a ciascun budget.

Eccezione: i metodi che decidono lo skip SENZA fare alcun IM (es. su/) per
le query skippate salvano il .txt di retrieval in top1/ (niente inlier da
riportare). check_performance.py gestisce entrambi: .torch -> ranking per
inlier, .txt -> ordine di retrieval.
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

# Import "pigri" (A2), separati per granularita':
#   - torch, tqdm: servono ai decisori SU e all'I/O .torch. Provati per primi.
#   - util, matching: servono SOLO all'image matching (metodi del prof). Se
#     mancano, solo le funzioni di matching diventano inutilizzabili; training,
#     validation e decisione SU (che usano torch ma non il matcher) funzionano.
try:
    import torch
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

    class _TorchStub:
        def __getattr__(self, name):
            raise ImportError(
                "torch non disponibile: serve per I/O .torch e image matching."
            )
    torch = _TorchStub()

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *a, **k):   # no-op iterabile se tqdm manca
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
            "Questa funzione richiede image-matching-models (util, matching), "
            f"non disponibili: {_MATCHER_IMPORT_ERROR}. Training/validation/decisione "
            "SU non ne hanno bisogno (il matching vero passa per match_queries_preds.py)."
        )
    get_matcher = _missing_matcher
    read_file_preds = _missing_matcher

# compatibilita': alcune funzioni potrebbero controllare _IM_DEPS_OK
_IM_DEPS_OK = _TORCH_OK and _MATCHER_OK


# ============================================================
# LETTURA QUERY E PARAMETRI
# ============================================================

def get_query_ids(preds_dir):
    """Lista degli id query (nomi file senza estensione) in preds_dir, ordinati."""
    txt_files = glob(os.path.join(preds_dir, "*.txt"))
    return sorted((Path(f).stem for f in txt_files), key=int)


def load_threshold_csv(path):
    """Legge threshold_<model>_<matcher>.csv in due formati:
      1) CSV piatto  -> una riga, colonne numeriche (es. youden: 'threshold').
      2) JSON annidato (validation SU/logistic) -> estrae i parametri del
         criterio calibrato (tau, ed eventualmente alpha) e li appiattisce.
    Ritorna sempre un dict {nome: float}.
    """
    with open(path) as f:
        head = f.read(1).lstrip()
        f.seek(0)

        # --- formato JSON (validation SU/logistic) ---
        if head == "{":
            data = json.load(f)
            fs = data["feature_sets"]
            fs_name = next(iter(fs))                     # unico feature set
            crit = fs[fs_name]["criteria"]
            if not crit:
                raise ValueError(
                    f"{path}: nessun criterio calibrato nel threshold JSON "
                    "(la validation ha saltato tutti i criteri)."
                )
            crit_name = next(iter(crit))                 # unico criterio calibrato
            params = crit[crit_name]
            out = {}
            if "tau" in params:
                out["tau"] = float(params["tau"])
            if "alpha" in params:
                out["alpha"] = float(params["alpha"])
            if "threshold" in params:                    # per SU su soglia scalare
                out["threshold"] = float(params["threshold"])
            if not out:
                raise ValueError(
                    f"{path}: criterio '{crit_name}' senza parametri numerici "
                    f"riconosciuti ({list(params.keys())})."
                )
            return out

        # --- formato CSV piatto (una riga) ---
        row = next(csv.DictReader(f))
        return {k: float(v) for k, v in row.items()}


def load_model_json(path):
    """Legge model.json (coefficienti di uno o piu' regressori logistici)."""
    with open(path) as f:
        return json.load(f)


def budget_folder(output_dir, budget):
    """Ritorna output_dir/top{budget}/ (creandola se manca). budget = numero
    di candidati effettivamente matchati per quelle query (1/5/10/20)."""
    d = os.path.join(output_dir, f"top{budget}")
    os.makedirs(d, exist_ok=True)
    return d


# ============================================================
# IMAGE MATCHING
# ============================================================

def run_im_top1_all(preds_dir, matcher_name, device="cpu", im_size=512):
    """IM tra query e candidato top-1 per OGNI query. Ritorna {id: num_inliers}."""
    matcher = get_matcher(matcher_name, device=device)
    inliers_by_query = {}
    for q_id in tqdm(get_query_ids(preds_dir), desc="IM top-1 su tutte le query"):
        txt_file = os.path.join(preds_dir, f"{q_id}.txt")
        q_path, pred_paths = read_file_preds(txt_file)
        img0 = matcher.load_image(q_path, resize=im_size)
        img1 = matcher.load_image(pred_paths[0], resize=im_size)
        inliers_by_query[q_id] = matcher(deepcopy(img0), img1)["num_inliers"]
    return inliers_by_query


def run_im_top1_with_results(preds_dir, matcher_name, device="cpu", im_size=512, query_ids=None):
    """Come run_im_top1_all, ma ritorna il risultato IM completo (non solo
    num_inliers): serve sia per salvare il .torch delle query skippate sia al
    sequenziale per accumulare. Se query_ids e' dato, solo quel sottoinsieme.
    Ritorna {id: result_dict}."""
    if query_ids is None:
        query_ids = get_query_ids(preds_dir)
    if not query_ids:
        return {}
    matcher = get_matcher(matcher_name, device=device)
    results_by_query = {}
    for q_id in tqdm(query_ids, desc="IM top-1 su tutte le query"):
        txt_file = os.path.join(preds_dir, f"{q_id}.txt")
        q_path, pred_paths = read_file_preds(txt_file)
        img0 = matcher.load_image(q_path, resize=im_size)
        img1 = matcher.load_image(pred_paths[0], resize=im_size)
        result = matcher(deepcopy(img0), img1)
        result["all_desc0"] = result["all_desc1"] = None
        results_by_query[q_id] = result
    return results_by_query


def run_im_extend(preds_dir, query_ids, accumulated_results, start_rank, end_rank,
                   matcher_name, device="cpu", im_size=512):
    """Estende accumulated_results (id -> lista di risultati IM gia' fatti)
    matchando i candidati da start_rank a end_rank (1-indexed, inclusivi) SOLO
    per le query in query_ids. Modifica accumulated_results in place."""
    if not query_ids:
        return
    matcher = get_matcher(matcher_name, device=device)
    for q_id in tqdm(query_ids, desc=f"IM candidati {start_rank}-{end_rank}"):
        txt_file = os.path.join(preds_dir, f"{q_id}.txt")
        q_path, pred_paths = read_file_preds(txt_file)
        img0 = matcher.load_image(q_path, resize=im_size)
        for rank in range(start_rank, end_rank + 1):
            img1 = matcher.load_image(pred_paths[rank - 1], resize=im_size)
            result = matcher(deepcopy(img0), img1)
            result["all_desc0"] = result["all_desc1"] = None
            accumulated_results[q_id].append(result)


# ============================================================
# OUTPUT
# ============================================================

def save_results_torch(query_id, results, folder):
    """Salva la lista di risultati IM reali (len = budget) come <id>.torch in
    folder. Formato per-elemento identico a match_queries_preds.py."""
    torch.save(list(results), os.path.join(folder, f"{query_id}.torch"))


def save_skipped_as_txt(query_ids, preds_dir, folder):
    """Copia il .txt di retrieval delle query skippate SENZA image matching
    (es. su/): non c'e' nessun inlier da inventare, resta valido il retrieval."""
    if not query_ids:
        return
    os.makedirs(folder, exist_ok=True)
    for q_id in query_ids:
        shutil.copy2(os.path.join(preds_dir, f"{q_id}.txt"),
                     os.path.join(folder, f"{q_id}.txt"))


def run_im_topN_subset(preds_dir, query_ids, output_dir, budget, matcher_name,
                        device="cpu", im_size=512):
    """IM sui primi `budget` candidati per ogni query in query_ids; salva un
    .torch (len = budget) in output_dir/top{budget}/."""
    if not query_ids:
        return
    folder = budget_folder(output_dir, budget)
    matcher = get_matcher(matcher_name, device=device)
    for q_id in tqdm(query_ids, desc=f"IM top-{budget} sulle query selezionate"):
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
# RESUME — una query e' "fatta" se gia' finalizzata in un topK
# ============================================================

def query_already_done(output_dir, query_id, budgets):
    """True se la query e' gia' salvata (.torch o .txt) in uno dei topK da una
    run precedente. Usato dal sequenziale per riprendere per-query."""
    for b in budgets:
        folder = os.path.join(output_dir, f"top{b}")
        if (os.path.exists(os.path.join(folder, f"{query_id}.torch")) or
                os.path.exists(os.path.join(folder, f"{query_id}.txt"))):
            return True
    return False


# ============================================================
# PARTIZIONAMENTO
#
# num_inliers BASSI = query incerta -> serve rerank (num_inliers < threshold).
# Per i regressori la probability modella P(serve rerank), quindi
# probability > tau -> rerank (mai il contrario).
# ============================================================

def partition_by_threshold(num_inliers_by_query, threshold):
    """num_inliers < threshold -> serve rerank. Altrimenti skip."""
    rerank_ids = [q for q, n in num_inliers_by_query.items() if n < threshold]
    skip_ids   = [q for q, n in num_inliers_by_query.items() if n >= threshold]
    return rerank_ids, skip_ids


def partition_by_probability(prob_by_query, tau):
    """probability > tau -> serve rerank. Altrimenti skip."""
    rerank_ids = [q for q, p in prob_by_query.items() if p > tau]
    skip_ids   = [q for q, p in prob_by_query.items() if p <= tau]
    return rerank_ids, skip_ids


# ============================================================
# REGRESSIONE LOGISTICA (applicazione, non training)
# ============================================================

def apply_sigmoid(signals_by_query, model_json):
    """Applica scaler + sigmoide a {id: {feature: valore, ...}}. model_json:
    feat_cols, scaler_mean, scaler_scale, coef, intercept. Ritorna {id: prob}.
    Per regressori a 1 feature non dipende dal NOME della feature (il segnale
    runtime usa 'inliers', il JSON puo' avere 'feature_0'/'num_inliers_top1')."""
    feat_cols = model_json["feat_cols"]
    mean  = np.array(model_json["scaler_mean"])
    scale = np.array(model_json["scaler_scale"])
    w     = np.array(model_json["coef"][0])
    b     = model_json["intercept"][0]

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


# ============================================================
# SEGNALE SU (Score Uncertainty) — solo per su/ e su_inliers/
# Calcolato dalle distanze L2 del retrieval, nessun image matching.
# ============================================================

def load_z_data_distances(z_data_path):
    """Carica z_data.torch (--save_for_uncertainty) e ritorna {id: array_L2},
    id = stringa dell'indice (stessa convenzione di preds/<idx>.txt)."""
    data = torch.load(z_data_path, map_location="cpu", weights_only=False)
    distances = np.asarray(data["distances"], dtype=float)  # (N, K)
    return {str(i): distances[i] for i in range(distances.shape[0])}


def l2_to_su(l2_distances, k=10, alpha=0.5, eps=1e-12):
    """s_i = 1/(1+L2_i); RS = media(s_1..k / s_0); SD = mediana(s)/max;
    SU = alpha*RS + (1-alpha)*SD."""
    d = np.asarray(l2_distances, dtype=float)[:k]
    s = 1.0 / (1.0 + d)
    if len(s) < 2:
        return 0.0
    rs = float(np.mean(s[1:] / (s[0] + eps)))
    sd = float(np.median(s) / (np.max(s) + eps))
    return alpha * rs + (1 - alpha) * sd


# ============================================================
# RUNNER GENERICI
#   scalar  -> youden / best_r1 / efficiency / local
#   logistic single -> logistic_hard / logistic_help
# ============================================================

def run_scalar_method(preds_dir, threshold, matcher_name, device, im_size,
                       num_preds, output_dir):
    # IM solo sul top-1 di tutte le query: e' la feature di decisione
    results_top1 = run_im_top1_with_results(preds_dir, matcher_name, device, im_size)
    inliers = {q: r["num_inliers"] for q, r in results_top1.items()}

    rerank_ids, skip_ids = partition_by_threshold(inliers, threshold)
    print_summary(rerank_ids, skip_ids)

    # skip -> top1/: salva il risultato IM del solo top-1 (gia' calcolato)
    folder1 = budget_folder(output_dir, 1)
    for q in skip_ids:
        save_results_torch(q, [results_top1[q]], folder1)

    # rerank -> top{num_preds}/: IM completo
    run_im_topN_subset(preds_dir, rerank_ids, output_dir, num_preds,
                        matcher_name, device, im_size)


def run_logistic_single(preds_dir, tau, model, matcher_name, device, im_size,
                         num_preds, output_dir):
    results_top1 = run_im_top1_with_results(preds_dir, matcher_name, device, im_size)
    signals = {q: {"inliers": r["num_inliers"]} for q, r in results_top1.items()}
    probs   = apply_sigmoid(signals, model)

    rerank_ids, skip_ids = partition_by_probability(probs, tau)
    print_summary(rerank_ids, skip_ids)

    folder1 = budget_folder(output_dir, 1)
    for q in skip_ids:
        save_results_torch(q, [results_top1[q]], folder1)

    run_im_topN_subset(preds_dir, rerank_ids, output_dir, num_preds,
                        matcher_name, device, im_size)


# ============================================================
# LOG
# ============================================================

def print_summary(rerank_ids, skip_ids):
    total = len(rerank_ids) + len(skip_ids)
    pct = 100 * len(rerank_ids) / total if total else 0.0
    print(f"\nQuery totali: {total}")
    print(f"  Rerank (top-20): {len(rerank_ids):5d}  ({pct:.1f}%)")
    print(f"  Skip (solo top-1): {len(skip_ids):5d}  ({100 - pct:.1f}%)")


# ████████████████████████████████████████████████████████████████████
# AGGIUNTE SU — TRAINING + VALIDATION dei metodi su/ e su_inliers/
#
# Tutto cio' che segue serve SOLO agli script in training/ e validation/
# per i metodi basati su SU (riattivati). Non e' usato dai metodi del prof.
# Richiede scikit-learn e pandas (gia' dipendenze del progetto di calibrazione).
# Le funzioni qui sotto riutilizzano l2_to_su() definita sopra.
# ████████████████████████████████████████████████████████████████████

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Colonne richieste nel CSV candidate-level.
#   BASE_REQUIRED_COLS: feature + label necessarie a CHIUNQUE.
#   SU_REQUIRED_COLS:   in piu' la l2_distance, richiesta SOLO da chi usa SU
#                       (metodi su/ e su_inliers/). Vedi load_query_level(needs_l2).
BASE_REQUIRED_COLS = ["query_id", "retrieval_rank",
                      "num_inliers", "is_positive", "rerank_rank_topK"]
SU_REQUIRED_COLS   = BASE_REQUIRED_COLS + ["l2_distance"]

# i tre target/criteri del notebook di calibrazione
SU_TARGETS  = ("hard", "help", "hurts")
SU_CRITERIA = ("P(hard)", "P(help)", "P(help)-aP(hurts)")


# --- LOADER: CSV candidate-level -> DataFrame a livello query + label -------

def load_query_level(csv_dir_or_file, k=10, alpha=0.5, needs_l2=True):
    """Legge uno o piu' CSV candidate-level (dir o file singolo), collassa al
    livello query e calcola SU (via l2_to_su), inliers (negato) e le label
    hard/helps/hurts. Usato da training (per X,y) e validation (per X + correct_*).

    needs_l2: se True (default) la colonna l2_distance e' OBBLIGATORIA (serve a
    calcolare SU: metodi su/ e su_inliers/). Se False (metodi che NON usano SU,
    es. logistic_help/logistic_hard/youden/...) la L2 non e' richiesta: se manca
    viene riempita con NaN e la colonna SU risulta NaN (non usata da quei metodi).

    Colonne ritornate: query_id_full, source_file, SU, inliers, num_inliers_top1,
      correct_0, correct_full_rerank, hard, helps, hurts
    """
    if os.path.isdir(csv_dir_or_file):
        files = sorted(glob(os.path.join(csv_dir_or_file, "*.csv")))
    else:
        files = [csv_dir_or_file]
    if not files:
        raise FileNotFoundError(f"Nessun CSV trovato in {csv_dir_or_file}")

    req = SU_REQUIRED_COLS if needs_l2 else BASE_REQUIRED_COLS

    rows = []
    for fp in files:
        stem = os.path.splitext(os.path.basename(fp))[0]
        df = pd.read_csv(fp)
        missing = [c for c in req if c not in df.columns]
        if missing:
            raise ValueError(f"{fp}: colonne mancanti {missing}")

        # l2_distance opzionale quando SU non serve: placeholder NaN cosi' il
        # resto (num_inliers_top1, correct_*) si calcola comunque.
        if "l2_distance" not in df.columns:
            df["l2_distance"] = np.nan

        for qid, g in df.groupby("query_id", sort=False):
            g = g.sort_values("retrieval_rank")
            if len(g) < k:
                continue  # SU non calcolabile

            l2 = g["l2_distance"].to_numpy(dtype=float)
            # SU solo se la L2 c'e' davvero; altrimenti NaN (metodi non-SU).
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
                # convenzione SU: pochi inlier -> feature alta -> query incerta
                "inliers":             -float(top1["num_inliers"]),
                # grezzo (non negato): feature per i metodi logistici su num_inliers
                "num_inliers_top1":    float(top1["num_inliers"]),
                "correct_0":           correct_0,
                "correct_full_rerank": correct_full_rerank,
                "hard":  int(correct_0 == 0),
                "helps": int((correct_0 == 0) and (correct_full_rerank == 1)),
                "hurts": int((correct_0 == 1) and (correct_full_rerank == 0)),
            })

    if not rows:
        raise ValueError("Nessuna query valida estratta dai CSV (controlla k e le colonne).")
    return pd.DataFrame(rows)


# --- TRAINING di un singolo regressore (Cella 2) ---------------------------

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


# --- Ricostruzione + applicazione regressore per la grid-search (Cella 3) ---

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
    """P(classe positiva). Nome esplicito per non confondersi con
    apply_sigmoid (che opera su dict per i metodi del prof)."""
    return clf.predict_proba(X)[:, 1]


def clean_scores(scores):
    """NaN/inf -> valori finiti (mediana / max / min finiti). Per la grid-search."""
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
