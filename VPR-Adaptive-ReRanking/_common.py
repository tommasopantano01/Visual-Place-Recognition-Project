"""
_common.py — Motore condiviso da tutti i metodi di adaptive reranking.

Convenzione di output, identica per ogni metodo, dentro --output-dir:
  output-dir/txt_folder/<id>.txt    — query che restano al retrieval puro
                                       (copia diretta del .txt originale,
                                       nessun dato di inliers da riportare)
  output-dir/torch_folder/<id>.torch — query con almeno un risultato di
                                       image matching reale (anche solo
                                       parziale, es. fermate a top-5)

Il retrieval restituisce .txt, l'image matching restituisce .torch (perche'
contiene gli inliers): la cartella di destinazione segue solo questo.
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
import torch
from tqdm import tqdm

# Risolve i path rispetto alla root del repo, indipendentemente da quale
# sottocartella importa questo file (VPR-adaptive-re-ranking/youden/, ecc.)
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_REPO_ROOT))
sys.path.append(str(_REPO_ROOT / "image-matching-models"))

from util import read_file_preds
from matching import get_matcher


# ============================================================
# LETTURA QUERY E PARAMETRI
# ============================================================

def get_query_ids(preds_dir):
    """Lista degli id query (nomi file senza estensione) in preds_dir, ordinati."""
    txt_files = glob(os.path.join(preds_dir, "*.txt"))
    return sorted((Path(f).stem for f in txt_files), key=int)


def load_threshold_csv(path):
    """Legge threshold.csv (una riga, una o piu' colonne numeriche) come dict."""
    with open(path) as f:
        row = next(csv.DictReader(f))
    return {k: float(v) for k, v in row.items()}


def load_model_json(path):
    """Legge model.json (coefficienti di uno o piu' regressori logistici)."""
    with open(path) as f:
        return json.load(f)


def output_subdirs(output_dir):
    """Ritorna (txt_folder, torch_folder) dentro output_dir, creandole se mancano."""
    txt_dir   = os.path.join(output_dir, "txt_folder")
    torch_dir = os.path.join(output_dir, "torch_folder")
    os.makedirs(txt_dir, exist_ok=True)
    os.makedirs(torch_dir, exist_ok=True)
    return txt_dir, torch_dir


# ============================================================
# IMAGE MATCHING
# ============================================================

def run_im_top1_all(preds_dir, matcher_name, device="cpu", im_size=512):
    """
    Esegue IM tra query e candidato top-1 per OGNI query in preds_dir.
    Ritorna {query_id: num_inliers}.
    """
    matcher = get_matcher(matcher_name, device=device)
    query_ids = get_query_ids(preds_dir)
    inliers_by_query = {}

    for q_id in tqdm(query_ids, desc="IM top-1 su tutte le query"):
        txt_file = os.path.join(preds_dir, f"{q_id}.txt")
        q_path, pred_paths = read_file_preds(txt_file)
        img0 = matcher.load_image(q_path, resize=im_size)
        img1 = matcher.load_image(pred_paths[0], resize=im_size)
        result = matcher(deepcopy(img0), img1)
        inliers_by_query[q_id] = result["num_inliers"]

    return inliers_by_query


def run_im_top1_with_results(preds_dir, matcher_name, device="cpu", im_size=512, query_ids=None):
    """
    Come run_im_top1_all, ma ritorna il risultato IM completo (non solo
    num_inliers). Serve a sequential/, dove il match del top-1 va riusato
    e accumulato negli stage successivi invece di essere rifatto.
    Se query_ids e' specificato, esegue IM solo su quel sottoinsieme
    (usato per il resume: non rifare query gia' completate).
    Ritorna {query_id: result_dict}.
    """
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
                   matcher_name, device="cpu", im_size=512,
                   checkpoint_fn=None):
    """
    Estende accumulated_results (dict query_id -> lista di risultati IM gia'
    fatti) eseguendo IM sui candidati da start_rank a end_rank (1-indexed,
    inclusivi) SOLO per le query in query_ids. Modifica accumulated_results
    in place. Ritorna {query_id: max_num_inliers_finora}.

    Se checkpoint_fn e' fornita, viene chiamata DOPO ogni singola query
    (non a fine batch) cosi' un'interruzione a meta' non perde il lavoro
    gia' fatto sulle query precedenti in questo stage.
    """
    if not query_ids:
        return {}
    matcher = get_matcher(matcher_name, device=device)
    max_inliers = {}

    for q_id in tqdm(query_ids, desc=f"IM candidati {start_rank}-{end_rank}"):
        txt_file = os.path.join(preds_dir, f"{q_id}.txt")
        q_path, pred_paths = read_file_preds(txt_file)
        img0 = matcher.load_image(q_path, resize=im_size)

        for rank in range(start_rank, end_rank + 1):
            pred_path = pred_paths[rank - 1]
            img1 = matcher.load_image(pred_path, resize=im_size)
            result = matcher(deepcopy(img0), img1)
            result["all_desc0"] = result["all_desc1"] = None
            accumulated_results[q_id].append(result)

        max_inliers[q_id] = max(r["num_inliers"] for r in accumulated_results[q_id])

        if checkpoint_fn is not None:
            checkpoint_fn(q_id, accumulated_results[q_id])

    return max_inliers


# ============================================================
# OUTPUT — query "skip" (retrieval puro, txt_folder)
# ============================================================

def save_skipped_as_txt(query_ids, preds_dir, txt_folder):
    """
    Copia il .txt originale del retrieval per le query che non necessitano
    alcun rerank. Non si inventa un risultato di inliers che non esiste:
    se non e' stato fatto matching aggiuntivo, il dato che resta valido e'
    solo il file di retrieval cosi' com'e'.
    """
    if not query_ids:
        return
    os.makedirs(txt_folder, exist_ok=True)
    for q_id in query_ids:
        src = os.path.join(preds_dir, f"{q_id}.txt")
        dst = os.path.join(txt_folder, f"{q_id}.txt")
        shutil.copy2(src, dst)


# ============================================================
# OUTPUT — query "rerank" (image matching reale, torch_folder)
# ============================================================

def run_im_top20_subset(preds_dir, query_ids, torch_folder, matcher_name,
                         device="cpu", im_size=512, num_preds=20):
    """
    Esegue IM completo (top-num_preds) solo per le query in query_ids.
    Salva un .torch per query in torch_folder, stesso formato di
    match_queries_preds.py (compatibile con check_performance.py).
    """
    if not query_ids:
        return
    matcher = get_matcher(matcher_name, device=device)
    os.makedirs(torch_folder, exist_ok=True)

    for q_id in tqdm(query_ids, desc="IM top-20 sulle query selezionate"):
        out_file = os.path.join(torch_folder, f"{q_id}.torch")
        if os.path.exists(out_file):
            continue

        txt_file = os.path.join(preds_dir, f"{q_id}.txt")
        q_path, pred_paths = read_file_preds(txt_file)
        img0 = matcher.load_image(q_path, resize=im_size)
        results = []
        for pred_path in pred_paths[:num_preds]:
            img1 = matcher.load_image(pred_path, resize=im_size)
            result = matcher(deepcopy(img0), img1)
            result["all_desc0"] = result["all_desc1"] = None
            results.append(result)
        torch.save(results, out_file)


# ============================================================
# CHECKPOINT / RESUME — solo per sequential/
#
# Ogni query nel torch_folder ha, accanto al .torch (sempre lungo
# num_preds con zero-padding, formato stabile per i consumer), un piccolo
# sidecar .progress con un intero: quante entry sono risultati IM reali.
# Permette di riprendere uno stage interrotto senza rifare le query gia'
# completate e senza dover ispezionare/indovinare il contenuto del .torch.
# ============================================================

def save_checkpoint(query_id, real_results, torch_folder, num_preds=20):
    """Sovrascrive il checkpoint corrente della query con lo stato aggiornato."""
    os.makedirs(torch_folder, exist_ok=True)
    n_done = len(real_results)
    padded = list(real_results) + [{"num_inliers": 0}] * (num_preds - n_done)
    torch.save(padded, os.path.join(torch_folder, f"{query_id}.torch"))
    with open(os.path.join(torch_folder, f"{query_id}.progress"), "w") as f:
        f.write(str(n_done))


def load_checkpoint(query_id, torch_folder):
    """
    Se esiste un checkpoint precedente per questa query, ritorna
    (n_done, lista_risultati_reali). Altrimenti (0, []).
    """
    progress_fp = os.path.join(torch_folder, f"{query_id}.progress")
    torch_fp    = os.path.join(torch_folder, f"{query_id}.torch")
    if not os.path.exists(progress_fp) or not os.path.exists(torch_fp):
        return 0, []
    with open(progress_fp) as f:
        n_done = int(f.read().strip())
    results = torch.load(torch_fp, map_location="cpu", weights_only=False)
    return n_done, results[:n_done]


# ============================================================
# PARTIZIONAMENTO
#
# probability/num_inliers ALTI = poco affidabile -> serve rerank.
# Confermato: probability > tau -> rerank (mai il contrario).
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
    """
    Applica scaler + sigmoide a un dict {query_id: {feature: valore, ...}}.
    model_json deve avere: feat_cols, scaler_mean, scaler_scale, coef, intercept.
    Ritorna {query_id: probability}.
    """
    feat_cols = model_json["feat_cols"]
    mean  = np.array(model_json["scaler_mean"])
    scale = np.array(model_json["scaler_scale"])
    w     = np.array(model_json["coef"][0])
    b     = model_json["intercept"][0]

    prob_by_query = {}
    for q_id, sig in signals_by_query.items():
        x = np.array([sig[c] for c in feat_cols], dtype=float)
        z = (x - mean) / scale
        logit_val = float(np.dot(w, z) + b)
        prob_by_query[q_id] = 1.0 / (1.0 + np.exp(-logit_val))
    return prob_by_query


# ============================================================
# SEGNALE SU (Score Uncertainty) — solo per su/ e su_inliers/
#
# Calcolato dalle distanze L2 del retrieval, nessun image matching.
# "L2" nella pipeline e' una scorciatoia per questo segnale (RS+SD), non
# la sola distanza grezza del top-1.
# ============================================================

def load_z_data_distances(z_data_path):
    """
    Carica z_data.torch (prodotto dal retrieval con --save_for_uncertainty)
    e ritorna {query_id: array_L2} per ogni riga, query_id come stringa
    dell'indice (stessa convenzione di preds/<idx>.txt).
    """
    data = torch.load(z_data_path, map_location="cpu", weights_only=False)
    distances = np.asarray(data["distances"], dtype=float)  # shape (N, K)
    return {str(i): distances[i] for i in range(distances.shape[0])}


def l2_to_su(l2_distances, k=10, alpha=0.5, eps=1e-12):
    """
    s_i = 1/(1+L2_i)   similarita'
    RS  = media(s_1..k / s_0)        quanto i successivi sono vicini al primo
    SD  = mediana(s_0..k) / max      compattezza della shortlist
    SU  = alpha*RS + (1-alpha)*SD
    """
    d = np.asarray(l2_distances, dtype=float)[:k]
    s = 1.0 / (1.0 + d)
    if len(s) < 2:
        return 0.0
    rs = float(np.mean(s[1:] / (s[0] + eps)))
    sd = float(np.median(s) / (np.max(s) + eps))
    return alpha * rs + (1 - alpha) * sd


# ============================================================
# RUNNER GENERICO PER I METODI "SCALARI" (youden/best_r1/efficiency/local)
# E PER I METODI LOGISTIC A SINGOLO REGRESSORE (hard/help)
# ============================================================

def run_scalar_method(preds_dir, threshold, matcher_name, device, im_size,
                       num_preds, output_dir):
    txt_folder, torch_folder = output_subdirs(output_dir)

    inliers = run_im_top1_all(preds_dir, matcher_name, device, im_size)
    rerank_ids, skip_ids = partition_by_threshold(inliers, threshold)
    print_summary(rerank_ids, skip_ids)

    save_skipped_as_txt(skip_ids, preds_dir, txt_folder)
    run_im_top20_subset(preds_dir, rerank_ids, torch_folder,
                         matcher_name, device, im_size, num_preds)


def run_logistic_single(preds_dir, tau, model, matcher_name, device, im_size,
                         num_preds, output_dir):
    txt_folder, torch_folder = output_subdirs(output_dir)

    inliers = run_im_top1_all(preds_dir, matcher_name, device, im_size)
    signals = {q: {"inliers": n} for q, n in inliers.items()}
    probs   = apply_sigmoid(signals, model)

    rerank_ids, skip_ids = partition_by_probability(probs, tau)
    print_summary(rerank_ids, skip_ids)

    save_skipped_as_txt(skip_ids, preds_dir, txt_folder)
    run_im_top20_subset(preds_dir, rerank_ids, torch_folder,
                         matcher_name, device, im_size, num_preds)


# ============================================================
# LOG
# ============================================================

def print_summary(rerank_ids, skip_ids):
    total = len(rerank_ids) + len(skip_ids)
    pct = 100 * len(rerank_ids) / total if total else 0.0
    print(f"\nQuery totali: {total}")
    print(f"  Rerank (top-20): {len(rerank_ids):5d}  ({pct:.1f}%)")
    print(f"  Skip (solo top-1): {len(skip_ids):5d}  ({100 - pct:.1f}%)")
