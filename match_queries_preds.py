import os
import sys
import json
import argparse
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from util import read_file_preds

sys.path.append(str(Path(__file__).parent.joinpath("image-matching-models")))
from matching import get_matcher, available_models
from matching.utils import get_default_device


# ---------------------------------------------------------------------------
# Extension 6.1 — Tabella delle threshold
#
# Chiave:  (tipo_threshold, metodo_vpr, matcher)
# Valore:  num_inliers del top-1 sopra il quale il re-ranking viene saltato
#
# Valori default stimati dal team su SVOX (train) + SF-XS (val).
# Se l'utente ha eseguito gli script in extension/, i valori calcolati
# su dataset propri vengono caricati automaticamente da thresholds_computed.json
# e sovrascrivono i default qui sotto.
#
# 4 tipi di threshold × 4 combinazioni (metodo_vpr, matcher) = 16 valori.
# ---------------------------------------------------------------------------
THRESHOLDS_DEFAULT = {
    # metodo1:
    ("metodo1", "megaloc",  "superpoint-lg"):  None,  # TODO
    ("metodo1", "megaloc",  "loftr"):           None,  # TODO
    ("metodo1", "cosplace", "superpoint-lg"):  None,  # TODO
    ("metodo1", "cosplace", "loftr"):           None,  # TODO
    # metodo2:
    ("metodo2", "megaloc",  "superpoint-lg"):  None,  # TODO
    ("metodo2", "megaloc",  "loftr"):           None,  # TODO
    ("metodo2", "cosplace", "superpoint-lg"):  None,  # TODO
    ("metodo2", "cosplace", "loftr"):           None,  # TODO
    # metodo3:
    ("metodo3", "megaloc",  "superpoint-lg"):  None,  # TODO
    ("metodo3", "megaloc",  "loftr"):           None,  # TODO
    ("metodo3", "cosplace", "superpoint-lg"):  None,  # TODO
    ("metodo3", "cosplace", "loftr"):           None,  # TODO
    # metodo4:
    ("metodo4", "megaloc",  "superpoint-lg"):  None,  # TODO
    ("metodo4", "megaloc",  "loftr"):           None,  # TODO
    ("metodo4", "cosplace", "superpoint-lg"):  None,  # TODO
    ("metodo4", "cosplace", "loftr"):           None,  # TODO
}

THRESHOLD_TYPES = ["metodo1", "metodo2", "metodo3", "metodo4"]

# Percorso del JSON prodotto dagli script in extension/
_COMPUTED_JSON = Path(__file__).parent / "extension" / "thresholds_computed.json"


def load_thresholds():
    """
    Parte dai default hardcoded e sovrascrive con i valori in thresholds_computed.json
    se il file esiste (prodotto dagli script in extension/).
    I valori possono essere:
      - int  → soglia semplice su num_inliers (backward compat)
      - {"type": "threshold", "value": N}  → soglia semplice
      - {"type": "logistic", ...}           → modello logistico con SU
    """
    thresholds = dict(THRESHOLDS_DEFAULT)

    if _COMPUTED_JSON.exists():
        with open(_COMPUTED_JSON) as f:
            content = f.read().strip()
            if content:
                computed = json.loads(content)
        for tipo, vpr_dict in computed.items():
            for vpr, matcher_dict in vpr_dict.items():
                for matcher, value in matcher_dict.items():
                    thresholds[(tipo, vpr, matcher)] = value
        print(f"[Extension 6.1] Threshold custom caricate da {_COMPUTED_JSON}")

    return thresholds


THRESHOLDS = load_thresholds()


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--preds-dir", type=str, help="directory with predictions of a VPR model")
    parser.add_argument("--out-dir", type=str, default=None, help="output directory of image matching results")

    # Scelta del matcher
    parser.add_argument(
        "--matcher",
        type=str,
        default="sift-lg",
        choices=available_models,
        help="choose your matcher",
    )
    parser.add_argument("--device", type=str, default=get_default_device(), choices=["cpu", "cuda"])
    parser.add_argument("--im-size", type=int, default=512, help="resize img to im_size x im_size")
    parser.add_argument("--num-preds", type=int, default=100, help="number of predictions to match")
    parser.add_argument("--start-query", type=int, default=-1, help="query to start from")
    parser.add_argument("--num-queries", type=int, default=-1, help="number of queries")

    # Extension 6.1: skip adattivo del re-ranking
    parser.add_argument(
        "--threshold-type",
        type=str,
        default=None,
        choices=THRESHOLD_TYPES,
        help="(Extension 6.1) tipo di threshold da usare per lo skip adattivo",
    )
    parser.add_argument(
        "--vpr-method",
        type=str,
        default=None,
        help="(Extension 6.1) metodo VPR usato (inferito da --preds-dir se non specificato)",
    )
    parser.add_argument(
        "--su-csv",
        type=str,
        default=None,
        help="(Extension 6.1) CSV con colonne query_id e SU, richiesto per threshold logistiche con SU",
    )

    return parser.parse_args()


def infer_vpr_method(preds_dir):
    """Inferisce il metodo VPR dal nome della cartella delle predizioni."""
    preds_dir = preds_dir.lower()
    known_methods = ["megaloc", "cosplace"]
    found = [m for m in known_methods if m in preds_dir]
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        raise ValueError(f"Metodo VPR ambiguo in '{preds_dir}': trovati {found}. Specifica --vpr-method.")
    raise ValueError(f"Metodo VPR non trovato in '{preds_dir}'. Specifica --vpr-method.")


def get_threshold(threshold_type, vpr_method, matcher_name):
    """
    Restituisce le info sulla threshold per la combinazione (tipo, metodo_vpr, matcher).
    Può restituire:
      - int  → soglia semplice su num_inliers
      - dict {"type": "threshold", "value": N}
      - dict {"type": "logistic", "features": [...], params...}
    """
    key = (threshold_type, vpr_method, matcher_name)
    if key not in THRESHOLDS:
        raise ValueError(f"Nessuna threshold definita per {key}. Aggiungila a THRESHOLDS_DEFAULT.")
    value = THRESHOLDS[key]
    if value is None:
        raise ValueError(
            f"Threshold per {key} non ancora inserita.\n"
            f"Esegui gli script in extension/ per calcolarla."
        )
    return value


def apply_logistic(num_inliers, su_value, logistic_info):
    """
    Applica il modello logistico e restituisce True se il re-ranking è consigliato.
    P(helps) = sigmoid(w * standardize(features) + b)
    Re-rank se P(helps) > tau.
    """
    features_list = []
    for feat in logistic_info["features"]:
        if feat == "SU":
            features_list.append(su_value)
        elif feat == "num_inliers_top1":
            features_list.append(float(num_inliers))

    x         = (np.array(features_list) - np.array(logistic_info["scaler_mean"])) / np.array(logistic_info["scaler_scale"])
    logit_val = np.dot(np.array(logistic_info["coef"][0]), x) + logistic_info["intercept"]
    p_help    = 1.0 / (1.0 + np.exp(-logit_val))
    return p_help > logistic_info["tau"]   # True → re-rank


def main(args):
    device       = args.device
    matcher_name = args.matcher
    img_size     = args.im_size
    num_preds    = args.num_preds
    matcher      = get_matcher(matcher_name, device=device)
    preds_folder = args.preds_dir
    start_query  = args.start_query
    num_queries  = args.num_queries

    # --- Extension 6.1: risolve threshold e modalità di skip ---
    use_threshold  = args.threshold_type is not None
    use_logistic   = False
    threshold      = None
    logistic_info  = None
    su_dict        = {}

    if use_threshold:
        vpr_method = args.vpr_method or infer_vpr_method(args.preds_dir)
        info = get_threshold(args.threshold_type, vpr_method, matcher_name)

        # Normalizza in dict
        if isinstance(info, int):
            info = {"type": "threshold", "value": info}

        if info["type"] == "threshold":
            # Soglia semplice su num_inliers
            threshold = info["value"]

        elif info["type"] == "logistic":
            # Modello logistico: carica SU dal CSV se richiesto dalle feature
            use_logistic  = True
            logistic_info = info
            if "SU" in logistic_info["features"]:
                if args.su_csv is None:
                    raise ValueError(
                        "La threshold selezionata usa il segnale SU. "
                        "Specifica --su-csv con il CSV contenente la colonna SU per il test set."
                    )
                su_df   = pd.read_csv(args.su_csv)
                su_dict = dict(zip(su_df["query_id"].astype(str), su_df["SU"].astype(float)))
                print(f"[Extension 6.1] SU caricato da {args.su_csv} ({len(su_dict)} query)")

    output_folder = Path(preds_folder + f"_{matcher_name}") if args.out_dir is None else Path(args.out_dir)
    output_folder.mkdir(exist_ok=True)

    txt_files = glob(os.path.join(preds_folder, "*.txt"))
    txt_files.sort(key=lambda x: int(Path(x).stem))
    start_query = start_query if start_query >= 0 else 0
    num_queries = num_queries if num_queries >= 0 else len(txt_files)

    for txt_file in tqdm(txt_files[start_query : start_query + num_queries]):
        q_num    = Path(txt_file).stem
        out_file = output_folder.joinpath(f"{q_num}.torch")
        if out_file.exists():
            continue

        results = []
        q_path, pred_paths = read_file_preds(txt_file)
        img0 = matcher.load_image(q_path, resize=img_size)

        if use_threshold:
            # --- Extension 6.1: esegui IM solo sul top-1, poi decidi ---
            img1        = matcher.load_image(pred_paths[0], resize=img_size)
            result_top1 = matcher(deepcopy(img0), img1)
            result_top1["all_desc0"] = result_top1["all_desc1"] = None
            results.append(result_top1)

            # Decisione: re-rank o salta?
            if use_logistic:
                # Applica il modello logistico (usa SU e/o num_inliers del top-1)
                su_val   = su_dict.get(q_num, 0.0)
                do_rerank = apply_logistic(result_top1["num_inliers"], su_val, logistic_info)
            else:
                # Soglia semplice: re-rank se num_inliers è basso
                do_rerank = result_top1["num_inliers"] < threshold

            if do_rerank:
                # top-1 non è affidabile: esegui IM su tutti i restanti candidati
                for pred_path in pred_paths[1:num_preds]:
                    img1   = matcher.load_image(pred_path, resize=img_size)
                    result = matcher(deepcopy(img0), img1)
                    result["all_desc0"] = result["all_desc1"] = None
                    results.append(result)
            else:
                # top-1 è affidabile: salta i restanti
                # num_inliers=0 li manda automaticamente in fondo al reranking
                for _ in range(num_preds - 1):
                    results.append({"num_inliers": 0})

        else:
            # --- Comportamento originale: IM su tutte le predizioni ---
            for pred_path in pred_paths[:num_preds]:
                img1   = matcher.load_image(pred_path, resize=img_size)
                result = matcher(deepcopy(img0), img1)
                result["all_desc0"] = result["all_desc1"] = None
                results.append(result)

        torch.save(results, out_file)


if __name__ == "__main__":
    import pandas as pd
    args = parse_arguments()
    main(args)
