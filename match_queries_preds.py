import os
import sys
import json
import argparse
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
# Se l'utente ha eseguito gli script in extension_6_1/, i valori calcolati
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

# Percorso del JSON prodotto dagli script in extension_6_1/
_COMPUTED_JSON = Path(__file__).parent / "extension_6_1" / "thresholds_computed.json"


def load_thresholds():
    """
    Parte dai default hardcoded e sovrascrive con i valori in thresholds_computed.json
    se il file esiste (prodotto dagli script nonparametric/logistic in extension_6_1/).
    """
    thresholds = dict(THRESHOLDS_DEFAULT)

    if _COMPUTED_JSON.exists():
        with open(_COMPUTED_JSON) as f:
            computed = json.load(f)
        # struttura JSON: { tipo: { vpr_method: { matcher: valore } } }
        for tipo, vpr_dict in computed.items():
            for vpr, matcher_dict in vpr_dict.items():
                for matcher, value in matcher_dict.items():
                    thresholds[(tipo, vpr, matcher)] = value
        print(f"[Extension 6.1] Threshold custom caricate da {_COMPUTED_JSON}")

    return thresholds


# Carica al momento dell'import: usa default o JSON se disponibile
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
        help="(Extension 6.1) se impostato, esegue IM su tutti i top-20 solo se num_inliers del top-1 < threshold",
    )
    parser.add_argument(
        "--vpr-method",
        type=str,
        default=None,
        help="(Extension 6.1) metodo VPR usato per generare le predizioni (obbligatorio con --threshold-type)",
    )

    return parser.parse_args()


def get_threshold(threshold_type, vpr_method, matcher_name):
    """Restituisce la threshold per la combinazione (tipo_threshold, metodo_vpr, matcher)."""
    key = (threshold_type, vpr_method, matcher_name)
    if key not in THRESHOLDS:
        raise ValueError(f"Nessuna threshold definita per la chiave {key}. Aggiungila a THRESHOLDS_DEFAULT.")
    value = THRESHOLDS[key]
    if value is None:
        raise ValueError(
            f"La threshold per {key} non è ancora stata inserita.\n"
            f"Esegui extension_6_1/nonparametric_threshold_estimator.py o "
            f"extension_6_1/logistic_threshold_estimator.py per calcolarla."
        )
    return value


def main(args):
    device = args.device
    matcher_name = args.matcher
    img_size = args.im_size
    num_preds = args.num_preds
    matcher = get_matcher(matcher_name, device=device)
    preds_folder = args.preds_dir
    start_query = args.start_query
    num_queries = args.num_queries

    # --- Extension 6.1: recupera la threshold se richiesto ---
    use_threshold = args.threshold_type is not None
    if use_threshold:
        if args.vpr_method is None:
            raise ValueError("--vpr-method deve essere specificato quando si usa --threshold-type.")
        threshold = get_threshold(args.threshold_type, args.vpr_method, matcher_name)

    output_folder = Path(preds_folder + f"_{matcher_name}") if args.out_dir is None else Path(args.out_dir)
    output_folder.mkdir(exist_ok=True)

    txt_files = glob(os.path.join(preds_folder, "*.txt"))
    txt_files.sort(key=lambda x: int(Path(x).stem))
    start_query = start_query if start_query >= 0 else 0
    num_queries = num_queries if num_queries >= 0 else len(txt_files)

    for txt_file in tqdm(txt_files[start_query : start_query + num_queries]):
        q_num = Path(txt_file).stem
        out_file = output_folder.joinpath(f"{q_num}.torch")
        if out_file.exists():
            continue

        results = []
        q_path, pred_paths = read_file_preds(txt_file)
        img0 = matcher.load_image(q_path, resize=img_size)

        if use_threshold:
            # --- Extension 6.1: esegui IM solo sul top-1, poi decidi ---
            img1 = matcher.load_image(pred_paths[0], resize=img_size)
            result_top1 = matcher(deepcopy(img0), img1)
            result_top1["all_desc0"] = result_top1["all_desc1"] = None
            results.append(result_top1)

            if result_top1["num_inliers"] < threshold:
                # top-1 non è affidabile: esegui IM su tutti i restanti candidati
                for pred_path in pred_paths[1:num_preds]:
                    img1 = matcher.load_image(pred_path, resize=img_size)
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
                img1 = matcher.load_image(pred_path, resize=img_size)
                result = matcher(deepcopy(img0), img1)
                result["all_desc0"] = result["all_desc1"] = None
                results.append(result)

        torch.save(results, out_file)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
