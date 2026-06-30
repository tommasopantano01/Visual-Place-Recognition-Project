"""
sequential/sequential.py — Sequential adaptive re-ranking: cascata a tre
cancelli invece di una singola decisione binaria.

    top-1 -> gate1 -> stop  oppure  top-5  -> gate5  -> stop  oppure  top-10
                                                          -> gate10 -> stop  oppure  top-20

Ogni cancello e' un regressore logistico a singola feature:
  gate1  (1->5):   input = num_inliers del top-1
  gate5  (5->10):  input = max(num_inliers) tra i candidati visti finora (top-5)
  gate10 (10->20): input = max(num_inliers) tra i candidati visti finora (top-10)
probability > tau -> continua al budget successivo. Altrimenti ferma qui.

NOTA — non ancora confermato al 100%: gate5/gate10 corrispondono a M5/M10
nel report, confermati. gate1 e' segnalato come "approssimato riusando il
modello P(helps|I_1) gia' allenato" (local/), ma il target di training
esatto e' ancora da confermare con Luca.

OUTPUT:
  output-dir/txt_folder/<id>.txt    — fermate al gate1 (mai rerankate)
  output-dir/torch_folder/<id>.torch — fermate a top-5/10/20, con tanti
                                       risultati IM reali quanti il
                                       budget raggiunto, resto a zero

RESUME: se il processo si interrompe, una nuova esecuzione con lo stesso
--output-dir riprende da dove era arrivata, query per query — non rifa'
il matching gia' completato. Il checkpoint si aggiorna dopo OGNI query,
non a fine stage, quindi anche un'interruzione a meta' di uno stage perde
al massimo il lavoro su una singola query.

Uso:
    python VPR-adaptive-re-ranking/sequential/sequential.py \
        --preds-dir preds/ --matcher superpoint-lg --output-dir out/
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from _common import (
    load_threshold_csv, load_model_json, output_subdirs, get_query_ids,
    run_im_top1_with_results, run_im_extend, apply_sigmoid,
    partition_by_probability, save_skipped_as_txt, save_checkpoint,
    load_checkpoint,
)

_THRESHOLD_CSV = Path(__file__).parent / "threshold.csv"
_MODEL_JSON    = Path(__file__).parent / "model.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Adaptive reranking — sequential (1->5->10->20)")
    parser.add_argument("--preds-dir",   required=True)
    parser.add_argument("--matcher",     required=True)
    parser.add_argument("--device",      default="cpu")
    parser.add_argument("--im-size",     type=int, default=512)
    parser.add_argument("--num-preds",   type=int, default=20)
    parser.add_argument("--output-dir",  required=True)
    return parser.parse_args()


def print_stage(label, stopped_ids, total):
    pct = 100 * len(stopped_ids) / total if total else 0.0
    print(f"  fermate a {label}: {len(stopped_ids):5d}  ({pct:.1f}%)")


def cleanup_torch_checkpoint(query_id, torch_folder):
    """Rimuove un eventuale checkpoint torch/.progress per una query che,
    dopo un resume, e' stata finalizzata come txt (gate1 = stop)."""
    for ext in (".torch", ".progress"):
        fp = os.path.join(torch_folder, f"{query_id}{ext}")
        if os.path.exists(fp):
            os.remove(fp)


def main(args):
    hp     = load_threshold_csv(_THRESHOLD_CSV)   # tau1, tau5, tau10
    models = load_model_json(_MODEL_JSON)         # gate1, gate5, gate10
    print(f"tau1={hp['tau1']}  tau5={hp['tau5']}  tau10={hp['tau10']}")

    txt_folder, torch_folder = output_subdirs(args.output_dir)
    query_ids = get_query_ids(args.preds_dir)
    total = len(query_ids)

    # --- Resume: determina lo stato di partenza di ogni query ---
    accumulated = {}
    pending = []
    for q in query_ids:
        if os.path.exists(os.path.join(txt_folder, f"{q}.txt")):
            continue  # gia' finalizzata come skip in una run precedente
        n_done, results = load_checkpoint(q, torch_folder)
        if n_done >= args.num_preds:
            continue  # gia' arrivata a top-20 in una run precedente
        accumulated[q] = results
        pending.append(q)

    already_done = total - len(pending)
    if already_done:
        print(f"Resume: {already_done} query gia' completate in precedenza, salto.")

    def checkpoint(q_id, results):
        save_checkpoint(q_id, results, torch_folder, args.num_preds)

    # --- Stage 1: IM top-1 per chi non l'ha ancora (fresco o mai iniziato) ---
    need_stage1 = [q for q in pending if len(accumulated[q]) == 0]
    new_top1 = run_im_top1_with_results(args.preds_dir, args.matcher, args.device,
                                         args.im_size, query_ids=need_stage1)
    for q, r in new_top1.items():
        accumulated[q] = [r]
        checkpoint(q, accumulated[q])

    # --- Gate 1: chiunque abbia esattamente 1 entry va valutato ---
    # (sia chi l'ha appena fatta, sia chi la riprende da un checkpoint precedente)
    at_gate1 = [q for q in pending if len(accumulated[q]) == 1]
    signals1 = {q: {"inliers": accumulated[q][0]["num_inliers"]} for q in at_gate1}
    p1 = apply_sigmoid(signals1, models["gate1"])
    continue_5, stop_1 = partition_by_probability(p1, hp["tau1"])

    save_skipped_as_txt(stop_1, args.preds_dir, txt_folder)
    for q in stop_1:
        cleanup_torch_checkpoint(q, torch_folder)
    print_stage("top-1", stop_1, total)

    # --- Stage 2: IM candidati 2-5 per chi continua, checkpoint per query ---
    run_im_extend(args.preds_dir, continue_5, accumulated, 2, 5,
                  args.matcher, args.device, args.im_size, checkpoint_fn=checkpoint)

    # --- Gate 2: chiunque abbia 5 entry va valutato ---
    at_gate2 = [q for q in pending if len(accumulated[q]) == 5]
    signals5 = {q: {"inliers": max(r["num_inliers"] for r in accumulated[q])} for q in at_gate2}
    p5 = apply_sigmoid(signals5, models["gate5"])
    continue_10, stop_5 = partition_by_probability(p5, hp["tau5"])
    print_stage("top-5", stop_5, total)   # gia' su disco grazie al checkpoint

    # --- Stage 3: IM candidati 6-10 per chi continua ---
    run_im_extend(args.preds_dir, continue_10, accumulated, 6, 10,
                  args.matcher, args.device, args.im_size, checkpoint_fn=checkpoint)

    at_gate3 = [q for q in pending if len(accumulated[q]) == 10]
    signals10 = {q: {"inliers": max(r["num_inliers"] for r in accumulated[q])} for q in at_gate3}
    p10 = apply_sigmoid(signals10, models["gate10"])
    continue_20, stop_10 = partition_by_probability(p10, hp["tau10"])
    print_stage("top-10", stop_10, total)

    # --- Stage 4: IM candidati 11-20, nessun ulteriore cancello ---
    run_im_extend(args.preds_dir, continue_20, accumulated, 11, args.num_preds,
                  args.matcher, args.device, args.im_size, checkpoint_fn=checkpoint)
    print_stage("top-20", continue_20, total)


if __name__ == "__main__":
    main(parse_args())
