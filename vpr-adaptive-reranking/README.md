# Visual Place Recognition project Extension

## Install

```sh
git clone --recursive https://github.com/tommasopantano01/Visual-Place-Recognition-Project/
cd Visual-Place-Recognition-Project/image-matching-models
pip install -e .[all]
pip install faiss-cpu
cd ..
python download.py
```

---

# Adaptive Re-ranking

Full re-ranking runs image matching (IM) on all top-20 candidates of every query — expensive. **Adaptive re-ranking** decides *per query* whether that's worth it: most methods buy the decision with one cheap IM on the top-1 candidate; `su` decides from retrieval scores alone, costing **zero** IM.

## Building blocks

Every workflow below starts with these two steps. Shown once here; the workflows just say "run this on split X."

**Retrieval** — images → ranked candidates per query:
```sh
python VPR-methods-evaluation/main.py \
--num_workers 8 --batch_size 32 --log_dir '<name>' \
--method='<method>' --backbone (if needed) --descriptors_dimension='<desc_dim>' --image_size '<img_W img_H >' \
--database_folder '<database-folder>' --queries_folder '<queries-folder>' \
--num_preds_to_save --recall_values \
--save_for_uncertainty     # keep ON — produces z_data.torch, needed by su/su_inliers
```
→ `logs/<name>/<timestamp>/preds/*.txt` + `z_data.torch`

**Image matching** — num_inliers for the top-K candidates of each query:
```sh
python match_queries_preds.py \
--preds-dir '<preds-folder>' --matcher '<matcher>' --device 'cuda' --num-preds
```
→ `<preds-folder>_<matcher>/*.torch`

---

## Workflow 1 — Training

1. Retrieval + image matching on the **train** split
2. Build the candidate-level CSV:
```sh
   python vpr-adaptive-reranking/build_candidate_level_csv.py \
   --preds_dir '<preds>' --match_dir '<preds>_<matcher>' \
   --z_data_path '<z_data.torch>' --output_csv '<candidate_level_train.csv>' --k '<k>'
```
3. Train the regressors:
```sh
   python vpr-adaptive-reranking/train_logistic.py --method '<method>' \
   --train-csv '<candidate_level_train.csv>' --model '<model>' --matcher '<matcher>'

   python vpr-adaptive-reranking/train_su.py --features '<feature>' \
   --train-csv '<candidate_level_train.csv>' --model '<model>' --matcher '<matcher>'
```
   → `validation/{logistic_hard,logistic_help,logistic_cost_sensitive,su,su_inliers}/model_*.json`

`youden` / `best_r1` / `efficiency` need no training — pure thresholds, start at Workflow 2.

## Workflow 2 — Validation

1. Retrieval + image matching on the **validation** split (different from train)
2. Build the candidate-level CSV (same command as above, different split)
3. Search thresholds — one script per family, or all at once:
```sh
   python vpr-adaptive-reranking/validation/<script>.py \
   --method '<method>' --val-csv '<val.csv>' --model '<model>' --matcher '<matcher>'
```
   (`thresholds.py`, `logistic.py`, `su.py` — uses `--features` instead of `--method` —, `sequential.py`)

   or all at once, setting `run_all.py`:
```sh
   python vpr-adaptive-reranking/validation/run_all.py \
   --val-csv-template '<candidate_level_val.csv>'
```
   → `validation/<subdir>/threshold_<model>_<matcher>.csv` + `validation/summary.csv`
## Workflow 3 — Test

1. Retrieval + image matching on the **test** split
2. Run adaptive re-ranking, per method:
```sh
   python vpr-adaptive-reranking/adaptive_reranking.py \
   --threshold '<method>' --preds-dir '<preds>' --model '<model>' --matcher '<matcher>' \
   --inliers-dir '<preds>_<matcher>' --output-dir '<out>' --num-preds 20
   # su / su_inliers also need: --z-data '<z_data.torch>'
   # drop --inliers-dir to match live instead of reusing step-1 results
```
   Output organized by stopping budget: `top0/` (su, no IM), `top1/` (decision only), `top5/`, `top10/`, `top20/` (full rerank). One query, one folder — counting files gives the cost distribution directly.
3. Measure it:
```sh
   python vpr-adaptive-reranking/check_performance.py \
   --preds-dir '<preds>' --adaptive-RR-dir '<out>' --num-preds 20 --recall-values 1 5 10 20
```
   Reports stop distribution, IM cost/saving vs full rerank, adaptive R@N vs base R@N.

   For the full-rerank reference on the same data: `python reranking.py --preds-dir '<preds>' --inliers-dir '<preds>_<matcher>' --num-preds 20 --recall-values 1 5 10 20` — same sort convention as `check_performance.py`, numbers are directly comparable.

---

## Methods

| `--threshold` | Rule | IM cost | Training | Needs L2 |
|---|---|---|---|---|
| `youden` / `best_r1` / `efficiency` | rerank if `num_inliers_top1 < T` | 1 | no | no |
| `logistic_hard` | rerank if `P(hard) > tau` | 1 | yes | no |
| `logistic_help` | rerank if `P(help) > tau` | 1 | yes | no |
| `logistic_cost_sensitive` | rerank if `P(help) − α·P(hurts) > tau` | 1 | yes | no |
| `sequential` | cascade 1→5→10→20, gates `tau1/tau5/tau10` | 1–20 | yes | no |
| `su` | decision from L2-based uncertainty | **0** | yes | **yes** |
| `su_inliers` | SU + `num_inliers_top1` | 1 | yes | **yes** |

## Notes

- `--model` / `--matcher` must match across all three workflows for the same regressor/threshold to apply.
- `--inliers-dir` (Workflow 3, `run_all_methods.py`) replays precomputed `.torch` instead of live matching — use it whenever IM for that split already exists.
- Uncertainty eval (AML students only): `python -m vpr_uncertainty.eval --preds-dir '<preds>' --inliers-dir '<inliers>' --z-data-path '<z_data.torch>'`
