# Visual Place Recognition project

> [!NOTE]
> ### Dataset filename convention
> `@ UTM_easting @ UTM_northing @ UTM_zone_number @ UTM_zone_letter @ latitude @ longitude @ pano_id @ tile_num @ heading @ pitch @ roll @ height @ timestamp @ note @ extension`
> Only the UTM coordinates are required; other fields can be empty.

> [!WARNING]
> Some models require code implementation — identify which ones and where.

## Install

```sh
git clone --recursive https://github.com/tommasopantano01/Visual-Place-Recognition-Project/
cd Visual-Place-Recognition-Project/image-matching-models
pip install -e .[all]
pip install faiss-cpu
cd ..
python download.py   # datasets + pretrained regressors
```

---

# Adaptive Re-ranking

Full re-ranking runs image matching (IM) on all top-20 candidates of every query — expensive. **Adaptive re-ranking** decides *per query* whether that's worth it: most methods buy the decision with one cheap IM on the top-1 candidate; `su` decides from retrieval scores alone, costing **zero** IM.

Code lives in [`vpr-adaptive-reranking/`](./vpr-adaptive-reranking).

## Building blocks

Every workflow below starts with these two steps. Shown once here; the workflows just say "run this on split X."

**Retrieval** — images → ranked candidates per query:
```sh
python VPR-methods-evaluation/main.py \
--num_workers 8 --batch_size 32 --log_dir <name> \
--method=cosplace --backbone=ResNet18 --descriptors_dimension=512 --image_size 512 512 \
--database_folder '<database-folder>' --queries_folder '<queries-folder>' \
--num_preds_to_save 20 --recall_values 1 5 10 20 \
--save_for_uncertainty     # keep ON — produces z_data.torch, needed by su/su_inliers
```
→ `logs/<name>/<timestamp>/preds/*.txt` + `z_data.torch`

**Image matching** — num_inliers for the top-K candidates of each query:
```sh
python match_queries_preds.py \
--preds-dir '<preds-folder>' --matcher 'superpoint-lg' --device 'cuda' --num-preds 20
```
→ `<preds-folder>_superpoint-lg/*.torch`

---

## Workflow 1 — Training

1. Retrieval + image matching on the **train** split
2. Build the candidate-level CSV:
```sh
   python vpr-adaptive-reranking/build_candidate_level_csv.py \
   --preds_dir '<preds>' --match_dir '<preds>_superpoint-lg' \
   --z_data_path '<z_data.torch>' --output_csv '<candidate_level_train.csv>' --k 20
```
3. Train the regressors:
```sh
   python vpr-adaptive-reranking/train_logistic.py --method hard \
   --train-csv '<candidate_level_train.csv>' --model cosplace --matcher superpoint-lg
   # repeat with --method help, --method cost_sensitive

   python vpr-adaptive-reranking/train_su.py --features su \
   --train-csv '<candidate_level_train.csv>' --model cosplace --matcher superpoint-lg
   # repeat with --features su_inliers
```
   → `validation/{logistic_hard,logistic_help,logistic_cost_sensitive,su,su_inliers}/model_*.json`

`youden` / `best_r1` / `efficiency` need no training — pure thresholds, start at Workflow 2.

## Workflow 2 — Validation

1. Retrieval + image matching on the **validation** split (different from train)
2. Build the candidate-level CSV (same command as above, different split)
3. Search thresholds — one script per family, or all at once:
```sh
   python vpr-adaptive-reranking/validation/run_all.py \
   --val-csv-template '<candidate_level_val.csv>'
```
   or individually: `validation/thresholds.py --method youden|best_r1|efficiency`, `validation/logistic.py --method hard|help|cost_sensitive`, `validation/su.py --features su|su_inliers`, `validation/sequential.py` — all take `--val-csv --model --matcher`.

   → `validation/<subdir>/threshold_<model>_<matcher>.csv` + `validation/summary.csv`

## Workflow 3 — Test / Deploy

1. Retrieval + image matching on the **test** split (different from train & validation)
2. Run adaptive re-ranking, per method:
```sh
   python vpr-adaptive-reranking/adaptive_reranking.py \
   --threshold '<method>' --preds-dir '<preds>' --model cosplace --matcher superpoint-lg \
   --inliers-dir '<preds>_superpoint-lg' --output-dir '<out>' --num-preds 20
   # su / su_inliers also need: --z-data '<z_data.torch>'
   # drop --inliers-dir to matcher live instead of reusing step-1 results
```
   Output organized by stopping budget: `top0/` (su, no IM), `top1/` (decision only), `top5/`, `top10/`, `top20/` (full rerank). One query, one folder — counting files gives the cost distribution directly.
3. Measure it:
```sh
   python vpr-adaptive-reranking/check_performance.py \
   --preds-dir '<preds>' --adaptive-RR-dir '<out>' --num-preds 20 --recall-values 1 5 10 20
```
   Reports stop distribution, IM cost/saving vs full rerank, adaptive R@N vs base R@N.

   For the full-rerank reference on the same data: `python reranking.py --preds-dir '<preds>' --inliers-dir '<preds>_superpoint-lg' --num-preds 20 --recall-values 1 5 10 20` — same sort convention as `check_performance.py`, numbers are directly comparable.

---

## Methods

| `--threshold` | Rule | IM cost | Training | Needs L2 |
|---|---|---|---|---|
| `youden` / `best_r1` / `efficiency` | rerank if `num_inliers_top1 < T` | 1 | no | no |
| `local` | non-parametric `P(helps)` | 1 | — | **not implemented** |
| `logistic_hard` | rerank if `P(hard) > tau` | 1 | yes | no |
| `logistic_help` | rerank if `P(help) > tau` | 1 | yes | no |
| `logistic_cost_sensitive` | rerank if `P(help) − α·P(hurts) > tau` | 1 | yes | no |
| `sequential` | cascade 1→5→10→20, gates `tau1/tau5/tau10` | 1–20 | yes | no |
| `su` | decision from L2-based uncertainty | **0** | yes | **yes** |
| `su_inliers` | SU + `num_inliers_top1` | 1 | yes | **yes** |

## Notes

- Train / validation / test **must be three different splits**, or numbers are inflated.
- `--model` / `--matcher` must match across all three workflows for the same regressor/threshold to apply.
- `--inliers-dir` (Workflow 3, `run_all_methods.py`) replays precomputed `.torch` instead of live matching — use it whenever IM for that split already exists.
- Uncertainty eval (AML students only): `python -m vpr_uncertainty.eval --preds-dir '<preds>' --inliers-dir '<inliers>' --z-data-path '<z_data.torch>'`
