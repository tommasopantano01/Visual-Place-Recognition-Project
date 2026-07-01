# Visual Place Recognition project
This repository provides a starting code for the **Visual Place Recognition** project of the Advanced Machine Learning / Data analysis and Artificial Intelligence Course.
The following commands are meant to be run locally. If you plan to use Colab, upload the notebook [start_your_project.ipynb](./start_your_project.ipynb) and start from there.
> [!NOTE]  
> ### About datasets format
> The adopted convention is that the names of the files with the images are:
> ```
> @ UTM_easting @ UTM_northing @ UTM_zone_number @ UTM_zone_letter @ latitude @ longitude @ pano_id @ tile_num @ heading @ pitch @ roll @ height @ timestamp @ note @ extension
> ```
> Note that some of these values can be empty (e.g. the timestamp might be unknown), and the only required values are UTM coordinates (obtained from latitude and longitude).
> [!WARNING]  
> Some models require code implementation. You should identify which models require them, where they should be implemented, and then implement them.
## Install the repo
```sh
git clone --recursive https://github.com/FarInHeight/Visual-Place-Recognition-Project.git
```
## Install dependencies
```sh
cd Visual-Place-Recognition-Project/image-matching-models
pip install -e .[all]
pip install faiss-cpu
```
## Download Datasets
```sh
cd ..
python download_datasets.py
```
## Run VPR Evaluation
```sh
python VPR-methods-evaluation/main.py \
--num_workers 8 \
--batch_size 32 \
--log_dir log_dir \
--method=cosplace --backbone=ResNet18 --descriptors_dimension=512 \
--image_size 512 512 \
--database_folder '<path-to-database-folder>' \
--queries_folder '<path-to-queries-folder>' \
--num_preds_to_save 20 \
--recall_values 1 5 10 20 \
--save_for_uncertainty #Must be on, we will use z_data torch in next steps 
```
## Run Image Matching on Retrieval Results
```sh
python match_queries_preds.py \
--preds-dir '<path-to-predictions-folder>' \
--matcher 'superpoint-lg' \
--device 'cuda' \
--num-preds 20
```
## Check Re-ranking Performance
```sh
python reranking.py \
--preds-dir '<path-to-predictions-folder>' \
--inliers-dir '<path-to-inliers-folder>' \
--num-preds 20 \
--recall-values 1 5 10 20
```
## Perform Uncertainty Evalutation [only for AML students]
```sh
python -m vpr_uncertainty.eval \
--preds-dir '<path-to-predictions-folder>' \
--inliers-dir '<path-to-inliers-folder>' \
--z-data-path '<path-to-z-data-file>'
```

---

# Adaptive Re-ranking (extension)

Full re-ranking runs image matching (IM) on **all** top-20 candidates of every query, which is expensive. **Adaptive re-ranking** first runs IM only on the **top-1** candidate and, from that cheap signal, decides **per query** whether it is worth re-ranking the full top-20 or keeping the retrieval order. Each method differs only in *how* it makes that decision; the decision variable is the number of inliers of the top-1 match (`num_inliers_top1`), except for SU approach which relies on retrieval results.

All the code lives in [`VPR-Adaptive-ReRanking/`](./VPR-Adaptive-ReRanking).

## Methods

| Method (`--threshold`) | Family | Decision rule | Needs training | Needs L2 distance |
|---|---|---|---|---|
| `youden` | hard-threshold | rerank if `num_inliers_top1 < T`, `T` = Youden's J | no | no |
| `best_r1` | hard-threshold | as above, `T` = max adaptive R@1 | no | no |
| `efficiency` | hard-threshold | as above, `T` = smallest cost keeping 95% of the R@1 gain | no | no |
| `local` | logistic | non-parametric estimate of `P(helps)` | yes | no |
| `logistic_hard` | logistic | rerank if `P(hard) > tau` | yes | no |
| `logistic_help` | logistic | rerank if `P(help) > tau` | yes | no |
| `logistic_cost_sensitive` | logistic | rerank if `P(help) - alpha*P(hurt) > tau` | yes | no |
| `sequential` | logistic (cascade) | 1 -> 5 -> 10 -> 20, three gates `tau1/tau5/tau10` | yes | no |
| `su` | SU (score uncertainty) | decision from L2-based SU signal | yes | **yes** |
| `su_inliers` | SU | decision from SU + `num_inliers_top1` | yes | **yes** |

> [!NOTE]
> Only the **SU** methods (`su`, `su_inliers`) need the retrieval **L2 distances**. All the other methods work from `num_inliers_top1` alone.

## Pipeline overview

```
retrieval  ─▶  IM top-20  ─▶  build candidate-level CSV  ─▶  [training]  ─▶  [validation]  ─▶  [deploy]  ─▶  check performance
(preds .txt)   (.torch)       (one row per query-candidate)   model.json     threshold.csv     top{K}/       recall@N
+ z_data.torch
```

The **candidate-level CSV** is the shared calibration dataset used by both training and validation. It is built **twice**, once on the **train** split and once on the **validation** split.

> [!IMPORTANT]
> The candidate-level CSVs are **not** shipped in the repo: they are produced by `build_candidate_level_csv.py`. To reproduce from scratch (e.g. on a different split) you must, for that split: run retrieval, run IM on top-20, then build the candidate-level CSV — and only then run training/validation.

## 1. Build the candidate-level CSV

Turns the retrieval `.txt` predictions and the top-20 IM `.torch` files into one row per (query, candidate).

```sh
python VPR-Adaptive-ReRanking/training/candidate_level/build_candidate_level_csv.py \
--preds_dir '<path-to-predictions-folder>' \
--match_dir '<path-to-top20-inliers-folder>' \
--z_data_path '<path-to-z_data.torch>' \
--output_csv '<path-to-candidate_level.csv>' \
--k 20
```

Output columns (with `--z_data_path`):
```
query_id, candidate_path, l2_distance, retrieval_rank, num_inliers, rerank_rank_topK, is_positive, K
```

> [!NOTE]
> `--z_data_path` is **optional** and adds the `l2_distance` column (from `z_data.torch`, saved by retrieval with `--save_for_uncertainty`). It is required **only** to calibrate the SU methods. For every other method it can be omitted.
> The file `training/candidate_level/master.csv` documents the meaning of each column.

## 2. Training (regressor-based methods only)

Trains the single-feature logistic regressors (`hard`, `help`, `hurt`) on `num_inliers_top1` from the **train** candidate-level CSV and serializes their weights to a `model.json`.

```sh
python VPR-Adaptive-ReRanking/training/su.py   # see the script for its arguments
```

> [!NOTE]
> The **hard-threshold** methods (`youden`, `best_r1`, `efficiency`) have **no training step**: their only parameter is the threshold, which is chosen directly in validation.
> The same trained regressors are reused by the logistic methods and (with the extra SU feature) by the SU methods.

## 3. Validation — choose the thresholds

Each method has its own folder `VPR-Adaptive-ReRanking/validation/<method>/`. Validation runs on the **validation** candidate-level CSV (chosen by the user via `--val-csv`) and writes the calibrated thresholds; the deploy step reads them back from here.

**Hard-threshold** (`youden`, `best_r1`, `efficiency`) — build a threshold sweep from the candidate-level and pick the threshold. No model, no L2 distance.
```sh
python VPR-Adaptive-ReRanking/validation/youden/youden.py \
--val-csv '<path-to-validation-candidate_level.csv>'
# best_r1 and efficiency have the same interface (efficiency also accepts --retention)
```
Writes `threshold.csv` (`threshold, r1_adaptive_pct, saving_pct`) plus `sweep.csv` and `selection.csv` for inspection.

**Logistic / SU** — load the trained regressor from `model.json` and grid-search the threshold that maximises the adaptive R@1.
```sh
python VPR-Adaptive-ReRanking/validation/logistic_help/logistic_help.py \
--val-csv '<path-to-validation-candidate_level.csv>' \
--model-json '<path-to-model.json>'
```
`sequential` grid-searches three thresholds (`tau1/tau5/tau10`); `su`/`su_inliers` require a candidate-level built **with** `--z_data_path`.

> [!NOTE]
> We validated on **SF-XS validation**. Since every team member had already computed their own thresholds, the shipped `threshold.csv`/`model.json` were also filled by hand; re-running validation on the same split reproduces the same thresholds.

## 4. Run adaptive re-ranking (deploy)

`main.py` selects the method with `--threshold` and runs it live: IM on top-1, decision, and full top-20 IM only for the queries that need it. Each method loads its calibrated threshold from `validation/<method>/`.

```sh
python VPR-Adaptive-ReRanking/main.py \
--threshold 'youden' \
--preds-dir '<path-to-predictions-folder>' \
--matcher 'superpoint-lg' \
--device 'cuda' \
--num-preds 20 \
--output-dir '<path-to-output-folder>'
# SU methods (su, su_inliers) additionally need:  --z-data '<path-to-z_data.torch>'
```

The output folder is organised by stopping budget: `top1/`, `top5/`, `top10/`, `top20/`. Each query ends up in exactly one, with its IM results saved as `<query_id>.torch`.

## 5. Check adaptive re-ranking performance

Reads the `top{K}/` folders, reports how many queries stopped at each budget and computes the adaptive recall@N.

```sh
python VPR-Adaptive-ReRanking/check_performance.py \
--preds-dir '<path-to-predictions-folder>' \
--adaptive-RR-dir '<path-to-output-folder>' \
--num-preds 20 \
--recall-values 1 5 10 20
```
