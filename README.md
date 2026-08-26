# Visual Place Recognition project
This repository provides a starting code for the **Visual Place Recognition** project of ML course.
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
git clone --recursive https://github.com/tommasopantano01/Visual-Place-Recognition-Project/
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
python download.py
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

Full re-ranking runs image matching (IM) on **all** top-20 candidates of every query, which is expensive. **Adaptive re-ranking** decides **per query** whether the full re-ranking is worth its cost. Most methods buy that decision with a single cheap IM on the **top-1** candidate; the `su` method decides from the retrieval scores alone and costs **zero** IM.

All the code lives in [`VPR-Adaptive-ReRanking/`](./VPR-Adaptive-ReRanking).

## Quick test — reuse the image matching you already have

If the top-20 IM results (`.torch`) already exist, **`--inliers-dir` replays them instead of recomputing**: the first `K` candidates are read from those files. Same numbers, seconds instead of hours, no GPU and no `image-matching-models` needed.

Run every method on every (model, matcher) pair and get one summary table:

```sh
python VPR-Adaptive-ReRanking/run_all_methods.py \
--preds-dir-template   '/content/drive/MyDrive/VPR/preds/{model}_{dataset}' \
--inliers-dir-template '/content/drive/MyDrive/VPR/matching_results/{model}_{dataset}_{matcher}' \
--z-data-template      '/content/drive/MyDrive/VPR/logs/{model}_{dataset}/z_data.torch' \
--dataset 'svox_training' \
--output-root '/content/drive/MyDrive/VPR/adaptive_test'
```

`{model}`, `{matcher}` and `{dataset}` are substituted for every combination (glob `*` is allowed). Combinations whose files are missing are skipped and listed at the end, so it also runs on partial data. It writes `<output-root>/summary_deploy.csv` with base R@1, adaptive R@1, matches per query and saving for every method — the table for the paper.

> [!NOTE]
> In offline mode the saved `.torch` keep only `num_inliers`, the single field re-ranking reads. That is what keeps the output small.

## Methods

| `--threshold` | Family | Decision rule | IM cost of the decision | Needs training | Needs L2 |
|---|---|---|---|---|---|
| `youden` | hard-threshold | rerank if `num_inliers_top1 < T`, `T` = Youden's J | 1 | no | no |
| `best_r1` | hard-threshold | as above, `T` = max adaptive R@1 | 1 | no | no |
| `efficiency` | hard-threshold | as above, `T` = cheapest `T` keeping 95% of the R@1 gain | 1 | no | no |
| `local` | hard-threshold | non-parametric estimate of `P(helps)` — **not implemented yet** | 1 | no | no |
| `logistic_hard` | logistic | rerank if `P(hard) > tau` | 1 | yes | no |
| `logistic_help` | logistic | rerank if `P(help) > tau` | 1 | yes | no |
| `logistic_cost_sensitive` | logistic | rerank if `P(help) - alpha*P(hurts) > tau` | 1 | yes | no |
| `sequential` | logistic cascade | 1 → 5 → 10 → 20, three gates `tau1/tau5/tau10` | 1–20 | yes | no |
| `su` | score uncertainty | decision from the L2-based SU signal | **0** | yes | **yes** |
| `su_inliers` | score uncertainty | decision from SU + `num_inliers_top1` | 1 | yes | **yes** |

## Pipeline

```
retrieval  ─▶  IM top-20  ─▶  candidate-level CSV  ─▶  training  ─▶  validation  ─▶  deploy  ─▶  check performance
(preds .txt)   (.torch)       (one row per                model.json   threshold.csv   top{K}/    recall@N + saving
+ z_data.torch                 query-candidate)
```

The **candidate-level CSV** is the shared calibration dataset, built **twice**: once on the **train** split (training) and once on the **validation** split (threshold selection). Training and validation must not use the same split.

## 1. Build the candidate-level CSV

One row per (query, candidate), with columns:

```
query_id, candidate_path, l2_distance, retrieval_rank, num_inliers, rerank_rank_topK, is_positive, K
```

`l2_distance` comes from `z_data.torch` (retrieval run with `--save_for_uncertainty`) and is required **only** by the SU methods.

> [!WARNING]
> The script that builds this CSV is **not currently in the repository**. The CSVs already produced are on Drive; add the builder under `VPR-Adaptive-ReRanking/training/` to make this step reproducible.

## 2. Training (regressor-based methods only)

```sh
python VPR-Adaptive-ReRanking/train_su.py \
--features 'su' \
--train-csv '<path-to-train-candidate_level.csv>' \
--model 'cosplace' --matcher 'superpoint-lg'
```

Trains the `hard` / `help` / `hurts` logistic regressors and writes `model_*.json` into `validation/<features>/`. Use `--features su_inliers` for the two-feature variant. The hard-threshold methods have **no** training step.

Already-trained regressors can be fetched instead of retrained:

```sh
pip install gdown
python VPR-Adaptive-ReRanking/validation/download_models.py --url '<Google Drive link of validation_models.zip>'
# zip already on the mounted Drive:  --zip /content/drive/MyDrive/VPR/validation_models.zip
# only check what is present:        --check
```

## 3. Validation — choose the thresholds

One script per family; `--method` / `--features` picks the variant. Each writes `threshold_<model>_<matcher>.csv` into `validation/<subdir>/`, which is exactly what the deploy reads back.

```sh
# all methods, all pairs, one command
python VPR-Adaptive-ReRanking/validation/run_all.py \
--val-csv-template '/content/drive/MyDrive/VPR/candidate_level/val_{model}_{matcher}.csv'

# or one at a time
python VPR-Adaptive-ReRanking/validation/thresholds.py --method 'youden' --val-csv '<val.csv>' --model 'cosplace' --matcher 'superpoint-lg'
python VPR-Adaptive-ReRanking/validation/logistic.py   --method 'help'   --val-csv '<val.csv>' --model 'cosplace' --matcher 'superpoint-lg'
python VPR-Adaptive-ReRanking/validation/su.py         --features 'su'   --val-csv '<val.csv>' --model 'cosplace' --matcher 'superpoint-lg'
python VPR-Adaptive-ReRanking/validation/sequential.py                   --val-csv '<val.csv>' --model 'cosplace' --matcher 'superpoint-lg'
```

It also writes `validation/summary.csv`: one row per (method, model, matcher) with the chosen parameters and the validation metrics. See [`validation/README.md`](./VPR-Adaptive-ReRanking/validation/README.md) for the exact columns.

## 4. Run adaptive re-ranking (deploy)

```sh
python VPR-Adaptive-ReRanking/adaptive_reranking.py \
--threshold 'logistic_help' \
--preds-dir '<path-to-predictions-folder>' \
--model 'cosplace' --matcher 'superpoint-lg' \
--inliers-dir '<path-to-top20-inliers-folder>' \
--output-dir '<path-to-output-folder>' \
--num-preds 20
# su and su_inliers also need:  --z-data '<path-to-z_data.torch>'
# drop --inliers-dir to run the matcher live (GPU + image-matching-models + dataset images)
```

The output folder is organised by **stopping budget** — the number of image matchings actually spent on that query:

| Folder | Meaning |
|---|---|
| `top0/<id>.txt` | no IM at all (`su` only): retrieval order kept |
| `top1/<id>.torch` | decision taken on the top-1 IM, no re-ranking |
| `top5/`, `top10/` | intermediate stops of the sequential cascade |
| `top20/<id>.torch` | full re-ranking |

Every query ends up in exactly **one** folder, so counting files per folder gives the cost distribution directly.

## 5. Check adaptive re-ranking performance

```sh
python VPR-Adaptive-ReRanking/check_performance.py \
--preds-dir '<path-to-predictions-folder>' \
--adaptive-RR-dir '<path-to-output-folder>' \
--num-preds 20 \
--recall-values 1 5 10 20
```

Reports where the queries stopped, the IM cost and saving versus full re-ranking, and the adaptive recall@N next to the **base** recall (retrieval only) as a reference.

For the full re-ranking reference on the same data, `reranking.py` prints base and full-rerank recalls side by side. Both scripts use the same conventions — stable sort by `num_inliers` with the retrieval rank as tie-break, and the same positive criterion — so their numbers are directly comparable.
