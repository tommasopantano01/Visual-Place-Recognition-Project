# Validation — choose the thresholds

Validation runs on a **validation candidate-level CSV chosen by the user** and
writes the calibrated thresholds/parameters that the deploy scripts in
`methods/` read back. It trains nothing.

The layout mirrors `methods/`: one script per family, `--method` / `--features`
selects the variant.

| Script | Variants | Needs model JSON | Needs `l2_distance` |
|---|---|---|---|
| `thresholds.py --method` | `youden` `best_r1` `efficiency` `local`* | no | no |
| `logistic.py --method` | `hard` `help` `cost_sensitive` | yes | no |
| `su.py --features` | `su` `su_inliers` | yes | **yes** |
| `sequential.py` | — | yes (3 gate JSONs) | no |

\* `local` (Luca's non-parametric estimate) is not implemented yet: paste the code in `thresholds.py::select_local_threshold`.

## 0. One-time setup: download the trained regressors

The model JSON files come from training and are **not tracked** in the repo
(`.gitignore`). They are shipped as one zip on Google Drive.

```sh
pip install gdown
python VPR-Adaptive-ReRanking/validation/download_models.py --url "<Google Drive share link of validation_models.zip>"
# zip already on your mounted Drive:  --zip /content/drive/MyDrive/VPR/validation_models.zip
# only check what is present:        --check
```

The zip layout (a leading `validation/` folder is tolerated):

```
logistic_hard/model_<model>_<matcher>.json
logistic_help/model_<model>_<matcher>.json
logistic_cost_sensitive/model_logistic_cost_sensitive_<model>_<matcher>.json
su/model_su_<model>_<matcher>.json
su_inliers/model_su_num_inliers_<model>_<matcher>.json
sequential/seq_model_continue_{1,5,10}_*<model>*<matcher>*.json
```
`<model>` ∈ {`cosplace`, `megaloc`}, `<matcher>` ∈ {`superpoint-lg`, `loftr`}.

## 1. Run everything (recommended)

```sh
python VPR-Adaptive-ReRanking/validation/run_all.py \
  --val-csv-template "/content/drive/MyDrive/VPR/candidate_level/val_{model}_{matcher}.csv"
```
`{model}` / `{matcher}` are replaced for every pair; glob `*` is allowed
(e.g. `".../*sfxs_val*{model}*{matcher}*.csv"`). Pairs whose CSV is missing are
skipped and listed at the end. Restrict with `--methods`, `--models`, `--matchers`.

## 2. Run a single method

```sh
python VPR-Adaptive-ReRanking/validation/thresholds.py --method youden --val-csv <csv> --model cosplace --matcher superpoint-lg
python VPR-Adaptive-ReRanking/validation/logistic.py   --method help   --val-csv <csv> --model cosplace --matcher superpoint-lg
python VPR-Adaptive-ReRanking/validation/su.py         --features su   --val-csv <csv> --model cosplace --matcher superpoint-lg
python VPR-Adaptive-ReRanking/validation/sequential.py                 --val-csv <csv> --model cosplace --matcher superpoint-lg
```
`--model-json` / `--models-dir` override the default JSON location; `sp-lg` is
accepted as an alias of `superpoint-lg`.

## 3. Outputs (written automatically)

In `validation/<subdir>/` for every (model, matcher):

| File | Content |
|---|---|
| `threshold_<model>_<matcher>.csv` | **one row, numeric only** — read by `methods/` |
| `selection_<model>_<matcher>.csv` | method, validation CSV used, metrics, params |
| `sweep_<model>_<matcher>[_crit].csv` | the full grid explored (for plots) |

and `validation/summary.csv`: **one row per (method, model, matcher)** with all
validated thresholds and parameters (`params` column) plus base / full-rerank /
adaptive R@1, % reranked, matches per query and saving.

Threshold columns per family:

| Family | `threshold_*.csv` columns |
|---|---|
| thresholds | `threshold, r1_adaptive_pct, saving_pct` |
| logistic hard/help | `tau, r1_adaptive_pct, reranked_pct, saving_pct` |
| logistic cost_sensitive | `alpha, tau, …` |
| su / su_inliers | `hard_tau, help_tau, cs_alpha, cs_tau` (+ metrics per criterion) |
| sequential | `tau1, tau5, tau10, r1_adaptive_pct, matches_per_query, saving_pct, stop_top{1,5,10,20}_pct` |

Re-running on a different validation CSV overwrites these files (the dataset
name is recorded in `selection_*.csv` and `summary.csv`).
