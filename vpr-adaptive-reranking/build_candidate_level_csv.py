from __future__ import annotations

import argparse
import os
from glob import glob
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


BASE_OUTPUT_COLUMNS = [
    "query_id",
    "candidate_path",
    "retrieval_rank",
    "num_inliers",
    "rerank_rank_topK",
    "is_positive",
    "K",
]

OUTPUT_COLUMNS_WITH_DISTANCE = [
    "query_id",
    "candidate_path",
    "l2_distance",
    "retrieval_rank",
    "num_inliers",
    "rerank_rank_topK",
    "is_positive",
    "K",
]


def sort_key(path: str | Path) -> tuple[int, int | str]:
    """Sort paths numerically when the stem is numeric, otherwise alphabetically."""
    stem = Path(path).stem
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def query_sort_value(query_id: Any) -> int | str:
    """Sort query ids numerically when possible, otherwise alphabetically."""
    try:
        return int(query_id)
    except (TypeError, ValueError):
        return str(query_id)


def to_int(value: Any) -> int:
    """Convert Python numbers, NumPy scalars, and scalar torch tensors to int."""
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().item())
    if isinstance(value, np.generic):
        return int(value.item())
    return int(value)


def parse_prediction_txt(txt_file: str | Path) -> tuple[list[str], list[str]]:
    """
    Read one pre-IM prediction file.

    Expected sections:
        Predictions paths:
        ...
        Positives paths:
        ...
    """
    pred_paths: list[str] = []
    positive_paths: list[str] = []

    reading_preds = False
    reading_pos = False

    with open(txt_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if line.startswith("Predictions paths"):
                reading_preds = True
                reading_pos = False
                continue

            if line.startswith("Positives paths"):
                reading_preds = False
                reading_pos = True
                continue

            if reading_preds and line:
                pred_paths.append(line)
            elif reading_pos and line:
                positive_paths.append(line)

    return pred_paths, positive_paths


def compute_rerank_ranks(
    num_inliers: Iterable[int],
    retrieval_ranks: Iterable[int],
) -> dict[int, int]:
    """
    Compute the post-IM reranking rank inside top-K.

    Sorting rule:
      1. higher num_inliers first;
      2. ties are broken by the original retrieval_rank, ascending.
    """
    num_inliers_array = np.asarray(list(num_inliers))
    retrieval_ranks_array = np.asarray(list(retrieval_ranks))

    order = np.lexsort((retrieval_ranks_array, -num_inliers_array))

    return {
        int(retrieval_ranks_array[idx]): int(new_rank)
        for new_rank, idx in enumerate(order, start=1)
    }


def load_matching_data(torch_file: str | Path) -> list[dict[str, Any]]:
    """Load one .torch matching file and return it as a list-like object."""
    data = torch.load(torch_file, map_location="cpu", weights_only=False)

    if not isinstance(data, (list, tuple)):
        raise TypeError(
            f"Expected a list/tuple in {torch_file}, found {type(data).__name__}."
        )

    return list(data)

def load_z_data_distances(z_data_path: str | Path | None) -> np.ndarray | None:
    """
    Load FAISS retrieval distances from z_data.torch.

    Expected structure:
        z_data["distances"] with shape (num_queries, num_retrieved_candidates)

    With faiss.IndexFlatL2 these values are the distances returned by squared L2 distances.
    """
    if z_data_path is None:
        return None

    z_data_path = Path(z_data_path)
    if not z_data_path.exists():
        raise FileNotFoundError(f"z_data_path not found: {z_data_path}")

    z_data = torch.load(z_data_path, map_location="cpu", weights_only=False)
    if not isinstance(z_data, dict):
        raise TypeError(
            f"Expected a dict in {z_data_path}, found {type(z_data).__name__}. Make sure 'save_for_uncertainty' is active in 'VPR-methods-evaluation/main.py'"
        )
    if "distances" not in z_data:
        raise KeyError(
            f"Missing key 'distances' in {z_data_path}. Available keys: {list(z_data.keys())}"
        )

    distances = z_data["distances"]
    if isinstance(distances, torch.Tensor):
        distances = distances.detach().cpu().numpy()
    else:
        distances = np.asarray(distances)

    if distances.ndim != 2:
        raise ValueError(
            f"Expected z_data['distances'] to be 2D, found shape {distances.shape}."
        )

    return distances.astype(float, copy=False)


def get_l2_distance(
    distances: np.ndarray | None,
    query_id: str,
    candidate_index: int,
) -> float | None:
    """Return distances[int(query_id), candidate_index] when z_data distances are available."""
    if distances is None:
        return None

    try:
        query_index = int(query_id)
    except ValueError as exc:
        raise ValueError(
            "Cannot read distances from z_data because query_id is not numeric: "
            f"{query_id!r}. The script expects txt files named like 000.txt, etc."
        ) from exc

    if query_index < 0 or query_index >= distances.shape[0]:
        raise IndexError(
            f"query_id {query_id} maps to row {query_index}, but z_data distances has "
            f"{distances.shape[0]} query rows."
        )
    if candidate_index < 0 or candidate_index >= distances.shape[1]:
        raise IndexError(
            f"candidate index {candidate_index} is outside z_data distances shape {distances.shape}."
        )

    return float(distances[query_index, candidate_index])


def build_rows_for_query(
    txt_file: str | Path,
    match_dir: str | Path,
    k: int,
    distances: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """
    Build candidate-level rows for one query.

    Returns:
        rows, status

    Status can be:
        ok, empty_preds, missing_torch, short_matching
    """
    txt_file = Path(txt_file)
    match_dir = Path(match_dir)
    query_id = txt_file.stem

    pred_paths, positive_paths = parse_prediction_txt(txt_file)
    positive_set = set(positive_paths)

    if not pred_paths:
        return [], "empty_preds"

    torch_file = match_dir / f"{query_id}.torch"
    if not torch_file.exists():
        return [], "missing_torch"

    data = load_matching_data(torch_file)

    n_available = min(k, len(pred_paths), len(data))
    status = "ok" if n_available == k else "short_matching"

    pred_paths_k = pred_paths[:n_available]
    data_k = data[:n_available]

    num_inliers_list: list[int] = []
    retrieval_ranks: list[int] = []

    for j, candidate_data in enumerate(data_k):
        retrieval_rank = j + 1

        if "num_inliers" not in candidate_data:
            raise KeyError(
                f"Missing key 'num_inliers' in {torch_file} for candidate rank {retrieval_rank}. "
                f"Available keys: {list(candidate_data.keys())}"
            )

        retrieval_ranks.append(retrieval_rank)
        num_inliers_list.append(to_int(candidate_data["num_inliers"]))

    rerank_rank_map = compute_rerank_ranks(num_inliers_list, retrieval_ranks)

    rows: list[dict[str, Any]] = []
    for j, candidate_path in enumerate(pred_paths_k):
        retrieval_rank = j + 1
        row = {
            "query_id": query_id,
            "candidate_path": candidate_path,
            "retrieval_rank": retrieval_rank,
            "num_inliers": num_inliers_list[j],
            "rerank_rank_topK": rerank_rank_map[retrieval_rank],
            "is_positive": int(candidate_path in positive_set),
            "K": k,
        }

        if distances is not None:
            row["l2_distance"] = get_l2_distance(
                distances=distances,
                query_id=query_id,
                candidate_index=j,
            )

        rows.append(row)

    return rows, status


def build_candidate_level_csv(
    preds_dir: str | Path,
    match_dir: str | Path,
    output_csv: str | Path,
    k: int = 20,
    z_data_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build and save the final candidate-level CSV in one pass."""
    preds_dir = Path(preds_dir)
    match_dir = Path(match_dir)
    output_csv = Path(output_csv)

    txt_files = sorted(glob(str(preds_dir / "*.txt")), key=sort_key)
    torch_files = sorted(glob(str(match_dir / "*.torch")), key=sort_key)

    print("PREDS_DIR:", preds_dir)
    print("MATCH_DIR:", match_dir)
    print("OUT_CANDIDATE_CSV:", output_csv)
    print("Z_DATA_PATH:", z_data_path if z_data_path is not None else "not provided")
    print("K:", k)
    print("Numero .txt predictions:", len(txt_files))
    print("Numero .torch matching:", len(torch_files))

    if not txt_files:
        raise RuntimeError(f"No .txt files found in {preds_dir}")
    if not torch_files:
        raise RuntimeError(f"No .torch files found in {match_dir}")

    distances = load_z_data_distances(z_data_path)
    output_columns = OUTPUT_COLUMNS_WITH_DISTANCE if distances is not None else BASE_OUTPUT_COLUMNS

    if distances is not None:
        print("z_data distances shape:", distances.shape)

    rows: list[dict[str, Any]] = []
    status_counts = {
        "ok": 0,
        "missing_torch": 0,
        "short_matching": 0,
        "empty_preds": 0,
    }

    for txt_file in tqdm(txt_files, desc="Building candidate-level CSV"):
        query_rows, status = build_rows_for_query(
            txt_file,
            match_dir=match_dir,
            k=k,
            distances=distances,
        )
        status_counts[status] = status_counts.get(status, 0) + 1
        rows.extend(query_rows)

    candidate_df = pd.DataFrame(rows, columns=output_columns)

    if not candidate_df.empty:
        before = len(candidate_df)
        candidate_df = candidate_df.drop_duplicates(
            subset=["query_id", "candidate_path", "retrieval_rank"],
            keep="first",
        ).copy()
        after = len(candidate_df)

        candidate_df["query_id_sort"] = candidate_df["query_id"].apply(query_sort_value)
        candidate_df = candidate_df.sort_values(
            ["query_id_sort", "retrieval_rank"]
        ).drop(columns=["query_id_sort"])
        candidate_df = candidate_df[output_columns]
    else:
        before = after = 0

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    candidate_df.to_csv(output_csv, index=False)

    print("\nSaved candidate-level CSV to:")
    print(output_csv)
    print("\nFinal shape:", candidate_df.shape)
    print("Rows before duplicate removal:", before)
    print("Rows after duplicate removal:", after)
    print("Duplicates removed:", before - after)
    print("\nStatus counts:")
    for key, value in status_counts.items():
        print(f"  {key}: {value}")

    n_queries = candidate_df["query_id"].nunique() if not candidate_df.empty else 0
    print("\nQueries in output:", n_queries)
    print("Expected rows if all have K rows:", n_queries * k)
    print("Actual rows:", len(candidate_df))

    if not candidate_df.empty:
        print("\nHead:")
        print(candidate_df.head().to_string(index=False))

    return candidate_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build candidate-level CSV from pre-IM .txt predictions and post-IM .torch files."
    )
    parser.add_argument(
        "--preds_dir",
        required=True,
        help="Directory containing pre-IM prediction .txt files.",
    )
    parser.add_argument(
        "--match_dir",
        required=True,
        help="Directory containing post-IM matching .torch files.",
    )
    parser.add_argument(
        "--output_csv",
        required=True,
        help="Path of the output candidate-level CSV.",
    )
    parser.add_argument(
        "--z_data_path",
        default=None,
        help=(
            "Optional path to z_data.torch containing z_data['distances']. "
            "If provided, a l2_distance column is added before retrieval_rank."
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        default=20,
        help="Number of candidates per query to export. Default: 20.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_candidate_level_csv(
        preds_dir=args.preds_dir,
        match_dir=args.match_dir,
        output_csv=args.output_csv,
        k=args.k,
        z_data_path=args.z_data_path,
    )


if __name__ == "__main__":
    main()
