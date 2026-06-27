import argparse
import pandas as pd


def select_youden_threshold(sweep: pd.DataFrame) -> pd.DataFrame:
    """
    Selects the R@1-based hard/easy threshold.

    Required columns:
    - threshold
    - adaptive_r1_pct
    - saving_pct
    - either youden or both tpr_hard_pct and fpr_easy_pct
    """

    #cpy dataframe to avoid modifying the original in order to clculate other metrics further
    df = sweep.copy()

    # we could insert the youden calculation in the sweep script so we can avoid this check
    if "youden" not in df.columns:
        df["youden"] = df["tpr_hard_pct"] - df["fpr_easy_pct"]

    # reorder dataframe by max youden, if equal max saving_pct, if equal max adaptive_r1_pct, if equal min threshold, then take the first row (best threshold)
    row = (
        df.sort_values(
            ["youden", "saving_pct", "adaptive_r1_pct", "threshold"],
            ascending=[False, False, False, True],
        )
        .iloc[0]
    )

    # output: best threshold and its corresponding adaptive R@1 and saving percentage
    return pd.DataFrame([{
        "method": "youden",
        "threshold": int(row["threshold"]),
        "r1_adaptive_pct": row["adaptive_r1_pct"],
        "saving_pct": row["saving_pct"],
    }])


# def main for command line execution
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Threshold sweep CSV path") 
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    sweep = pd.read_csv(args.input)
    result = select_youden_threshold(sweep)
    result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()