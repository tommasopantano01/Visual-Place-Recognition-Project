import argparse
import pandas as pd


def select_eff95_threshold(sweep: pd.DataFrame, retention: float = 0.95) -> pd.DataFrame:
    """
    Selects the most efficient threshold that retains at least 95%
    of the best adaptive R@1 gain.

    Required columns:
    - threshold
    - pre_r1_pct
    - adaptive_r1_pct
    - saving_pct
    """
    
    #cpy dataframe to avoid modifying the original in order to clculate other metrics further
    df = sweep.copy()

    pre_r1 = df["pre_r1_pct"].iloc[0] #R@1_retrievial
    best_r1 = df["adaptive_r1_pct"].max() #R@1_best

    delta_r = best_r1 - pre_r1 #R@1 Delta

    #calculate the best threshold (in terms of saving) that mantains at least 95% of the best R@1 gain
    if delta_r <= 0:
        target_r1 = pre_r1
    else:
        target_r1 = pre_r1 + retention * delta_r

    candidates = df[df["adaptive_r1_pct"] >= target_r1]

    #sort the candidates by max saving_pct, if equal max adaptive_r1_pct, if equal min threshold, then take the first row (best threshold)
    row = (
        candidates.sort_values(
            ["saving_pct", "adaptive_r1_pct", "threshold"],
            ascending=[False, False, True],
        )
        .iloc[0]
    )

    # output: best threshold and its corresponding adaptive R@1 and saving percentage
    return pd.DataFrame([{
        "method": f"eff{int(retention * 100)}",
        "threshold": int(row["threshold"]),
        "r1_adaptive_pct": row["adaptive_r1_pct"],
        "saving_pct": row["saving_pct"],
    }])


# def main for command line execution
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Threshold sweep CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--retention", type=float, default=0.95)
    args = parser.parse_args()

    sweep = pd.read_csv(args.input)
    result = select_eff95_threshold(sweep, retention=args.retention)
    result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()