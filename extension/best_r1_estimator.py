import argparse
import pandas as pd


def select_best_r1_threshold(sweep: pd.DataFrame) -> pd.DataFrame:
    """
    Selects the threshold that maximizes adaptive R@1.

    Required columns:
    - threshold
    - adaptive_r1_pct
    - saving_pct
    """
    
    #cpy dataframe to avoid modifying the original in order to clculate other metrics further
    df = sweep.copy()

    #cut the dataframe to only include rows with the best adaptive R@1
    best_r1 = df["adaptive_r1_pct"].max()
    candidates = df[df["adaptive_r1_pct"] == best_r1]

    #sort the candidates by max saving_pct, if equal min threshold, then take the first row (best threshold)
    row = (
        candidates.sort_values(
            ["saving_pct", "threshold"],
            ascending=[False, True],
        )
        .iloc[0]
    )

    # output: best threshold and its corresponding adaptive R@1 and saving percentage
    return pd.DataFrame([{
        "method": "best_r1",
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
    result = select_best_r1_threshold(sweep)
    result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()