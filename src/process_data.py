from collections import defaultdict

import pandas as pd


def length_distribution(data: pd.DataFrame, sequence_column: str):
    lengths = [len(seq) for seq in data[sequence_column]]
    distr = defaultdict(int)
    for num in lengths:
        distr[num] += 1
    return distr


def length_filter(data: pd.DataFrame, sequence_column: str, max_len: int = 20):
    lengths = data[sequence_column].map(len)
    mask = lengths <= max_len
    removed = int((~mask).sum())
    return data[mask].reset_index(drop=True), removed


def filter_noncanonical(data: pd.DataFrame, sequence_column: str):
    s = data[sequence_column].astype("string")
    mask = s.str.fullmatch(f"[{'ACDEFGHIKLMNPQRSTVWY'}]+")
    removed = int((~mask.fillna(False)).sum())
    return data[mask.fillna(False)].reset_index(drop=True), removed


def process_sequences(data: pd.DataFrame, sequence_column: str, max_len: int = 20):
    report = {"input_rows": len(data)}

    df = data.dropna(subset=[sequence_column]).copy()
    df = df[df[sequence_column].astype("string").str.len() > 0]
    report["removed_empty"] = report["input_rows"] - len(df)

    df, aa_removed = filter_noncanonical(df, sequence_column)
    report["removed_noncanonical"] = aa_removed

    df, length_removed = length_filter(df, sequence_column, max_len=max_len)
    report["removed_length"] = length_removed

    distr = length_distribution(df, sequence_column)
    report["output_rows"] = len(df)
    report["removed_total"] = report["input_rows"] - report["output_rows"]

    return df, distr, report


if __name__ == "__main__":
    df = pd.read_csv("data/proteins.csv")
    clean, distr, report = process_sequences(df, "sequence", max_len=800)

    clean.to_csv("data/proteins_clean.csv", index=False)

    print("Report:", report)
    print("Results saved to data/proteins_clean.csv")
