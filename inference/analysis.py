import pandas as pd
import numpy as np
import ast
from collections import defaultdict

def load_and_expand_csv(path):
    df = pd.read_csv(path, header=None, skiprows=1)  # Skip header row
    times = df.iloc[:, 0]
    parsed = df.iloc[:, 1].apply(ast.literal_eval)
    expanded_time = times.repeat(parsed.str.len()).reset_index(drop=True)
    expanded_values = pd.Series(np.concatenate(parsed.to_list()))
    return expanded_time, expanded_values

def analyze_high_tail_mape(gt_path, pred_path, out_high_tail):
    # Load and expand data
    _, gt_values = load_and_expand_csv(gt_path)
    _, pred_values = load_and_expand_csv(pred_path)

    # Calculate MAPE
    gt_values = gt_values.astype(float)
    pred_values = pred_values.astype(float)
    mape = np.abs((gt_values - pred_values) / gt_values) * 100

    # Group by ground truth
    grouped = defaultdict(list)
    for gt, m in zip(gt_values, mape):
        grouped[gt].append(m)

    # Evaluate average of last 5 MAPEs
    high_tail_rows = []
    for gt in sorted(grouped.keys()):
        values = grouped[gt]
        if len(values) >= 2:
            avg_last5 = np.mean(values[-3:])
            if avg_last5 > 10:
                high_tail_rows.append([gt, avg_last5])

    # Sort by MAPE descending
    df_tail = pd.DataFrame(high_tail_rows, columns=["ground_truth", "avg_mape_last5"])
    df_tail.sort_values(by="avg_mape_last5", ascending=False, inplace=True)
    df_tail.to_csv(out_high_tail, index=False)

# === Usage ===
analyze_high_tail_mape(
    "ground_truth.csv",
    "prediction.csv",
    "high_tail_mape.csv"
)