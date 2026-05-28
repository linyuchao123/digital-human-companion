import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from tslearn.metrics import dtw as tslearn_dtw
except ImportError:
    tslearn_dtw = None


RECOLA_ROLE_MAP = {
    "P25": "P1",
    "P26": "P2",
    "P41": "P1",
    "P42": "P2",
    "P45": "P1",
    "P46": "P2",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate facial behavior generation metrics.")
    parser.add_argument("--data-root", required=True, help="Dataset root directory containing split folders.")
    parser.add_argument("--split", default="val", help="Dataset split to evaluate, e.g. val.")
    parser.add_argument("--index-csv", required=True, help="Official person_specific_<split>.csv file.")
    parser.add_argument("--neighbor-matrix", required=True, help="Official neighbor matrix .npy file.")
    parser.add_argument("--prediction", required=True, help="prediction_emotion.npy with shape [N, K, T, 25].")
    parser.add_argument("--output-json", help="Optional output path for saving metrics as JSON.")
    parser.add_argument("--fps", type=int, default=25, help="Frame rate used for FRSyn.")
    parser.add_argument(
        "--metrics",
        default="frcorr,frcorr_star,frdist,frdiv,frdvs,frvar,frsyn",
        help="Comma-separated metrics to compute.",
    )
    return parser.parse_args()


def normalize_metric_names(text):
    mapping = {
        "frcorr*": "frcorr_star",
        "frcorrstar": "frcorr_star",
    }
    items = []
    for token in text.split(","):
        name = token.strip().lower()
        if not name:
            continue
        items.append(mapping.get(name, name))
    return items


def load_person_specific_order(index_csv):
    with open(index_csv, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) <= 1:
        raise ValueError(f"Index CSV is empty or missing data rows: {index_csv}")

    data_rows = rows[1:]
    speaker_paths = []
    listener_paths = []
    for row in data_rows:
        if len(row) < 3:
            raise ValueError("Each row in person_specific CSV must have at least 3 columns.")
        speaker_paths.append(row[1].strip())
        listener_paths.append(row[2].strip())

    speaker_order = speaker_paths + listener_paths
    listener_order = listener_paths + speaker_paths
    return speaker_order, listener_order


def to_emotion_rel_path(video_rel_path):
    rel_path = video_rel_path.replace("\\", "/") + ".csv"
    if "NoXI" in rel_path:
        rel_path = rel_path.replace("/Novice_video/", "/P2/")
        rel_path = rel_path.replace("/Expert_video/", "/P1/")
    if "/RECOLA/" in rel_path:
        for src, dst in RECOLA_ROLE_MAP.items():
            rel_path = rel_path.replace(f"/{src}/", f"/{dst}/")
    return rel_path


def load_emotion_sequence(data_root, split, video_rel_path, cache, target_len):
    rel_path = to_emotion_rel_path(video_rel_path)
    if rel_path not in cache:
        emotion_path = Path(data_root) / split / "Emotion" / Path(rel_path)
        if not emotion_path.exists():
            raise FileNotFoundError(f"Emotion file not found: {emotion_path}")
        data = np.loadtxt(emotion_path, delimiter=",", skiprows=1, dtype=np.float32)
        if data.ndim != 2 or data.shape[1] != 25:
            raise ValueError(f"Unexpected emotion shape in {emotion_path}: {data.shape}")
        cache[rel_path] = data

    sequence = cache[rel_path]
    if sequence.shape[0] < target_len:
        raise ValueError(
            f"Emotion sequence shorter than target length {target_len}: {rel_path}, got {sequence.shape[0]}"
        )
    return sequence[:target_len]


def load_ground_truth_arrays(data_root, split, index_csv, target_len):
    speaker_order, listener_order = load_person_specific_order(index_csv)
    cache = {}

    speaker_sequences = []
    listener_sequences = []
    iterator = zip(speaker_order, listener_order)
    for speaker_rel, listener_rel in tqdm(
        iterator,
        total=len(speaker_order),
        desc="Loading ground truth emotion sequences",
    ):
        speaker_sequences.append(load_emotion_sequence(data_root, split, speaker_rel, cache, target_len))
        listener_sequences.append(load_emotion_sequence(data_root, split, listener_rel, cache, target_len))

    return np.stack(speaker_sequences), np.stack(listener_sequences)


def corrcoef(x, y):
    c = np.cov(x, y)
    try:
        d = np.diag(c)
    except ValueError:
        return c / c
    stddev = np.sqrt(d.real)
    c = c / stddev[:, None]
    c = c / stddev[None, :]
    c = np.nan_to_num(c)
    np.clip(c.real, -1, 1, out=c.real)
    return c


def concordance_correlation_coefficient(y_true, y_pred):
    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError("CCC expects 2D arrays with shape [T, D].")

    ccc_list = []
    for dim_idx in range(y_true.shape[1]):
        cor = corrcoef(y_true[:, dim_idx], y_pred[:, dim_idx])[0][1]
        mean_true = np.mean(y_true[:, dim_idx])
        mean_pred = np.mean(y_pred[:, dim_idx])
        var_true = np.var(y_true[:, dim_idx])
        var_pred = np.var(y_pred[:, dim_idx])
        sd_true = np.std(y_true[:, dim_idx])
        sd_pred = np.std(y_pred[:, dim_idx])
        numerator = 2 * cor * sd_true * sd_pred
        denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
        ccc = numerator / (denominator + 1e-8)
        ccc_list.append(ccc)
    return float(np.mean(ccc_list))


def dtw_distance(x, y):
    if tslearn_dtw is not None:
        return float(tslearn_dtw(x.astype(np.float32), y.astype(np.float32)))

    n, m = x.shape[0], y.shape[0]
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        xi = x[i - 1]
        for j in range(1, m + 1):
            cost = np.linalg.norm(xi - y[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m])


def weighted_dtw(pred_seq, gt_seq):
    total = 0.0
    for start, end, weight in ((0, 15, 1.0 / 15.0), (15, 17, 1.0), (17, 25, 1.0 / 8.0)):
        total += weight * dtw_distance(pred_seq[:, start:end], gt_seq[:, start:end])
    return total


def compute_frcorr(pred, listener_gt, neighbor_matrix):
    total = 0.0
    for sample_idx in tqdm(range(pred.shape[0]), desc="Computing FRCorr"):
        neighbor_indices = np.flatnonzero(neighbor_matrix[sample_idx])
        if neighbor_indices.size == 0:
            raise ValueError(f"Sample {sample_idx} has an empty neighbor set.")
        for cand_idx in range(pred.shape[1]):
            best = max(
                concordance_correlation_coefficient(listener_gt[neighbor_idx], pred[sample_idx, cand_idx])
                for neighbor_idx in neighbor_indices
            )
            total += best
    return total / pred.shape[0]


def compute_frdist(pred, listener_gt, neighbor_matrix):
    total = 0.0
    for sample_idx in tqdm(range(pred.shape[0]), desc="Computing FRdist"):
        neighbor_indices = np.flatnonzero(neighbor_matrix[sample_idx])
        if neighbor_indices.size == 0:
            raise ValueError(f"Sample {sample_idx} has an empty neighbor set.")
        for cand_idx in range(pred.shape[1]):
            best = min(
                weighted_dtw(pred[sample_idx, cand_idx], listener_gt[neighbor_idx])
                for neighbor_idx in neighbor_indices
            )
            total += best
    return total / pred.shape[0]


def _sum_ordered_pairwise_sq(flat_array):
    count = flat_array.shape[0]
    if count <= 1:
        return 0.0
    sq_norm_sum = np.sum(flat_array * flat_array)
    sum_vector = np.sum(flat_array, axis=0)
    return float(2 * count * sq_norm_sum - 2 * np.dot(sum_vector, sum_vector))


def compute_frdiv(pred):
    n_samples, n_candidates, seq_len, dim = pred.shape
    if n_candidates <= 1:
        return 0.0

    total = 0.0
    flat_dim = seq_len * dim
    for sample_idx in range(n_samples):
        flat = pred[sample_idx].reshape(n_candidates, flat_dim)
        total += _sum_ordered_pairwise_sq(flat) / (n_candidates * (n_candidates - 1) * flat_dim)
    return total / n_samples


def compute_frdvs(pred):
    n_samples, n_candidates, seq_len, dim = pred.shape
    if n_samples <= 1:
        return 0.0

    flat_dim = seq_len * dim
    total = 0.0
    for cand_idx in range(n_candidates):
        flat = pred[:, cand_idx].reshape(n_samples, flat_dim)
        total += _sum_ordered_pairwise_sq(flat)
    return total / (n_samples * (n_samples - 1) * n_candidates * flat_dim)


def compute_frvar(pred):
    return float(np.mean(np.var(pred, axis=2, ddof=1)))


def shift(x, y, lag):
    if lag > 0:
        return x[lag:], y[:-lag]
    if lag < 0:
        return x[:lag], y[-lag:]
    return x, y


def crosscorr(datax, datay, lag):
    dim = datax.shape[1]
    pcc_list = []
    for dim_idx in range(dim):
        x_shifted, y_shifted = shift(datax[:, dim_idx], datay[:, dim_idx], lag)
        corr = np.corrcoef(x_shifted, y_shifted)[0, 1]
        pcc_list.append(float(corr))
    return float(np.nanmean(np.array(pcc_list, dtype=np.float64)))


def calculate_tlcc(pred_seq, speaker_seq, fps):
    max_lag = int(2 * fps - 1)
    lags = range(-max_lag, max_lag + 1)
    scores = [crosscorr(pred_seq, speaker_seq, lag) for lag in lags]
    scores = np.nan_to_num(np.array(scores, dtype=np.float64), nan=0.0)
    best_idx = int(np.argmax(scores))
    best_lag = list(lags)[best_idx]
    return abs(best_lag)


def compute_frsyn(pred, speaker_gt, fps):
    offsets = []
    for sample_idx in tqdm(range(pred.shape[0]), desc="Computing FRSyn"):
        for cand_idx in range(pred.shape[1]):
            offsets.append(calculate_tlcc(pred[sample_idx, cand_idx], speaker_gt[sample_idx], fps))
    return float(np.mean(np.array(offsets, dtype=np.float64)))


def main():
    args = parse_args()
    selected_metrics = normalize_metric_names(args.metrics)

    prediction = np.load(args.prediction).astype(np.float32)
    if prediction.ndim != 4:
        raise ValueError(f"prediction must have shape [N, K, T, 25], got {prediction.shape}")
    if prediction.shape[-1] != 25:
        raise ValueError(f"prediction last dimension must be 25, got {prediction.shape[-1]}")

    neighbor_matrix = np.load(args.neighbor_matrix)
    if neighbor_matrix.ndim != 2 or neighbor_matrix.shape[0] != neighbor_matrix.shape[1]:
        raise ValueError(f"neighbor matrix must be square, got {neighbor_matrix.shape}")
    if neighbor_matrix.shape[0] != prediction.shape[0]:
        raise ValueError(
            f"neighbor matrix size {neighbor_matrix.shape[0]} does not match prediction N {prediction.shape[0]}"
        )

    speaker_gt, listener_gt = load_ground_truth_arrays(
        data_root=args.data_root,
        split=args.split,
        index_csv=args.index_csv,
        target_len=prediction.shape[2],
    )

    if speaker_gt.shape[0] != prediction.shape[0]:
        raise ValueError(
            f"Expanded sample count from index CSV is {speaker_gt.shape[0]}, but prediction N is {prediction.shape[0]}"
        )

    results = {}

    if "frcorr" in selected_metrics or "frcorr_star" in selected_metrics:
        frcorr = compute_frcorr(prediction, listener_gt, neighbor_matrix)
        results["FRCorr"] = frcorr
        results["FRCorr*"] = frcorr

    if "frdist" in selected_metrics:
        results["FRdist"] = compute_frdist(prediction, listener_gt, neighbor_matrix)

    if "frdiv" in selected_metrics:
        frdiv = compute_frdiv(prediction)
        results["FRDiv"] = frdiv
        results["FRDiv(table)"] = frdiv * 100.0

    if "frdvs" in selected_metrics:
        frdvs = compute_frdvs(prediction)
        results["FRDvs"] = frdvs
        results["FRDvs(table)"] = frdvs * 100.0

    if "frvar" in selected_metrics:
        frvar = compute_frvar(prediction)
        results["FRVar"] = frvar
        results["FRVar(table)"] = frvar * 100.0

    if "frsyn" in selected_metrics:
        results["FRSyn"] = compute_frsyn(prediction, speaker_gt, args.fps)

    print("========== Evaluation Results ==========")
    for key, value in results.items():
        print(f"{key}: {value:.6f}")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
