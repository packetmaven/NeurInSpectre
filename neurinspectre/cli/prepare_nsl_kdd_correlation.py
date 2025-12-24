"""
Prepare real attack data (NSL-KDD) for correlation analysis.

Downloads NSL-KDD KDDTrain+.txt and KDDTest+.txt, encodes categorical
features, standardizes, and aggregates into time×feature matrices for:
- primary: one selected attack family (default: neptune)
- secondary: benign normal traffic

Outputs:
- {out_dir}/primary.npy
- {out_dir}/secondary.npy
"""

import argparse
import os
import urllib.request
from pathlib import Path


NSL_TRAIN_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
NSL_TEST_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt"

# NSL-KDD feature names (41 features) + label + difficulty
NSL_COLUMNS = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent',
    'hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate',
    'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count',
    'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
    'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty'
]


def _download_if_missing(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urllib.request.urlretrieve(url, dest.as_posix())
    return dest


def _load_and_encode(csv_path: Path):
    import pandas as pd
    df = pd.read_csv(csv_path.as_posix(), header=None, names=NSL_COLUMNS)
    # Separate label, drop difficulty
    labels = df['label'].astype(str)
    df = df.drop(columns=['difficulty'])
    # One-hot encode categorical columns
    cat_cols = ['protocol_type','service','flag','label']
    df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    return df_enc.astype('float32'), labels


def _standardize(df_enc):
    import numpy as np
    arr = df_enc.to_numpy(dtype='float32', copy=False)
    mu = arr.mean(axis=0, keepdims=True)
    sd = arr.std(axis=0, keepdims=True) + 1e-6
    return (arr - mu) / sd


def _aggregate_time_series(arr, window: int):
    import numpy as np
    n = arr.shape[0]
    if n < window:
        # pad by repetition to at least one window
        reps = int((window + n - 1) // n)
        arr = np.tile(arr, (reps, 1))
        n = arr.shape[0]
    num_win = n // window
    arr = arr[:num_win * window]
    series = arr.reshape(num_win, window, arr.shape[1]).mean(axis=1)
    return series


def main():
    p = argparse.ArgumentParser(description='Prepare NSL-KDD for NeurInSpectre correlation')
    p.add_argument('--out-dir', required=True, help='Output directory for .npy files')
    p.add_argument('--attack', default='neptune', help='Attack family for primary time series')
    p.add_argument('--window', type=int, default=128, help='Aggregation window size')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    train_path = _download_if_missing(NSL_TRAIN_URL, out_dir / 'KDDTrain+.txt')
    test_path = _download_if_missing(NSL_TEST_URL, out_dir / 'KDDTest+.txt')

    try:
        import pandas  # noqa: F401
    except Exception:
        raise SystemExit('pandas is required. Install with: pip install pandas')

    # Load and encode
    train_enc, train_labels = _load_and_encode(train_path)
    test_enc, test_labels = _load_and_encode(test_path)

    # Align columns (union of one-hot categories) to avoid shape mismatches
    all_cols = sorted(set(train_enc.columns) | set(test_enc.columns))
    train_enc = train_enc.reindex(columns=all_cols, fill_value=0.0)
    test_enc = test_enc.reindex(columns=all_cols, fill_value=0.0)

    import numpy as np
    # Concatenate, then standardize across the combined distribution
    all_df = __import__('pandas').concat([train_enc, test_enc], axis=0, ignore_index=True)
    all_arr = _standardize(all_df)
    all_labels = np.concatenate([train_labels.to_numpy(), test_labels.to_numpy()], axis=0)

    # Primary: rows matching selected attack family
    attack = args.attack.lower()
    is_attack = np.array([attack in str(lbl).lower() and 'normal' not in str(lbl).lower() for lbl in all_labels])
    prim_arr = all_arr[is_attack]
    # Secondary: normal traffic
    sec_arr = all_arr[np.array([str(lbl).lower() == 'normal' for lbl in all_labels])]

    if prim_arr.size == 0:
        raise SystemExit(f'No rows found for attack family "{args.attack}". Try another (e.g., neptune, smurf, teardrop, satan, ipsweep).')
    if sec_arr.size == 0:
        raise SystemExit('No rows found for normal traffic in dataset.')

    prim_ts = _aggregate_time_series(prim_arr, args.window)
    sec_ts = _aggregate_time_series(sec_arr, args.window)

    # Match timesteps
    T = min(prim_ts.shape[0], sec_ts.shape[0])
    prim_ts = prim_ts[:T]
    sec_ts = sec_ts[:T]

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save((out_dir / 'primary.npy').as_posix(), prim_ts.astype('float32'))
    np.save((out_dir / 'secondary.npy').as_posix(), sec_ts.astype('float32'))
    print((out_dir / 'primary.npy').as_posix())
    print((out_dir / 'secondary.npy').as_posix())


if __name__ == '__main__':
    main()


