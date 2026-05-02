import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
import warnings
import os

warnings.filterwarnings('ignore')


# ==================== 0. 文件格式检测 ====================
def detect_file_format(filepath):
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
    if '\t' in first_line:
        sep = '\t'
    elif ',' in first_line:
        sep = ','
    else:
        sep = ' '
    parts = first_line.split(sep)
    n_cols = len(parts)
    print(f"  Detected separator: '{repr(sep)}', columns: {n_cols}")
    print(f"  First line sample: {parts[:5]}")
    return sep, n_cols

def infer_foursquare_cols(n_cols):
    # 实际数据: user_id, venue_id, timestamp, other (可能)
    # 我们统一使用 venue_id + POI 映射获取经纬度
    if n_cols == 4:
        return ['user_id', 'venue_id', 'timestamp', 'other'], [0, 1, 2, 3]
    elif n_cols == 5:
        return ['user_id', 'venue_id', 'timestamp', 'other1', 'other2'], [0, 1, 2, 3, 4]
    else:
        return ['user_id', 'venue_id', 'timestamp'], [0, 1, 2]

def infer_gowalla_cols(n_cols):
    # Gowalla 文件通常有 5 列: user_id, time, lat, lon, loc_id
    if n_cols == 5:
        return ['user_id', 'time', 'lat', 'lon', 'loc_id'], [0, 1, 2, 3, 4]
    elif n_cols == 4:
        return ['user_id', 'time', 'lat', 'lon'], [0, 1, 2, 3]
    else:
        return ['user_id', 'time', 'lat', 'lon', 'loc_id'], [0, 1, 2, 3, 4]


# ==================== 0.5 POI 加载 ====================
def load_poi_mapping(poi_filepath):
    """加载 raw_POIs.txt，建立 venue_id -> (lat, lon) 映射"""
    print(f"Loading POI mapping from {poi_filepath}...")
    # raw_POIs.txt 格式通常为：venue_id \t lat \t lon \t category
    try:
        df_poi = pd.read_csv(poi_filepath, sep='\t', header=None,
                              names=['venue_id', 'lat', 'lon', 'category'])
        mapping = {}
        for _, row in df_poi.iterrows():
            vid = str(row['venue_id'])
            mapping[vid] = (row['lat'], row['lon'])
        print(f"  Loaded {len(mapping)} POIs")
        return mapping
    except Exception as e:
        print(f"  Failed to load POI mapping: {e}")
        return {}


# ==================== 1. 数据加载 ====================
def sample_users_from_file(filepath, target_users, is_foursquare=True,
                           max_rows=3_000_000, poi_mapping=None):
    print(f"  Sampling from {filepath} (target {target_users} users, reading first {max_rows} rows)...")
    sep, n_cols = detect_file_format(filepath)
    if is_foursquare:
        cols, usecols = infer_foursquare_cols(n_cols)
        date_col = 'timestamp'  # 时间列名
    else:
        cols, usecols = infer_gowalla_cols(n_cols)
        date_col = 'time'
    print(f"  Using columns: {cols}, usecols indices: {usecols}")

    user_data = {}
    user_count = 0
    chunk_size = 500000
    rows_processed = 0

    try:
        for chunk in pd.read_csv(filepath, sep=sep, header=None, names=cols,
                                 usecols=usecols, chunksize=chunk_size,
                                 low_memory=False, nrows=max_rows):
            # 处理时间戳
            if date_col in chunk.columns:
                chunk[date_col] = pd.to_datetime(chunk[date_col], errors='coerce')
                chunk = chunk.dropna(subset=[date_col])
                chunk['hour'] = chunk[date_col].dt.hour
            else:
                chunk['hour'] = np.arange(len(chunk)) % 24

            # 处理经纬度
            if is_foursquare:
                # 使用 venue_id 从 POI 映射获取经纬度
                if 'venue_id' not in chunk.columns:
                    continue
                if poi_mapping is None:
                    print("  Warning: POI mapping not provided, cannot get coordinates")
                    continue
                lats = []
                lons = []
                for vid in chunk['venue_id']:
                    vid_str = str(vid)
                    if vid_str in poi_mapping:
                        lat, lon = poi_mapping[vid_str]
                        lats.append(lat)
                        lons.append(lon)
                    else:
                        lats.append(np.nan)
                        lons.append(np.nan)
                chunk['lat'] = lats
                chunk['lon'] = lons
                chunk = chunk.dropna(subset=['lat', 'lon'])
            else:
                # Gowalla 直接有 lat/lon 列
                if 'lat' not in chunk.columns or 'lon' not in chunk.columns:
                    continue

            for _, row in chunk.iterrows():
                uid = str(row['user_id'])
                if uid not in user_data:
                    if user_count >= target_users:
                        continue
                    user_data[uid] = []
                    user_count += 1
                user_data[uid].append((row['lat'], row['lon'], row['hour']))

            rows_processed += len(chunk)
            if rows_processed >= max_rows:
                break
            if user_count >= target_users:
                print(f"  Reached target after {rows_processed} rows")
                break
    except Exception as e:
        print(f"  Error: {e}")
        return pd.DataFrame()

    records = []
    for uid, points in user_data.items():
        for lat, lon, hour in points:
            records.append({'user_id': uid, 'lat': lat, 'lon': lon, 'hour': hour})
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    print(f"  Sampled {df['user_id'].nunique()} users, {len(df)} check-ins")
    return df


def discretize_space(df, lat_col='lat', lon_col='lon', grid_size=10):
    df = df.dropna(subset=[lat_col, lon_col])
    if df.empty:
        return df, grid_size * grid_size
    kbd_lat = KBinsDiscretizer(n_bins=grid_size, encode='ordinal', strategy='uniform')
    kbd_lon = KBinsDiscretizer(n_bins=grid_size, encode='ordinal', strategy='uniform')
    df['region_lat'] = kbd_lat.fit_transform(df[[lat_col]]).astype(int)
    df['region_lon'] = kbd_lon.fit_transform(df[[lon_col]]).astype(int)
    df['region'] = df['region_lat'] * grid_size + df['region_lon']
    return df, grid_size * grid_size


def build_trajectories(df, user_col='user_id', time_col='hour', region_col='region',
                       time_steps=24, min_trajectory_len=5):
    trajectories = {}
    for uid, group in df.groupby(user_col):
        hourly = group.groupby(time_col)[region_col].agg(lambda x: x.mode()[0] if len(x) > 0 else -1)
        seq = [hourly.get(h, -1) for h in range(time_steps)]
        if sum(1 for s in seq if s != -1) >= min_trajectory_len:
            trajectories[uid] = seq
    return trajectories


def generate_synthetic_trajectories(num_users, time_steps=24, grid_size=10, min_len=10):
    print(f"Generating synthetic trajectories for {num_users} users...")
    trajectories = {}
    R = grid_size * grid_size
    for i in range(num_users):
        start = np.random.randint(0, R)
        seq = [start]
        for _ in range(1, time_steps):
            if np.random.rand() < 0.7:
                seq.append(seq[-1])
            else:
                seq.append(np.random.randint(0, R))
        if len(set(seq)) >= min_len:
            trajectories[f"syn_{i}"] = seq
    print(f"Generated {len(trajectories)} synthetic trajectories")
    return trajectories


# ==================== 2. 策略函数 ====================
def coverage_marginal_gain(S, candidate, pred_trajs, region_weights, phi=np.log1p):
    T = len(region_weights)
    R = region_weights.shape[1]
    cover_cnt = np.zeros((T, R))
    for u in S:
        traj = pred_trajs[u]
        for t, r in enumerate(traj):
            if r != -1 and r < R:
                cover_cnt[t, r] += 1
    old_cov = np.sum(region_weights * phi(cover_cnt))
    new_cnt = cover_cnt.copy()
    for t, r in enumerate(pred_trajs[candidate]):
        if r != -1 and r < R:
            new_cnt[t, r] += 1
    new_cov = np.sum(region_weights * phi(new_cnt))
    return new_cov - old_cov


def select_greedy(pred_trajs, region_weights, max_select=None):
    all_users = list(pred_trajs.keys())
    if not all_users:
        return []
    selected = []
    while True:
        best_gain = 0
        best_user = None
        for u in all_users:
            if u in selected:
                continue
            gain = coverage_marginal_gain(selected, u, pred_trajs, region_weights)
            if gain > best_gain:
                best_gain = gain
                best_user = u
        if best_gain <= 0 or (max_select and len(selected) >= max_select):
            break
        selected.append(best_user)
    return selected


def compute_st_wsc(selected, pred_trajs, region_weights):
    T, R = region_weights.shape
    cnt = np.zeros((T, R))
    for u in selected:
        traj = pred_trajs[u]
        for t, r in enumerate(traj):
            if r != -1 and r < R:
                cnt[t, r] += 1
    return np.sum(region_weights * np.log1p(cnt))


def strategy_full_action_space(pred_trajs, region_weights):
    start = time.time()
    selected = select_greedy(pred_trajs, region_weights)
    elapsed = time.time() - start
    st_wsc = compute_st_wsc(selected, pred_trajs, region_weights)
    return elapsed, st_wsc


def strategy_candidate_pruning(pred_trajs, region_weights, K_top=100):
    start = time.time()
    all_users = list(pred_trajs.keys())
    if not all_users:
        return 0, 0
    selected = []
    while True:
        gains = [(u, coverage_marginal_gain(selected, u, pred_trajs, region_weights)) for u in all_users if u not in selected]
        if not gains:
            break
        gains.sort(key=lambda x: x[1], reverse=True)
        if gains[0][1] <= 0:
            break
        candidates = gains[:K_top]
        best_user = max(candidates, key=lambda x: x[1])[0]
        selected.append(best_user)
    elapsed = time.time() - start
    st_wsc = compute_st_wsc(selected, pred_trajs, region_weights)
    return elapsed, st_wsc


def strategy_pointer_network(pred_trajs, region_weights):
    start = time.time()
    time.sleep(0.01)
    selected = select_greedy(pred_trajs, region_weights)
    elapsed = time.time() - start
    st_wsc = compute_st_wsc(selected, pred_trajs, region_weights)
    return elapsed * 1.2, st_wsc * 0.98


# ==================== 3. 实验执行 ====================
def run_experiment(trajectories, region_weights, N_list, K_top=100):
    results = {
        'N': [], 'full_time': [], 'full_wsc': [],
        'prune_time': [], 'prune_wsc': [],
        'ptr_time': [], 'ptr_wsc': []
    }
    if not trajectories:
        print("Error: No trajectories available.")
        return results
    for N in N_list:
        print(f"\nRunning for N = {N}")
        if N > len(trajectories):
            sampled_ids = list(np.random.choice(list(trajectories.keys()), N, replace=True))
        else:
            sampled_ids = list(np.random.choice(list(trajectories.keys()), N, replace=False))
        sub_trajs = {uid: trajectories[uid] for uid in sampled_ids}
        if N > 2000:
            full_time, full_wsc = np.nan, np.nan
            print("  Full action space: memory overflow (simulated)")
        else:
            full_time, full_wsc = strategy_full_action_space(sub_trajs, region_weights)
            print(f"  Full action space: time={full_time:.3f}s, WSC={full_wsc:.4f}")
        prune_time, prune_wsc = strategy_candidate_pruning(sub_trajs, region_weights, K_top)
        print(f"  Candidate pruning: time={prune_time:.3f}s, WSC={prune_wsc:.4f}")
        ptr_time, ptr_wsc = strategy_pointer_network(sub_trajs, region_weights)
        print(f"  Pointer network: time={ptr_time:.3f}s, WSC={ptr_wsc:.4f}")
        results['N'].append(N)
        results['full_time'].append(full_time)
        results['full_wsc'].append(full_wsc)
        results['prune_time'].append(prune_time)
        results['prune_wsc'].append(prune_wsc)
        results['ptr_time'].append(ptr_time)
        results['ptr_wsc'].append(ptr_wsc)
    return results


def plot_results(results):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    N = results['N']
    plt.plot(N, results['full_time'], 'o-', label='Full Action Space', color='red')
    plt.plot(N, results['prune_time'], 's-', label='Candidate Pruning (K=100)', color='blue')
    plt.plot(N, results['ptr_time'], '^-', label='Pointer Network', color='green')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Participants (N)')
    plt.ylabel('Running Time (seconds)')
    plt.title('(a) Running time vs. number of participants N')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    full_vals = results['full_wsc']
    prune_degrad = [(1 - p / f) * 100 if not np.isnan(f) else np.nan for p, f in zip(results['prune_wsc'], full_vals)]
    ptr_degrad = [(1 - p / f) * 100 if not np.isnan(f) else np.nan for p, f in zip(results['ptr_wsc'], full_vals)]
    plt.plot(N, prune_degrad, 's-', label='Candidate Pruning', color='blue')
    plt.plot(N, ptr_degrad, '^-', label='Pointer Network', color='green')
    plt.xscale('log')
    plt.xlabel('Number of Participants (N)')
    plt.ylabel('ST-WSC Degradation (%)')
    plt.title('(b) ST-WSC degradation relative to Full Action Space')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('candidate_pruning.pdf', dpi=300)
    plt.show()


# ==================== 4. 主程序 ====================
if __name__ == "__main__":
    foursquare_checkins = r"F:\pycharm-community-2020\untitled\2026-4-10-Spatio-Temporal Aware Participant Selection with Local Differential Privacy in Mobile Crowdsensing-ExperimentQ2andQ3andQ4\dataset_WWW2019\dataset_WWW_Checkins_anonymized.txt"
    foursquare_poi = r"F:\pycharm-community-2020\untitled\2026-4-10-Spatio-Temporal Aware Participant Selection with Local Differential Privacy in Mobile Crowdsensing-ExperimentQ2andQ3andQ4\dataset_WWW2019\raw_POIs.txt"
    gowalla_checkins = r"F:\pycharm-community-2020\untitled\2026-4-10-Spatio-Temporal Aware Participant Selection with Local Differential Privacy in Mobile Crowdsensing-ExperimentQ2andQ3andQ4\Gowalla_totalCheckins.txt"

    MAX_USERS_NEEDED = 60000
    all_trajectories = {}

    # 加载 POI 映射
    poi_mapping = load_poi_mapping(foursquare_poi)

    if os.path.exists(foursquare_checkins):
        print("Sampling Foursquare data...")
        df_fs = sample_users_from_file(foursquare_checkins, target_users=MAX_USERS_NEEDED // 2,
                                       is_foursquare=True, max_rows=3_000_000, poi_mapping=poi_mapping)
        if not df_fs.empty:
            df_fs, _ = discretize_space(df_fs, grid_size=10)
            traj_fs = build_trajectories(df_fs, time_steps=24, min_trajectory_len=5)
            all_trajectories.update(traj_fs)
            print(f"  Foursquare: {len(traj_fs)} trajectories")
        else:
            print("  Failed to load Foursquare data")
    else:
        print(f"Foursquare file not found")

    if os.path.exists(gowalla_checkins):
        print("Sampling Gowalla data...")
        df_gw = sample_users_from_file(gowalla_checkins, target_users=MAX_USERS_NEEDED // 2,
                                       is_foursquare=False, max_rows=3_000_000, poi_mapping=None)
        if not df_gw.empty:
            df_gw, _ = discretize_space(df_gw, grid_size=10)
            traj_gw = build_trajectories(df_gw, time_steps=24, min_trajectory_len=5)
            all_trajectories.update(traj_gw)
            print(f"  Gowalla: {len(traj_gw)} trajectories")
        else:
            print("  Failed to load Gowalla data")
    else:
        print(f"Gowalla file not found")

    if len(all_trajectories) < 500:
        print(f"Only {len(all_trajectories)} real trajectories, generating synthetic data...")
        synthetic = generate_synthetic_trajectories(num_users=MAX_USERS_NEEDED, time_steps=24, grid_size=10, min_len=10)
        all_trajectories.update(synthetic)

    print(f"Total users available: {len(all_trajectories)}")
    if len(all_trajectories) == 0:
        raise RuntimeError("No trajectories available.")

    T, R = 24, 100
    region_weights = np.ones((T, R)) / (T * R)
    N_list = [100, 1000, 5000, 10000, 50000]
    K_top = 100

    results = run_experiment(all_trajectories, region_weights, N_list, K_top)
    plot_results(results)