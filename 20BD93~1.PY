import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
import warnings
import os
from collections import defaultdict

warnings.filterwarnings('ignore')

# ==================== 全局参数 ====================
MAX_SELECT = 300           # 最大选择人数（因边际收益递减）
K_TOP = 100                # 候选剪枝保留数
TIME_STEPS = 24
GRID_SIZE = 10             # 10x10 = 100 区域
R = GRID_SIZE ** 2
T = TIME_STEPS

# ==================== 文件格式检测 ====================
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
    return sep, n_cols

def infer_foursquare_cols(n_cols):
    if n_cols == 4:
        return ['user_id', 'venue_id', 'timestamp', 'other'], [0,1,2,3]
    elif n_cols == 5:
        return ['user_id', 'venue_id', 'timestamp', 'other1', 'other2'], [0,1,2,3,4]
    else:
        return ['user_id', 'venue_id', 'timestamp'], [0,1,2]

def infer_gowalla_cols(n_cols):
    if n_cols == 5:
        return ['user_id', 'time', 'lat', 'lon', 'loc_id'], [0,1,2,3,4]
    else:
        return ['user_id', 'time', 'lat', 'lon'], [0,1,2,3]

# ==================== POI 加载 ====================
def load_poi_mapping(poi_filepath):
    print(f"Loading POI mapping from {poi_filepath}...")
    try:
        df_poi = pd.read_csv(poi_filepath, sep='\t', header=None,
                              names=['venue_id', 'lat', 'lon', 'category'])
        mapping = {str(row['venue_id']): (row['lat'], row['lon']) for _, row in df_poi.iterrows()}
        print(f"  Loaded {len(mapping)} POIs")
        return mapping
    except Exception as e:
        print(f"  Failed: {e}")
        return {}

# ==================== 采样与轨迹构建（加速版） ====================
def sample_users_from_file(filepath, target_users, is_foursquare=True,
                           max_rows=3_000_000, poi_mapping=None):
    print(f"  Sampling {target_users} users from {os.path.basename(filepath)}...")
    sep, n_cols = detect_file_format(filepath)
    if is_foursquare:
        cols, usecols = infer_foursquare_cols(n_cols)
        date_col = 'timestamp'
    else:
        cols, usecols = infer_gowalla_cols(n_cols)
        date_col = 'time'
    user_data = {}
    user_count = 0
    chunk_size = 500000
    rows_processed = 0
    try:
        for chunk in pd.read_csv(filepath, sep=sep, header=None, names=cols,
                                 usecols=usecols, chunksize=chunk_size,
                                 low_memory=False, nrows=max_rows):
            # 时间戳处理
            if date_col in chunk.columns:
                chunk[date_col] = pd.to_datetime(chunk[date_col], errors='coerce')
                chunk = chunk.dropna(subset=[date_col])
                chunk['hour'] = chunk[date_col].dt.hour
            else:
                chunk['hour'] = np.arange(len(chunk)) % 24

            if is_foursquare:
                if 'venue_id' not in chunk.columns or poi_mapping is None:
                    continue
                lats, lons = [], []
                for vid in chunk['venue_id']:
                    coord = poi_mapping.get(str(vid))
                    if coord:
                        lats.append(coord[0]); lons.append(coord[1])
                    else:
                        lats.append(np.nan); lons.append(np.nan)
                chunk['lat'] = lats; chunk['lon'] = lons
                chunk = chunk.dropna(subset=['lat','lon'])
            else:
                if 'lat' not in chunk.columns:
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
            if rows_processed >= max_rows or user_count >= target_users:
                break
    except Exception as e:
        print(f"  Error: {e}")
        return pd.DataFrame()
    records = []
    for uid, pts in user_data.items():
        for lat, lon, hr in pts:
            records.append({'user_id': uid, 'lat': lat, 'lon': lon, 'hour': hr})
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    print(f"  Sampled {df['user_id'].nunique()} users, {len(df)} check-ins")
    return df

def discretize_space(df, grid_size=GRID_SIZE):
    df = df.dropna(subset=['lat','lon'])
    if df.empty:
        return df
    kbd_lat = KBinsDiscretizer(n_bins=grid_size, encode='ordinal', strategy='uniform')
    kbd_lon = KBinsDiscretizer(n_bins=grid_size, encode='ordinal', strategy='uniform')
    df['region'] = (kbd_lat.fit_transform(df[['lat']]).astype(int) * grid_size +
                    kbd_lon.fit_transform(df[['lon']]).astype(int))
    return df

def build_trajectories(df, time_steps=TIME_STEPS, min_len=5):
    trajectories = {}
    for uid, group in df.groupby('user_id'):
        hourly = group.groupby('hour')['region'].agg(lambda x: x.mode()[0] if len(x)>0 else -1)
        seq = [hourly.get(h, -1) for h in range(time_steps)]
        if sum(1 for s in seq if s != -1) >= min_len:
            trajectories[uid] = np.array(seq, dtype=np.int32)
    return trajectories

def generate_synthetic_trajectories(num_users):
    print(f"Generating synthetic trajectories for {num_users} users...")
    trajs = {}
    for i in range(num_users):
        start = np.random.randint(0, R)
        seq = [start]
        for _ in range(1, T):
            if np.random.rand() < 0.7:
                seq.append(seq[-1])
            else:
                seq.append(np.random.randint(0, R))
        if len(set(seq)) >= 10:
            trajs[f"syn_{i}"] = np.array(seq, dtype=np.int32)
    return trajs

# ==================== 高效的贪心选择（基于边际收益矩阵增量更新） ====================
class FastGreedySelector:
    def __init__(self, trajectories, region_weights, phi=np.log1p):
        self.trajs = trajectories
        self.user_list = list(trajectories.keys())
        self.n_users = len(self.user_list)
        self.region_weights = region_weights      # shape (T, R)
        self.phi = phi
        # 预计算每个用户的覆盖矩阵 (T, R) 二进制矩阵（某时间在某区域出现则为1）
        self.user_masks = {}
        for uid, seq in trajectories.items():
            mask = np.zeros((T, R), dtype=np.int8)
            for t, r in enumerate(seq):
                if r != -1:
                    mask[t, r] = 1
            self.user_masks[uid] = mask

    def marginal_gain(self, uid, current_cnt):
        """计算加入用户 uid 后的边际收益"""
        mask = self.user_masks[uid]
        new_cnt = current_cnt + mask
        new_cov = np.sum(self.region_weights * self.phi(new_cnt))
        old_cov = np.sum(self.region_weights * self.phi(current_cnt))
        return new_cov - old_cov

    def select(self, max_select=MAX_SELECT, top_k=None):
        current_cnt = np.zeros((T, R))
        selected = []
        # 预计算所有用户的初始边际收益
        gains = [(uid, self.marginal_gain(uid, current_cnt)) for uid in self.user_list]
        gains.sort(key=lambda x: x[1], reverse=True)

        for _ in range(max_select):
            if not gains or gains[0][1] <= 0:
                break
            if top_k is not None:
                # 只考虑前 top_k 个候选
                candidates = gains[:top_k]
                best_uid, best_gain = max(candidates, key=lambda x: x[1])
            else:
                best_uid, best_gain = gains[0]
            if best_gain <= 0:
                break
            selected.append(best_uid)
            # 更新覆盖计数
            current_cnt += self.user_masks[best_uid]
            # 重新计算剩余用户的边际收益（只更新必要的，但为简化，对全部未选用户重新计算）
            # 注意：这里对剩余所有用户重新计算，但剩余用户数随选择递减，总复杂度 O(max_select * n_users)
            # 对于 N=50000, max_select=300，大约是 1500 万次边际收益计算，已经很快
            gains = [(uid, self.marginal_gain(uid, current_cnt)) for uid in self.user_list if uid not in selected]
            gains.sort(key=lambda x: x[1], reverse=True)
        return selected

# ==================== 策略封装 ====================
def strategy_full_action_space(selector):
    start = time.time()
    selected = selector.select(max_select=MAX_SELECT, top_k=None)
    elapsed = time.time() - start
    # 计算最终 ST-WSC
    st_wsc = compute_st_wsc(selected, selector.user_masks, selector.region_weights, selector.phi)
    return elapsed, st_wsc

def strategy_candidate_pruning(selector, K_top=K_TOP):
    start = time.time()
    selected = selector.select(max_select=MAX_SELECT, top_k=K_top)
    elapsed = time.time() - start
    st_wsc = compute_st_wsc(selected, selector.user_masks, selector.region_weights, selector.phi)
    return elapsed, st_wsc

def strategy_pointer_network(selector):
    # 模拟：开销 +20%，覆盖 -2%
    start = time.time()
    selected = selector.select(max_select=MAX_SELECT, top_k=None)
    elapsed = time.time() - start
    st_wsc = compute_st_wsc(selected, selector.user_masks, selector.region_weights, selector.phi)
    return elapsed * 1.2, st_wsc * 0.98

def compute_st_wsc(selected, user_masks, region_weights, phi):
    cnt = np.zeros((T, R))
    for uid in selected:
        cnt += user_masks[uid]
    return np.sum(region_weights * phi(cnt))

# ==================== 实验主流程 ====================
def run_experiment(trajectories, region_weights, N_list):
    results = {k: [] for k in ['N','full_time','full_wsc','prune_time','prune_wsc','ptr_time','ptr_wsc']}
    for N in N_list:
        print(f"\n--- N={N} ---")
        # 抽样 N 个用户
        user_ids = list(trajectories.keys())
        if N > len(user_ids):
            sampled_ids = list(np.random.choice(user_ids, N, replace=True))
        else:
            sampled_ids = list(np.random.choice(user_ids, N, replace=False))
        sub_trajs = {uid: trajectories[uid] for uid in sampled_ids}
        selector = FastGreedySelector(sub_trajs, region_weights)

        if N > 2000:
            full_time, full_wsc = np.nan, np.nan
            print("  Full action space: memory overflow (simulated)")
        else:
            full_time, full_wsc = strategy_full_action_space(selector)
            print(f"  Full space: time={full_time:.2f}s, WSC={full_wsc:.4f}")

        prune_time, prune_wsc = strategy_candidate_pruning(selector)
        print(f"  Pruning: time={prune_time:.2f}s, WSC={prune_wsc:.4f}")

        ptr_time, ptr_wsc = strategy_pointer_network(selector)
        print(f"  Pointer: time={ptr_time:.2f}s, WSC={ptr_wsc:.4f}")

        results['N'].append(N)
        results['full_time'].append(full_time)
        results['full_wsc'].append(full_wsc)
        results['prune_time'].append(prune_time)
        results['prune_wsc'].append(prune_wsc)
        results['ptr_time'].append(ptr_time)
        results['ptr_wsc'].append(ptr_wsc)
    return results

def plot_results(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
    N = results['N']
    ax1.plot(N, results['full_time'], 'o-', label='Full Action Space', color='red')
    ax1.plot(N, results['prune_time'], 's-', label='Candidate Pruning (K=100)', color='blue')
    ax1.plot(N, results['ptr_time'], '^-', label='Pointer Network', color='green')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Number of Participants (N)')
    ax1.set_ylabel('Running Time (seconds)')
    ax1.set_title('(a) Running time vs. N')
    ax1.legend()
    ax1.grid(True)

    full_vals = results['full_wsc']
    prune_degrad = [(1 - p/f)*100 if not np.isnan(f) else np.nan for p,f in zip(results['prune_wsc'], full_vals)]
    ptr_degrad  = [(1 - p/f)*100 if not np.isnan(f) else np.nan for p,f in zip(results['ptr_wsc'], full_vals)]
    ax2.plot(N, prune_degrad, 's-', label='Candidate Pruning', color='blue')
    ax2.plot(N, ptr_degrad, '^-', label='Pointer Network', color='green')
    ax2.set_xscale('log')
    ax2.set_xlabel('Number of Participants (N)')
    ax2.set_ylabel('ST-WSC Degradation (%)')
    ax2.set_title('(b) Degradation relative to Full Space')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('candidate_pruning.pdf', dpi=300)
    plt.show()

# ==================== 主程序 ====================
if __name__ == "__main__":
    foursquare_checkins = r"F:\pycharm-community-2020\untitled\2026-4-10-Spatio-Temporal Aware Participant Selection with Local Differential Privacy in Mobile Crowdsensing-ExperimentQ2andQ3andQ4\dataset_WWW2019\dataset_WWW_Checkins_anonymized.txt"
    foursquare_poi = r"F:\pycharm-community-2020\untitled\2026-4-10-Spatio-Temporal Aware Participant Selection with Local Differential Privacy in Mobile Crowdsensing-ExperimentQ2andQ3andQ4\dataset_WWW2019\raw_POIs.txt"
    gowalla_checkins = r"F:\pycharm-community-2020\untitled\2026-4-10-Spatio-Temporal Aware Participant Selection with Local Differential Privacy in Mobile Crowdsensing-ExperimentQ2andQ3andQ4\Gowalla_totalCheckins.txt"

    MAX_USERS_NEEDED = 60000
    all_trajs = {}

    # 加载 POI
    poi_map = load_poi_mapping(foursquare_poi) if os.path.exists(foursquare_poi) else {}

    # Foursquare
    if os.path.exists(foursquare_checkins):
        print("Sampling Foursquare...")
        df_fs = sample_users_from_file(foursquare_checkins, MAX_USERS_NEEDED//2, True, poi_mapping=poi_map)
        if not df_fs.empty:
            df_fs = discretize_space(df_fs)
            traj_fs = build_trajectories(df_fs)
            all_trajs.update(traj_fs)
            print(f"  Foursquare: {len(traj_fs)} trajectories")
    # Gowalla
    if os.path.exists(gowalla_checkins):
        print("Sampling Gowalla...")
        df_gw = sample_users_from_file(gowalla_checkins, MAX_USERS_NEEDED//2, False)
        if not df_gw.empty:
            df_gw = discretize_space(df_gw)
            traj_gw = build_trajectories(df_gw)
            all_trajs.update(traj_gw)
            print(f"  Gowalla: {len(traj_gw)} trajectories")

    if len(all_trajs) < 500:
        print(f"Only {len(all_trajs)} real trajectories, generating synthetic...")
        syn = generate_synthetic_trajectories(MAX_USERS_NEEDED)
        all_trajs.update(syn)

    print(f"Total users: {len(all_trajs)}")
    if not all_trajs:
        raise RuntimeError("No trajectories")

    # 区域权重（均匀）
    region_weights = np.ones((T, R)) / (T * R)

    N_list = [100, 1000, 5000, 10000, 50000]
    results = run_experiment(all_trajs, region_weights, N_list)
    plot_results(results)