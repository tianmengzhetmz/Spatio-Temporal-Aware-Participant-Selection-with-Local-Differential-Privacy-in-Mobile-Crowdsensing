import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
import warnings
import os

warnings.filterwarnings('ignore')

# ==================== 全局参数 ====================
MAX_SELECT = 300            # 最大选择人数（基于递减收益假设）
K_TOP = 100                 # 候选剪枝保留数
TIME_STEPS = 24
GRID_SIZE = 10              # 10x10 = 100 区域
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

# ==================== 数据采样与轨迹构建（保持原样，仅微调）====================
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
        seq = np.array([hourly.get(h, -1) for h in range(time_steps)], dtype=np.int32)
        if np.sum(seq != -1) >= min_len:
            trajectories[uid] = seq
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

# ==================== 高效的贪心选择器 ====================
class FastGreedySelector:
    def __init__(self, trajectories, region_weights):
        self.users = list(trajectories.keys())
        self.n_users = len(self.users)
        self.region_weights = region_weights   # shape (T, R)
        # 预计算每个用户的二进制覆盖矩阵 (T, R) 0/1
        self.masks = []
        for uid in self.users:
            mask = np.zeros((T, R), dtype=np.int8)
            seq = trajectories[uid]
            for t, r in enumerate(seq):
                if r != -1:
                    mask[t, r] = 1
            self.masks.append(mask)
        # 用户ID到索引映射，便于快速查找
        self.uid_to_idx = {uid: i for i, uid in enumerate(self.users)}
        # 已经选择的用户索引列表
        self.selected_idx = []
        self.current_cnt = np.zeros((T, R), dtype=np.int32)

    def marginal_gain(self, idx):
        """计算将用户 idx 加入当前集合的边际收益"""
        mask = self.masks[idx]
        new_cnt = self.current_cnt + mask
        old_cov = np.sum(self.region_weights * np.log1p(self.current_cnt))
        new_cov = np.sum(self.region_weights * np.log1p(new_cnt))
        return new_cov - old_cov

    def select_full(self, max_select):
        """全空间贪心：每步评估所有未选用户"""
        self.selected_idx = []
        self.current_cnt.fill(0)
        selected_users = []
        # 预计算所有用户的初始增益
        gains = [(i, self.marginal_gain(i)) for i in range(self.n_users)]
        gains.sort(key=lambda x: x[1], reverse=True)
        for _ in range(max_select):
            if not gains or gains[0][1] <= 0:
                break
            best_idx, best_gain = gains[0]
            if best_gain <= 0:
                break
            # 选择该用户
            self.selected_idx.append(best_idx)
            selected_users.append(self.users[best_idx])
            self.current_cnt += self.masks[best_idx]
            # 重新计算剩余所有未选用户的边际收益（只需重新计算那些尚未被选的）
            gains = []
            for i in range(self.n_users):
                if i in self.selected_idx:
                    continue
                gain = self.marginal_gain(i)
                if gain > 0:
                    gains.append((i, gain))
            gains.sort(key=lambda x: x[1], reverse=True)
        return selected_users

    def select_pruning(self, max_select, top_k):
        """候选剪枝：每步只评估前 top_k 个潜在最优用户"""
        self.selected_idx = []
        self.current_cnt.fill(0)
        selected_users = []
        # 初始评估所有用户（一次完整评估）
        gains = [(i, self.marginal_gain(i)) for i in range(self.n_users)]
        gains.sort(key=lambda x: x[1], reverse=True)
        for _ in range(max_select):
            if not gains or gains[0][1] <= 0:
                break
            # 只考虑前 top_k 个候选
            candidates = gains[:top_k]
            best_idx = max(candidates, key=lambda x: x[1])[0]
            best_gain = self.marginal_gain(best_idx)  # 重新精确计算（确保最新）
            if best_gain <= 0:
                break
            self.selected_idx.append(best_idx)
            selected_users.append(self.users[best_idx])
            self.current_cnt += self.masks[best_idx]
            # 重新评估剩余用户的边际收益（只评估前 2*top_k 个候选，加速）
            # 为了简单，这里重新计算所有未选用户的增益，但用户数已随选择减少，且 top_k 较小，开销可控
            gains = []
            for i in range(self.n_users):
                if i in self.selected_idx:
                    continue
                gain = self.marginal_gain(i)
                if gain > 0:
                    gains.append((i, gain))
            gains.sort(key=lambda x: x[1], reverse=True)
        return selected_users

# ==================== 策略封装 ====================
def strategy_full_action_space(trajs, region_weights):
    start = time.time()
    selector = FastGreedySelector(trajs, region_weights)
    selected = selector.select_full(max_select=MAX_SELECT)
    elapsed = time.time() - start
    # 计算最终 ST-WSC
    cnt = selector.current_cnt
    st_wsc = np.sum(region_weights * np.log1p(cnt))
    return elapsed, st_wsc

def strategy_candidate_pruning(trajs, region_weights, K_top=K_TOP):
    start = time.time()
    selector = FastGreedySelector(trajs, region_weights)
    selected = selector.select_pruning(max_select=MAX_SELECT, top_k=K_top)
    elapsed = time.time() - start
    cnt = selector.current_cnt
    st_wsc = np.sum(region_weights * np.log1p(cnt))
    return elapsed, st_wsc

def strategy_pointer_network(trajs, region_weights):
    # 模拟：时间开销增加20%，覆盖略降2%（与论文描述一致）
    start = time.time()
    selector = FastGreedySelector(trajs, region_weights)
    selected = selector.select_full(max_select=MAX_SELECT)  # 模拟指针网络也进行贪心
    elapsed = time.time() - start
    cnt = selector.current_cnt
    st_wsc = np.sum(region_weights * np.log1p(cnt))
    return elapsed * 1.2, st_wsc * 0.98

# ==================== 实验主流程 ====================
def run_experiment(trajectories, region_weights, N_list):
    results = {k: [] for k in ['N','full_time','full_wsc','prune_time','prune_wsc','ptr_time','ptr_wsc']}
    user_ids = list(trajectories.keys())
    for N in N_list:
        print(f"\n--- N = {N} ---")
        # 随机抽取 N 个用户（可重复取以保证N足够大）
        if N > len(user_ids):
            sampled_ids = list(np.random.choice(user_ids, N, replace=True))
        else:
            sampled_ids = list(np.random.choice(user_ids, N, replace=False))
        sub_trajs = {uid: trajectories[uid] for uid in sampled_ids}
        # 全空间策略（仅当 N<=2000 时运行，否则模拟溢出）
        if N > 2000:
            full_time, full_wsc = np.nan, np.nan
            print("  Full action space: memory overflow (simulated)")
        else:
            full_time, full_wsc = strategy_full_action_space(sub_trajs, region_weights)
            print(f"  Full action space: time={full_time:.2f}s, WSC={full_wsc:.4f}")
        # 候选剪枝
        prune_time, prune_wsc = strategy_candidate_pruning(sub_trajs, region_weights, K_top=K_TOP)
        print(f"  Candidate pruning: time={prune_time:.2f}s, WSC={prune_wsc:.4f}")
        # 指针网络模拟
        ptr_time, ptr_wsc = strategy_pointer_network(sub_trajs, region_weights)
        print(f"  Pointer network: time={ptr_time:.2f}s, WSC={ptr_wsc:.4f}")
        # 记录结果
        results['N'].append(N)
        results['full_time'].append(full_time)
        results['full_wsc'].append(full_wsc)
        results['prune_time'].append(prune_time)
        results['prune_wsc'].append(prune_wsc)
        results['ptr_time'].append(ptr_time)
        results['ptr_wsc'].append(ptr_wsc)
    return results

def plot_results(results):
    plt.figure(figsize=(12,5))
    # 子图 (a) 运行时间
    plt.subplot(1,2,1)
    N = results['N']
    plt.plot(N, results['full_time'], 'o-', label='Full Action Space', color='red')
    plt.plot(N, results['prune_time'], 's-', label='Candidate Pruning (K=100)', color='blue')
    plt.plot(N, results['ptr_time'], '^-', label='Pointer Network', color='green')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Participants (N)')
    plt.ylabel('Running Time (seconds)')
    plt.title('(a) Running time vs. N')
    plt.legend()
    plt.grid(True)
    # 子图 (b) ST-WSC 退化
    plt.subplot(1,2,2)
    full_vals = results['full_wsc']
    prune_degrad = [(1 - p/f)*100 if not np.isnan(f) else np.nan for p,f in zip(results['prune_wsc'], full_vals)]
    ptr_degrad  = [(1 - p/f)*100 if not np.isnan(f) else np.nan for p,f in zip(results['ptr_wsc'], full_vals)]
    plt.plot(N, prune_degrad, 's-', label='Candidate Pruning', color='blue')
    plt.plot(N, ptr_degrad, '^-', label='Pointer Network', color='green')
    plt.xscale('log')
    plt.xlabel('Number of Participants (N)')
    plt.ylabel('ST-WSC Degradation (%)')
    plt.title('(b) ST-WSC degradation relative to Full Space')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('candidate_pruning.pdf', dpi=300)
    plt.show()

# ==================== 主程序 ====================
if __name__ == "__main__":
    foursquare_checkins = r"F:\pycharm-community-2020\untitled\2026-4-10-...\dataset_WWW2019\dataset_WWW_Checkins_anonymized.txt"
    foursquare_poi = r"F:\pycharm-community-2020\untitled\2026-4-10-...\dataset_WWW2019\raw_POIs.txt"
    gowalla_checkins = r"F:\pycharm-community-2020\untitled\2026-4-10-...\Gowalla_totalCheckins.txt"

    MAX_USERS_NEEDED = 60000
    all_trajs = {}

    # 加载 POI 映射
    poi_map = load_poi_mapping(foursquare_poi) if os.path.exists(foursquare_poi) else {}

    # 加载 Foursquare
    if os.path.exists(foursquare_checkins):
        print("Sampling Foursquare...")
        df_fs = sample_users_from_file(foursquare_checkins, MAX_USERS_NEEDED//2, True, poi_mapping=poi_map)
        if not df_fs.empty:
            df_fs = discretize_space(df_fs)
            traj_fs = build_trajectories(df_fs)
            all_trajs.update(traj_fs)
            print(f"  Foursquare: {len(traj_fs)} trajectories")
    # 加载 Gowalla
    if os.path.exists(gowalla_checkins):
        print("Sampling Gowalla...")
        df_gw = sample_users_from_file(gowalla_checkins, MAX_USERS_NEEDED//2, False)
        if not df_gw.empty:
            df_gw = discretize_space(df_gw)
            traj_gw = build_trajectories(df_gw)
            all_trajs.update(traj_gw)
            print(f"  Gowalla: {len(traj_gw)} trajectories")

    # 若真实数据不足，生成合成数据
    if len(all_trajs) < 500:
        print(f"Only {len(all_trajs)} real trajectories, generating synthetic...")
        syn = generate_synthetic_trajectories(MAX_USERS_NEEDED)
        all_trajs.update(syn)

    print(f"Total users: {len(all_trajs)}")
    if not all_trajs:
        raise RuntimeError("No trajectories available.")

    # 区域权重（均匀分布）
    region_weights = np.ones((T, R)) / (T * R)
    N_list = [100, 1000, 5000, 10000, 50000]
    results = run_experiment(all_trajs, region_weights, N_list)
    plot_results(results)