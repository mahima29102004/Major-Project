# ── SMART LOAD: All buys + sampled interactions ──────────────────
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import warnings, gc
warnings.filterwarnings('ignore')

# Paper's exact timestamps
TS_START = 1511539200  # Nov 25 2017
TS_END   = 1512230400  # Dec 03 2017
TS_TRAIN = 1512144000  # Dec 02 2017 (train cutoff)

COL_NAMES  = ['user_id','item_id','category_id',
              'behavior_type','timestamp']
CHUNK_SIZE = 500_000

print("⏳ Pass 1: Loading ALL buy events in 21-day window...")
buy_chunks = []
total_read = 0

for i, chunk in enumerate(pd.read_csv(
        '/content/taobao/data/UserBehavior.csv',
        header    = None,
        names     = COL_NAMES,
        chunksize = CHUNK_SIZE,
        dtype     = {'user_id'      : np.int32,
                     'item_id'      : np.int32,
                     'category_id'  : np.int64,
                     'behavior_type': str,
                     'timestamp'    : np.int32}
    )):
    # Keep only buy events in window
    filtered = chunk[
        (chunk['timestamp'] >= TS_START) &
        (chunk['timestamp'] <= TS_END)   &
        (chunk['behavior_type'] == 'buy')
    ]
    if len(filtered) > 0:
        buy_chunks.append(filtered)
    total_read += len(chunk)
    if (i+1) % 20 == 0:
        print(f"   Processed {total_read:,} rows | "
              f"Buy events: "
              f"{sum(len(c) for c in buy_chunks):,}")

df_buy = pd.concat(buy_chunks, ignore_index=True)
del buy_chunks
gc.collect()
print(f"✅ All buy events loaded: {len(df_buy):,} rows")

# ── Pass 2: Sample non-buy events (10%) for embeddings ────────
print("\n⏳ Pass 2: Sampling non-buy events (10%) for embeddings...")
other_chunks = []
total_read   = 0

for i, chunk in enumerate(pd.read_csv(
        '/content/taobao/data/UserBehavior.csv',
        header    = None,
        names     = COL_NAMES,
        chunksize = CHUNK_SIZE,
        dtype     = {'user_id'      : np.int32,
                     'item_id'      : np.int32,
                     'category_id'  : np.int64,
                     'behavior_type': str,
                     'timestamp'    : np.int32}
    )):
    filtered = chunk[
        (chunk['timestamp'] >= TS_START) &
        (chunk['timestamp'] <= TS_END)   &
        (chunk['behavior_type'] != 'buy')
    ].sample(frac=0.10, random_state=42)

    if len(filtered) > 0:
        other_chunks.append(filtered)
    total_read += len(chunk)
    if (i+1) % 20 == 0:
        print(f"   Processed {total_read:,} rows | "
              f"Non-buy sampled: "
              f"{sum(len(c) for c in other_chunks):,}")

df_other = pd.concat(other_chunks, ignore_index=True)
del other_chunks
gc.collect()
print(f"✅ Non-buy events sampled: {len(df_other):,} rows")

# ── Combine ────────────────────────────────────────────────────
df_raw = pd.concat([df_buy, df_other], ignore_index=True)
del df_other
gc.collect()

df_raw['datetime'] = pd.to_datetime(df_raw['timestamp'], unit='s')
print(f"\n✅ Combined dataset: {len(df_raw):,} rows")
print(f"   Memory usage: "
      f"{df_raw.memory_usage(deep=True).sum()/1024**2:.0f} MB")

# ── Train / Test split ─────────────────────────────────────────
df_train_all = df_raw[df_raw['timestamp'] <= TS_TRAIN].copy()
df_test_all  = df_raw[df_raw['timestamp'] >  TS_TRAIN].copy()
df_train_buy = df_train_all[df_train_all['behavior_type']=='buy'].copy()
df_test_buy  = df_test_all [df_test_all ['behavior_type']=='buy'].copy()

print(f"\n📅 Date range  : "
      f"{df_raw['datetime'].min().date()} → "
      f"{df_raw['datetime'].max().date()}")
print(f"   Train records  : {len(df_train_all):,}")
print(f"   Test  records  : {len(df_test_all):,}")
print(f"   Train purchases: {len(df_train_buy):,}")
print(f"   Test  purchases: {len(df_test_buy):,}")

# ── Filter users with >= 2 train purchases ─────────────────────
user_buy_counts = df_train_buy.groupby('user_id')['item_id'].count()
valid_users     = user_buy_counts[user_buy_counts >= 2].index
df_train_all    = df_train_all[df_train_all['user_id'].isin(valid_users)]
df_train_buy    = df_train_buy[df_train_buy['user_id'].isin(valid_users)]
df_test_buy     = df_test_buy [df_test_buy ['user_id'].isin(valid_users)]

print(f"\n✅ Valid users (≥2 purchases): {len(valid_users):,}")

# ── Build H(u) G(u) ───────────────────────────────────────────
print("\n⏳ Building H(u) and G(u)...")
H_u, G_u = {}, {}
for uid, grp in tqdm(df_train_buy.groupby('user_id'),
                     desc="H(u)/G(u)"):
    grp   = grp.sort_values('timestamp')
    split = max(1, int(len(grp) * 0.8))
    H_u[uid] = grp.iloc[:split]['item_id'].tolist()
    G_u[uid] = grp.iloc[split:]['item_id'].tolist()

# ── Test ground truth ──────────────────────────────────────────
GT_test = (
    df_test_buy
    .groupby('user_id')['item_id']
    .apply(list)
    .to_dict()
)

# ── Item category mapping ──────────────────────────────────────
item_cat = df_raw.groupby('item_id')['category_id'].first().to_dict()

# ── Check GT coverage ─────────────────────────────────────────
print("\n🔍 GT Coverage Check:")
hits, total = 0, 0
all_train_items = set(df_train_buy['item_id'].unique())
for uid, gt_items in GT_test.items():
    for item in gt_items:
        total += 1
        if item in all_train_items:
            hits += 1
coverage = hits/total*100 if total > 0 else 0
print(f"   GT items seen in training : {hits}/{total} "
      f"({coverage:.1f}%)")

# ── Final summary ──────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  SMART LOAD COMPLETE")
print(f"{'='*50}")
print(f"  Total rows          : {len(df_raw):,}")
print(f"  Valid users         : {len(valid_users):,}")
print(f"  H(u) users          : {len(H_u):,}")
print(f"  Test GT users       : {len(GT_test):,}")
print(f"  Unique train items  : {df_train_buy['item_id'].nunique():,}")
print(f"  GT coverage         : {coverage:.1f}%")
print(f"{'='*50}")

beh_counts = df_train_all['behavior_type'].value_counts()
print(f"\nBehavior distribution (train):")
for b, cnt in beh_counts.items():
    pct = cnt/len(df_train_all)*100
    bar = '█' * int(pct/3)
    print(f"  {b:5s}: {cnt:8,} ({pct:5.1f}%) {bar}")