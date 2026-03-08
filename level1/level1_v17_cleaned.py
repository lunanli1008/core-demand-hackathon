"""
Level 1 v17 — v13 + full 12-step data cleaning pipeline
  1. Date parse + out-of-range filter
  2. Exact row dedup (keeps repeat orders with different fields)
  3. Missing eclass dropped, manufacturer filled
  4. Manufacturer normalized (lowercase, strip suffixes/symbols, unify unknown)
  5. Eclass validation (report non-8-digit, keep all)
  6. Price outliers (clip negatives to 0, report P99.9)
  7. Quantity validation (drop qty ≤ 0)
  8. Spend computed after cleaning
  9. NACE codes validated, 2d/4d derived
  10. Employee count validated, size buckets
  11. Entity consistency: conflicting NACE/size per buyer resolved via mode
  12. Month ordinal derived
  All model/portfolio logic identical to v13.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ──
SAVINGS_RATE = 0.10
FEE_PER_ITEM = 10.0
INPUT_PATH = 'plis_training.csv'
TEST_PATH = 'customer_test.csv.gz'
OUTPUT_PATH = 'submission_level1_v17.csv'

TARGET_PREDICTIONS = 49_000

# Backtest cutoff — leaves ~6 months future window
CUTOFF_DATE = '2025-01-01'

# Warm candidate generation (permissive — let model rank)
WARM_STALE_MAX_MONTHS = 36
WARM_HALFLIFE_MONTHS = 6

# Cold/hybrid — loosened from v9 to expand pool
COLD_NACE4_MIN_FRAC = 0.03
COLD_NACE3_MIN_FRAC = 0.05
COLD_NACE2_MIN_FRAC = 0.07
COLD_UNIVERSAL_MIN_FRAC = 0.15
COLD_MIN_AVG_SPEND = 30
COLD_MAX_PER_BUYER = 700
HYBRID_MAX_PER_BUYER = 200
HYBRID_MIN_FRAC = 0.08
HYBRID_MIN_AVG_SPEND = 60

# Set-based co-buy — slightly loosened
COBUY_SET_MIN_PAIR_COUNT = 6
COBUY_SET_MIN_COND_PROB = 0.05
COBUY_TOP_EDGES_PER_SRC = 40
COBUY_MAX_PER_BUYER = 150

# Cold neighbor — loosened
COLD_NEIGHBOR_POOL_CAP = 150
COLD_NEIGHBOR_MIN_SUPPORT = 6
COLD_NEIGHBOR_TOP_ECLASSES = 30

# Portfolio
WARM_BASE_PER_BUYER = 50
COLD_BASE_PER_BUYER = 30
WARM_MAX_PER_BUYER = 2500
COLD_MAX_PER_BUYER_GLOBAL = 1200

EV_FLOOR = {
    'warm': -8.0,
    'warm_cobuy': -2.0,
    'warm_hierarchy': -2.0,
    'hybrid': 0.0,
    'cold': -5.0,
    'cold_neighbor': 0.0,
}


def employee_bucket(val):
    if pd.isna(val):
        return "unknown"
    n = float(val)
    if n < 10: return "lt10"
    if n < 50: return "10_49"
    if n < 250: return "50_249"
    if n < 1000: return "250_999"
    return "1000_plus"


def normalize_nace(val):
    if pd.isna(val):
        return ""
    return "".join(ch for ch in str(val) if ch.isdigit())


_NACE_SECTION_MAP = {
    range(1, 4): 'A', range(5, 10): 'B', range(10, 34): 'C',
    range(35, 36): 'D', range(36, 40): 'E', range(41, 44): 'F',
    range(45, 48): 'G', range(49, 54): 'H', range(55, 57): 'I',
    range(58, 64): 'J', range(64, 67): 'K', range(68, 69): 'L',
    range(69, 76): 'M', range(77, 83): 'N', range(84, 85): 'O',
    range(85, 86): 'P', range(86, 89): 'Q', range(90, 94): 'R',
    range(94, 97): 'S', range(97, 99): 'T', range(99, 100): 'U',
}

def nace_to_section(nace_str):
    if not nace_str or len(nace_str) < 2:
        return ''
    try:
        code = int(nace_str[:2])
    except ValueError:
        return ''
    for rng, letter in _NACE_SECTION_MAP.items():
        if code in rng:
            return letter
    return ''


# ══════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════
print("Loading test buyers...")
test_buyers = pd.read_csv(TEST_PATH, sep='\t')
test_ids = set(test_buyers['legal_entity_id'])
print(f"Test buyers: {len(test_ids)}")

test_buyers['size_bucket'] = test_buyers['estimated_number_employees'].apply(employee_bucket)
test_buyers['nace_primary'] = test_buyers['nace_code'].apply(normalize_nace)
if 'secondary_nace_code' in test_buyers.columns:
    test_buyers['nace_secondary'] = test_buyers['secondary_nace_code'].apply(normalize_nace)
else:
    test_buyers['nace_secondary'] = ''

print("Loading training data...")
df = pd.read_csv(INPUT_PATH, sep='\t', low_memory=False)

# ── 1. Dates: parse + filter out-of-range ──
df['orderdate'] = pd.to_datetime(df['orderdate'], errors='coerce')
df = df.dropna(subset=['orderdate'])
df = df[(df['orderdate'] >= '2023-01-01') & (df['orderdate'] <= '2025-12-31')]

# ── 2. Duplicates: exact row dedup only (repeat orders kept) ──
before_dedup = len(df)
df = df.drop_duplicates()
print(f"Exact dedup: {before_dedup:,} → {len(df):,} ({before_dedup - len(df):,} removed)")

# ── 3. Missing values: drop missing eclass, fill manufacturer ──
df['eclass'] = df['eclass'].fillna('').astype(str).str.strip()
df['eclass'] = df['eclass'].str.replace(r'\.0$', '', regex=True)
df = df[df['eclass'] != '']

# ── 4. Manufacturer normalization (not used in L1, but clean for features) ──
if 'manufacturer' in df.columns:
    df['manufacturer'] = df['manufacturer'].fillna('unknown').astype(str).str.strip().str.lower()
    # Unify unknown variants
    unknown_variants = {'unbekannt', 'unknown', 'n/a', 'na', 'none', '-', '', 'nan'}
    df.loc[df['manufacturer'].isin(unknown_variants), 'manufacturer'] = 'unknown'
    # Strip legal suffixes and symbols
    df['manufacturer'] = df['manufacturer'].str.replace(r'[®™©]', '', regex=True).str.strip()
    df['manufacturer'] = df['manufacturer'].str.replace(r'\s+(gmbh|ag|kg|ohg|co\.?\s*kg|inc\.?|ltd\.?|corp\.?|se|sa)\s*$', '', regex=True).str.strip()

# ── 5. Eclass validation: enforce 8-digit format ──
df['eclass_len'] = df['eclass'].str.len()
valid_eclass = df['eclass'].str.match(r'^\d{8}$')
invalid_eclass = (~valid_eclass).sum()
if invalid_eclass > 0:
    print(f"Non-8-digit eclass rows: {invalid_eclass:,} (keeping — may be valid short codes)")
    # Keep them — some valid eclasses may be shorter; don't drop real data

# ── 6. Price outliers: clip negatives, cap extreme ──
df['vk_per_item'] = pd.to_numeric(df['vk_per_item'], errors='coerce').fillna(0)
df.loc[df['vk_per_item'] < 0, 'vk_per_item'] = 0
p999 = df['vk_per_item'].quantile(0.999)
print(f"Price P99.9: €{p999:.2f}")
# Don't clip high prices — expensive items are high-value for scoring

# ── 7. Quantity validation: drop qty ≤ 0 ──
df['quantityvalue'] = pd.to_numeric(df['quantityvalue'], errors='coerce').fillna(1)
before_qty = len(df)
df = df[df['quantityvalue'] > 0]
print(f"Qty ≤ 0 removed: {before_qty - len(df):,}")

# ── 8. Compute spend ──
df['spend'] = (df['quantityvalue'] * df['vk_per_item']).clip(lower=0)

# ── 9. NACE codes: validate, derive 2d/4d ──
df['nace_code'] = df['nace_code'].astype(str).str.strip()
df['nace_primary'] = df['nace_code'].apply(normalize_nace)
if 'secondary_nace_code' in df.columns:
    df['nace_secondary'] = df['secondary_nace_code'].apply(normalize_nace)
else:
    df['nace_secondary'] = ''

# ── 10. Employee count: validate, bin ──
df['estimated_number_employees'] = pd.to_numeric(df['estimated_number_employees'], errors='coerce')
df['size_bucket'] = df['estimated_number_employees'].apply(employee_bucket)

# ── 11. Entity consistency: resolve conflicting NACE/employee per buyer via mode ──
buyer_nace_mode = df.groupby('legal_entity_id')['nace_primary'].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
buyer_nace2_mode = df.groupby('legal_entity_id')['nace_secondary'].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
buyer_size_mode = df.groupby('legal_entity_id')['size_bucket'].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
df['nace_primary'] = df['legal_entity_id'].map(buyer_nace_mode)
df['nace_secondary'] = df['legal_entity_id'].map(buyer_nace2_mode)
df['size_bucket'] = df['legal_entity_id'].map(buyer_size_mode)

# ── 12. Month ordinal ──
df['month_ord'] = df['orderdate'].dt.year * 12 + df['orderdate'].dt.month
df = df.drop(columns=['eclass_len'], errors='ignore')
print(f"Training rows after cleaning: {len(df):,}")

latest_month = df['month_ord'].max()

# Split warm vs cold
warm_ids = test_ids & set(df['legal_entity_id'])
cold_ids = test_ids - warm_ids
print(f"Warm: {len(warm_ids)}, Cold: {len(cold_ids)}")


# ══════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING FUNCTION
# ══════════════════════════════════════════════════════════════════════
def build_features(source_df, ref_latest_month):
    """Build (buyer, eclass) features from a dataframe."""
    source_df = source_df.copy()
    source_df['quarter'] = source_df['orderdate'].dt.to_period('Q')

    agg = source_df.groupby(['legal_entity_id', 'eclass']).agg(
        total_spend=('spend', 'sum'),
        n_orders=('orderdate', 'count'),
        n_months=('month_ord', 'nunique'),
        last_month=('month_ord', 'max'),
        first_month=('month_ord', 'min'),
        avg_price=('vk_per_item', 'mean'),
        median_price=('vk_per_item', 'median'),
        max_price=('vk_per_item', 'max'),
        avg_qty=('quantityvalue', 'mean'),
        total_qty=('quantityvalue', 'sum'),
    ).reset_index()

    # Quarter counts
    qc = source_df.groupby(['legal_entity_id', 'eclass'])['quarter'].nunique().reset_index(name='n_quarters')
    agg = agg.merge(qc, on=['legal_entity_id', 'eclass'], how='left')
    agg['n_quarters'] = agg['n_quarters'].fillna(1).astype(int)

    # Recent 6 months
    recent_cutoff = ref_latest_month - 6
    recent = source_df[source_df['month_ord'] >= recent_cutoff]
    recent_agg = recent.groupby(['legal_entity_id', 'eclass']).agg(
        recent_spend=('spend', 'sum'),
        recent_orders=('orderdate', 'count'),
        recent_months=('month_ord', 'nunique'),
        recent_avg_price=('vk_per_item', 'mean'),
    ).reset_index()

    # Recent 3 months (very recent)
    very_recent_cutoff = ref_latest_month - 3
    very_recent = source_df[source_df['month_ord'] >= very_recent_cutoff]
    vr_agg = very_recent.groupby(['legal_entity_id', 'eclass']).agg(
        vrecent_spend=('spend', 'sum'),
        vrecent_months=('month_ord', 'nunique'),
    ).reset_index()

    agg = agg.merge(recent_agg, on=['legal_entity_id', 'eclass'], how='left')
    agg = agg.merge(vr_agg, on=['legal_entity_id', 'eclass'], how='left')
    for col in ['recent_spend', 'recent_orders', 'recent_months', 'recent_avg_price',
                'vrecent_spend', 'vrecent_months']:
        agg[col] = agg[col].fillna(0)

    # Derived features
    agg['recency_gap'] = ref_latest_month - agg['last_month']
    agg['recency_weight'] = np.exp(-np.log(2.0) * agg['recency_gap'] / WARM_HALFLIFE_MONTHS)
    agg['lifespan_months'] = agg['last_month'] - agg['first_month'] + 1
    agg['avg_monthly_spend'] = agg['total_spend'] / agg['n_months']
    agg['projected_annual'] = agg['avg_monthly_spend'] * 12.0 * agg['recency_weight']
    agg['spend_per_order'] = agg['total_spend'] / agg['n_orders'].clip(lower=1)
    agg['orders_per_month'] = agg['n_orders'] / agg['lifespan_months'].clip(lower=1)

    # Regularity: how evenly spread are purchases
    agg['month_coverage'] = agg['n_months'] / agg['lifespan_months'].clip(lower=1)
    agg['quarter_coverage'] = agg['n_quarters'] / (agg['lifespan_months'] / 3).clip(lower=1)

    # sqrt-price features (aligned with scoring formula)
    agg['sqrt_avg_price'] = np.sqrt(agg['avg_price'].clip(lower=0.01))
    agg['sqrt_freq_score'] = agg['sqrt_avg_price'] * (0.6 * agg['n_months'] + 0.4 * agg['n_quarters']) * agg['recency_weight']

    # Spend trends
    agg['recent_spend_ratio'] = agg['recent_spend'] / agg['total_spend'].clip(lower=1)
    agg['vrecent_spend_ratio'] = agg['vrecent_spend'] / agg['total_spend'].clip(lower=1)
    agg['is_recent'] = (agg['recency_gap'] <= 6).astype(int)
    agg['is_very_recent'] = (agg['recency_gap'] <= 3).astype(int)
    agg['is_recurring'] = (agg['n_months'] >= 2).astype(int)
    agg['is_frequent'] = (agg['n_months'] >= 4).astype(int)

    # Buyer-level features
    buyer_stats = source_df.groupby('legal_entity_id').agg(
        buyer_total_spend=('spend', 'sum'),
        buyer_n_eclasses=('eclass', 'nunique'),
        buyer_n_orders=('orderdate', 'count'),
        buyer_n_months=('month_ord', 'nunique'),
    ).reset_index()
    agg = agg.merge(buyer_stats, on='legal_entity_id', how='left')

    # Eclass share within buyer
    agg['spend_share'] = agg['total_spend'] / agg['buyer_total_spend'].clip(lower=1)
    agg['order_share'] = agg['n_orders'] / agg['buyer_n_orders'].clip(lower=1)

    # Eclass-level global stats
    eclass_stats = source_df.groupby('eclass').agg(
        eclass_n_buyers=('legal_entity_id', 'nunique'),
        eclass_total_spend=('spend', 'sum'),
        eclass_avg_price=('vk_per_item', 'mean'),
    ).reset_index()
    total_buyers = source_df['legal_entity_id'].nunique()
    eclass_stats['eclass_buyer_frac'] = eclass_stats['eclass_n_buyers'] / max(total_buyers, 1)
    eclass_stats['eclass_avg_spend_per_buyer'] = eclass_stats['eclass_total_spend'] / eclass_stats['eclass_n_buyers'].clip(lower=1)
    agg = agg.merge(eclass_stats, on='eclass', how='left')

    return agg


FEATURE_COLS = [
    'total_spend', 'n_orders', 'n_months', 'n_quarters',
    'avg_price', 'median_price', 'max_price', 'avg_qty', 'total_qty',
    'recent_spend', 'recent_orders', 'recent_months', 'recent_avg_price',
    'vrecent_spend', 'vrecent_months',
    'recency_gap', 'recency_weight', 'lifespan_months',
    'avg_monthly_spend', 'projected_annual', 'spend_per_order', 'orders_per_month',
    'month_coverage', 'quarter_coverage',
    'sqrt_avg_price', 'sqrt_freq_score',
    'recent_spend_ratio', 'vrecent_spend_ratio',
    'is_recent', 'is_very_recent', 'is_recurring', 'is_frequent',
    'buyer_total_spend', 'buyer_n_eclasses', 'buyer_n_orders', 'buyer_n_months',
    'spend_share', 'order_share',
    'eclass_n_buyers', 'eclass_buyer_frac', 'eclass_avg_spend_per_buyer', 'eclass_avg_price',
]


# ══════════════════════════════════════════════════════════════════════
# 3. BACKTEST: TRAIN LIGHTGBM
# ══════════════════════════════════════════════════════════════════════
print("\n── Building Backtest ──")
cutoff = pd.to_datetime(CUTOFF_DATE)
hist_df = df[df['orderdate'] < cutoff].copy()
future_df = df[df['orderdate'] >= cutoff].copy()
hist_latest = hist_df['month_ord'].max()
print(f"Hist rows: {len(hist_df):,}, Future rows: {len(future_df):,}")
print(f"Hist period: {hist_df['orderdate'].min().date()} to {hist_df['orderdate'].max().date()}")
print(f"Future period: {future_df['orderdate'].min().date()} to {future_df['orderdate'].max().date()}")

# Build ground truth labels: core = appears in ≥2 months in future
future_core = future_df.groupby(['legal_entity_id', 'eclass'])['month_ord'].nunique().reset_index(name='future_months')
future_core = future_core[future_core['future_months'] >= 2]
core_set = set(zip(future_core['legal_entity_id'].astype(str), future_core['eclass'].astype(str)))
print(f"Core (buyer,eclass) pairs in future: {len(core_set):,}")

# Build features from historical data (only for buyers in hist)
print("Building historical features...")
hist_features = build_features(hist_df, hist_latest)

# Label
hist_features['label'] = hist_features.apply(
    lambda r: 1 if (str(r['legal_entity_id']), str(r['eclass'])) in core_set else 0, axis=1
)
print(f"Positive rate: {hist_features['label'].mean():.3f} ({hist_features['label'].sum():,} / {len(hist_features):,})")

# Train LightGBM
print("\n── Training LightGBM ──")
X_train = hist_features[FEATURE_COLS].fillna(0)
y_train = hist_features['label']

# Use all warm test buyers in hist for validation if possible
val_mask = hist_features['legal_entity_id'].isin(warm_ids)
if val_mask.sum() > 100:
    X_val = X_train[val_mask]
    y_val = y_train[val_mask]
    X_fit = X_train[~val_mask]
    y_fit = y_train[~val_mask]
    eval_set = [(X_val, y_val)]
    print(f"Train: {len(X_fit):,}, Val (test warm buyers): {len(X_val):,}")
else:
    X_fit = X_train
    y_fit = y_train
    eval_set = None
    print(f"Train: {len(X_fit):,}, no separate val set")

# Scale positive weight for imbalanced data
pos_rate = y_fit.mean()
scale_pos = (1 - pos_rate) / max(pos_rate, 0.001)
print(f"scale_pos_weight: {scale_pos:.1f}")

model = lgb.LGBMClassifier(
    n_estimators=800,
    max_depth=7,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    random_state=42,
    verbose=-1,
)

callbacks = []
if eval_set:
    callbacks = [lgb.early_stopping(50, verbose=True), lgb.log_evaluation(100)]

model.fit(X_fit, y_fit, eval_set=eval_set, callbacks=callbacks if callbacks else None)

# Validation metrics
if eval_set:
    val_proba = model.predict_proba(X_val)[:, 1]
    val_pred = (val_proba > 0.3).astype(int)
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    print(f"\nVal AUC: {roc_auc_score(y_val, val_proba):.4f}")
    print(f"Val Precision@0.3: {precision_score(y_val, val_pred):.4f}")
    print(f"Val Recall@0.3: {recall_score(y_val, val_pred):.4f}")

# Feature importance
importance = pd.DataFrame({
    'feature': FEATURE_COLS,
    'importance': model.feature_importances_,
}).sort_values('importance', ascending=False)
print("\nTop 15 features:")
for _, row in importance.head(15).iterrows():
    print(f"  {row['feature']:30s} {row['importance']:6.0f}")


# ══════════════════════════════════════════════════════════════════════
# 4. RETRAIN ON FULL DATA + PREDICT FOR TEST BUYERS
# ══════════════════════════════════════════════════════════════════════
print("\n── Retrain on Full Data ──")

# Build features from ALL training data
full_features = build_features(df, latest_month)
X_full = full_features[FEATURE_COLS].fillna(0)
y_full_proxy = full_features.apply(
    lambda r: 1 if (str(r['legal_entity_id']), str(r['eclass'])) in core_set else 0, axis=1
)

# Retrain on everything
best_iter = 600
if hasattr(model, 'best_iteration_') and model.best_iteration_ and model.best_iteration_ > 0:
    best_iter = model.best_iteration_
print(f"Using {best_iter} iterations for full retrain")

model_full = lgb.LGBMClassifier(
    n_estimators=best_iter,
    max_depth=7,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    random_state=42,
    verbose=-1,
)
model_full.fit(X_full, y_full_proxy)

# Score warm candidates
warm_features = full_features[full_features['legal_entity_id'].isin(warm_ids)].copy()
X_warm = warm_features[FEATURE_COLS].fillna(0)
warm_features['p_core_lgb'] = model_full.predict_proba(X_warm)[:, 1]

print(f"Warm scored: {len(warm_features):,}")
print(f"  p_core > 0.5: {(warm_features['p_core_lgb'] > 0.5).sum():,}")
print(f"  p_core > 0.3: {(warm_features['p_core_lgb'] > 0.3).sum():,}")
print(f"  p_core > 0.1: {(warm_features['p_core_lgb'] > 0.1).sum():,}")


# ══════════════════════════════════════════════════════════════════════
# 5. WARM CANDIDATES (LightGBM-scored)
# ══════════════════════════════════════════════════════════════════════
print("\n── Warm Start (LightGBM) ──")

warm_candidates = warm_features[
    (warm_features['recency_gap'] <= WARM_STALE_MAX_MONTHS)
].copy()

# capture_score uses learned p_core × value signal
warm_candidates['p_core'] = warm_candidates['p_core_lgb']
warm_candidates['expected_savings'] = warm_candidates['projected_annual'] * SAVINGS_RATE
warm_candidates['ev'] = warm_candidates['p_core'] * warm_candidates['expected_savings'] - FEE_PER_ITEM

# Capture score: blend learned probability with value
warm_candidates['capture_score'] = (
    warm_candidates['p_core']
    * (0.55 * warm_candidates['sqrt_freq_score'] + 0.45 * np.sqrt(warm_candidates['projected_annual'].clip(lower=1)))
)

warm_candidates['source'] = 'warm'
warm_candidates['task'] = 'predict future'
warm_candidates['n_sources'] = 1

print(f"Warm candidates: {len(warm_candidates):,}")
print(f"  EV > 0: {(warm_candidates['ev'] > 0).sum():,}")

# Build buyer → eclasses lookup (used by hierarchy, cobuy, hybrid)
warm_buyer_eclasses = {}
for bid in warm_ids:
    warm_buyer_eclasses[bid] = set(warm_candidates[warm_candidates['legal_entity_id'] == bid]['eclass'])


# ══════════════════════════════════════════════════════════════════════
# 5.5 ECLASS HIERARCHY SHARING
# ══════════════════════════════════════════════════════════════════════
print("\n── Eclass Hierarchy Sharing ──")
HIERARCHY_MIN_GROUP_BUYERS = 5
HIERARCHY_MIN_BUYER_FRAC = 0.12
HIERARCHY_MAX_PER_BUYER = 80

all_eclasses_h = df[['eclass', 'legal_entity_id', 'spend', 'vk_per_item']].copy()
all_eclasses_h['eclass_6d'] = all_eclasses_h['eclass'].str[:6]

group_profiles = all_eclasses_h.groupby(['eclass_6d', 'eclass']).agg(
    n_buyers=('legal_entity_id', 'nunique'),
    avg_spend=('spend', 'mean'),
    avg_price=('vk_per_item', 'mean'),
).reset_index()
group_totals = all_eclasses_h.groupby('eclass_6d')['legal_entity_id'].nunique().to_dict()
group_profiles['group_buyer_frac'] = group_profiles.apply(
    lambda r: r['n_buyers'] / max(group_totals.get(r['eclass_6d'], 1), 1), axis=1
)

hierarchy_rows = []
for bid in warm_ids:
    existing = warm_buyer_eclasses.get(bid, set())
    if not existing:
        continue
    existing_groups = {ec[:6] for ec in existing if len(ec) >= 6}
    count = 0
    for grp in existing_groups:
        siblings = group_profiles[
            (group_profiles['eclass_6d'] == grp) &
            (group_profiles['group_buyer_frac'] >= HIERARCHY_MIN_BUYER_FRAC) &
            (group_profiles['n_buyers'] >= HIERARCHY_MIN_GROUP_BUYERS)
        ]
        for _, sib in siblings.iterrows():
            if sib['eclass'] in existing or count >= HIERARCHY_MAX_PER_BUYER:
                continue
            p_core = min(sib['group_buyer_frac'] * 0.6, 0.50)
            spend = sib['avg_spend']
            price = sib['avg_price']
            ev = p_core * spend * SAVINGS_RATE - FEE_PER_ITEM
            sf_score = np.sqrt(max(price, 0.01)) * (1.0 + sib['group_buyer_frac'] * 3.0)
            capture_score = p_core * (0.55 * sf_score + 0.45 * np.sqrt(max(spend, 1)))
            hierarchy_rows.append({
                'legal_entity_id': bid, 'eclass': sib['eclass'],
                'capture_score': capture_score, 'ev': ev,
                'projected_annual': spend, 'p_core': p_core,
                'source': 'warm_hierarchy', 'task': 'predict future', 'n_sources': 1,
            })
            count += 1

hierarchy_candidates = pd.DataFrame(hierarchy_rows) if hierarchy_rows else pd.DataFrame(
    columns=['legal_entity_id', 'eclass', 'capture_score', 'ev', 'projected_annual', 'p_core', 'source', 'task', 'n_sources']
)
print(f"Hierarchy candidates: {len(hierarchy_candidates):,}")


# ══════════════════════════════════════════════════════════════════════
# 6. CO-BUY (set-based, same as v6)
# ══════════════════════════════════════════════════════════════════════
print("\n── Set-Based Co-Buy ──")

eclass_avg_spend = df.groupby('eclass')['spend'].sum() / df.groupby('eclass')['legal_entity_id'].nunique()
eclass_avg_spend = eclass_avg_spend.to_dict()
eclass_avg_price = df.groupby('eclass')['vk_per_item'].mean().to_dict()

if 'set_id' in df.columns:
    print("  Building set-based co-occurrence...")
    set_df = df[['set_id', 'eclass']].dropna(subset=['set_id']).copy()
    set_df['set_id'] = set_df['set_id'].astype(str).str.strip()
    set_df = set_df[(set_df['set_id'] != '') & (set_df['set_id'] != 'nan')]
    set_df = set_df.drop_duplicates(['set_id', 'eclass'])
    set_to_eclasses = set_df.groupby('set_id')['eclass'].apply(set).to_dict()
    print(f"  Sets: {len(set_to_eclasses):,}")

    eclass_set_counts = defaultdict(int)
    pair_counts = defaultdict(int)
    for eclasses in set_to_eclasses.values():
        unique = sorted(eclasses)
        for ec in unique:
            eclass_set_counts[ec] += 1
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                pair_counts[(unique[i], unique[j])] += 1

    cobuy_edges = defaultdict(list)
    for (left, right), pc in pair_counts.items():
        if pc < COBUY_SET_MIN_PAIR_COUNT:
            continue
        lc = eclass_set_counts.get(left, 0)
        rc = eclass_set_counts.get(right, 0)
        if lc > 0:
            cond = pc / lc
            if cond >= COBUY_SET_MIN_COND_PROB:
                cobuy_edges[left].append((right, cond * np.log1p(pc), pc, cond))
        if rc > 0:
            cond = pc / rc
            if cond >= COBUY_SET_MIN_COND_PROB:
                cobuy_edges[right].append((left, cond * np.log1p(pc), pc, cond))

    for src in cobuy_edges:
        cobuy_edges[src] = sorted(cobuy_edges[src], key=lambda x: x[1], reverse=True)[:COBUY_TOP_EDGES_PER_SRC]
    print(f"  Co-buy sources: {len(cobuy_edges):,}")
else:
    cobuy_edges = defaultdict(list)

cobuy_rows = []
for bid in warm_ids:
    existing = warm_buyer_eclasses.get(bid, set())
    if not existing:
        continue
    cobuy_scores = defaultdict(lambda: {'edge_score': 0, 'cond_prob': 0, 'spend': 0, 'price': 0, 'sources': 0})
    for ec_a in existing:
        if ec_a not in cobuy_edges:
            continue
        for ec_b, edge_score, pc, cond_prob in cobuy_edges[ec_a]:
            if ec_b in existing:
                continue
            entry = cobuy_scores[ec_b]
            entry['edge_score'] = max(entry['edge_score'], edge_score)
            entry['cond_prob'] = max(entry['cond_prob'], cond_prob)
            entry['spend'] = max(entry['spend'], eclass_avg_spend.get(ec_b, 0))
            entry['price'] = max(entry['price'], eclass_avg_price.get(ec_b, 0))
            entry['sources'] += 1

    ranked = sorted(cobuy_scores.items(), key=lambda x: x[1]['edge_score'], reverse=True)
    for ec_b, info in ranked[:COBUY_MAX_PER_BUYER]:
        p_core = min(info['cond_prob'] * 0.85, 0.75)
        projected = info['spend']
        price = info['price']
        sf_score = np.sqrt(max(price, 0.01)) * (1.0 + info['cond_prob'] * 5.0)
        ev = p_core * projected * SAVINGS_RATE - FEE_PER_ITEM
        capture_score = p_core * (0.55 * sf_score + 0.45 * np.sqrt(max(projected, 1))) * (1.0 + 0.15 * min(info['sources'] - 1, 3))
        cobuy_rows.append({
            'legal_entity_id': bid, 'eclass': ec_b,
            'capture_score': capture_score, 'ev': ev,
            'projected_annual': projected, 'p_core': p_core,
            'source': 'warm_cobuy', 'task': 'predict future', 'n_sources': info['sources'],
        })

cobuy_candidates = pd.DataFrame(cobuy_rows) if cobuy_rows else pd.DataFrame(
    columns=['legal_entity_id', 'eclass', 'capture_score', 'ev', 'projected_annual', 'p_core', 'source', 'task', 'n_sources']
)
print(f"Co-buy candidates: {len(cobuy_candidates):,}")


# ══════════════════════════════════════════════════════════════════════
# 7. COLD START (NACE profiles + neighbor CF, same as v6)
# ══════════════════════════════════════════════════════════════════════
print("\n── Cold Start ──")

buyer_meta = df.groupby('legal_entity_id').agg(
    nace_primary=('nace_primary', 'first'),
    nace_secondary=('nace_secondary', 'first'),
    size_bucket=('size_bucket', 'first'),
).reset_index()

nace_rows = []
for _, row in buyer_meta.iterrows():
    bid = row['legal_entity_id']
    for nace in {row['nace_primary'], row['nace_secondary']}:
        if nace and len(nace) >= 2:
            nace_rows.append((bid, nace))

buyer_nace_long = pd.DataFrame(nace_rows, columns=['legal_entity_id', 'nace_code']).drop_duplicates()
buyer_nace_long['nace_2d'] = buyer_nace_long['nace_code'].str[:2]
buyer_nace_long['nace_3d'] = buyer_nace_long['nace_code'].str[:3]
buyer_nace_long['nace_4d'] = buyer_nace_long['nace_code'].str[:4]

eclass_data = df[['legal_entity_id', 'eclass', 'spend', 'vk_per_item']].copy()

def build_nace_profile(nace_col):
    merged = eclass_data.merge(
        buyer_nace_long[['legal_entity_id', nace_col]].drop_duplicates(),
        on='legal_entity_id'
    )
    profile = merged.groupby([nace_col, 'eclass']).agg(
        n_buyers=('legal_entity_id', 'nunique'),
        total_spend=('spend', 'sum'),
        avg_price=('vk_per_item', 'mean'),
    ).reset_index()
    profile['avg_spend'] = profile['total_spend'] / profile['n_buyers']
    nace_totals = merged.groupby(nace_col)['legal_entity_id'].nunique().to_dict()
    profile['buyer_frac'] = profile.apply(
        lambda r: r['n_buyers'] / max(nace_totals.get(r[nace_col], 1), 1), axis=1
    )
    return profile

print("  Building NACE profiles...")
nace4_profile = build_nace_profile('nace_4d')
nace3_profile = build_nace_profile('nace_3d')
nace2_profile = build_nace_profile('nace_2d')

universal = eclass_data.groupby('eclass').agg(
    n_buyers=('legal_entity_id', 'nunique'),
    total_spend=('spend', 'sum'),
    avg_price=('vk_per_item', 'mean'),
).reset_index()
universal['avg_spend'] = universal['total_spend'] / universal['n_buyers']
total_buyers_all = df['legal_entity_id'].nunique()
universal['buyer_frac'] = universal['n_buyers'] / total_buyers_all

# Size-bucket profiles
buyer_meta_with_nace = buyer_nace_long.merge(buyer_meta[['legal_entity_id', 'size_bucket']], on='legal_entity_id')
eclass_with_size = eclass_data.merge(
    buyer_meta_with_nace[['legal_entity_id', 'nace_2d', 'size_bucket']].drop_duplicates(),
    on='legal_entity_id'
)
size_profile = eclass_with_size.groupby(['nace_2d', 'size_bucket', 'eclass']).agg(
    n_buyers=('legal_entity_id', 'nunique'),
    total_spend=('spend', 'sum'),
    avg_price=('vk_per_item', 'mean'),
).reset_index()
size_profile['avg_spend'] = size_profile['total_spend'] / size_profile['n_buyers']
size_totals = eclass_with_size.groupby(['nace_2d', 'size_bucket'])['legal_entity_id'].nunique().to_dict()
size_profile['buyer_frac'] = size_profile.apply(
    lambda r: r['n_buyers'] / max(size_totals.get((r['nace_2d'], r['size_bucket']), 1), 1), axis=1
)


def generate_nace_candidates(nace_p, nace_s, sb, existing_eclasses, is_cold=True):
    min_fracs = {
        'nace4': COLD_NACE4_MIN_FRAC, 'nace3': COLD_NACE3_MIN_FRAC,
        'nace2': COLD_NACE2_MIN_FRAC, 'universal': COLD_UNIVERSAL_MIN_FRAC,
    }
    min_spend = COLD_MIN_AVG_SPEND

    if not is_cold:
        min_fracs = {k: max(v, HYBRID_MIN_FRAC) for k, v in min_fracs.items()}
        min_spend = HYBRID_MIN_AVG_SPEND

    candidates = {}
    def add_candidate(eclass, buyer_frac, avg_spend, avg_price, source, tier_weight):
        if eclass in existing_eclasses:
            return
        signal = buyer_frac * avg_spend * tier_weight
        if eclass in candidates:
            candidates[eclass]['score'] += signal
            candidates[eclass]['spend'] = max(candidates[eclass]['spend'], avg_spend)
            candidates[eclass]['price'] = max(candidates[eclass]['price'], avg_price)
            candidates[eclass]['buyer_frac'] = max(candidates[eclass]['buyer_frac'], buyer_frac)
            candidates[eclass]['sources'].add(source)
        else:
            candidates[eclass] = {
                'score': signal, 'spend': avg_spend, 'price': avg_price,
                'buyer_frac': buyer_frac, 'sources': {source},
            }

    for nace, nw in [(nace_p, 1.0), (nace_s, 0.75)]:
        if not nace or len(nace) < 2:
            continue
        if len(nace) >= 4:
            subset = nace4_profile[nace4_profile['nace_4d'] == nace[:4]]
            for _, item in subset.iterrows():
                if item['buyer_frac'] >= min_fracs['nace4'] and item['avg_spend'] >= min_spend:
                    add_candidate(item['eclass'], item['buyer_frac'], item['avg_spend'], item['avg_price'], 'nace4', 1.0 * nw)
        if len(nace) >= 3:
            subset = nace3_profile[nace3_profile['nace_3d'] == nace[:3]]
            for _, item in subset.iterrows():
                if item['buyer_frac'] >= min_fracs['nace3'] and item['avg_spend'] >= min_spend:
                    add_candidate(item['eclass'], item['buyer_frac'], item['avg_spend'], item['avg_price'], 'nace3', 0.75 * nw)
        n2 = nace[:2]
        subset = nace2_profile[nace2_profile['nace_2d'] == n2]
        for _, item in subset.iterrows():
            if item['buyer_frac'] >= min_fracs['nace2'] and item['avg_spend'] >= min_spend:
                add_candidate(item['eclass'], item['buyer_frac'], item['avg_spend'], item['avg_price'], 'nace2', 0.5 * nw)
        subset = size_profile[(size_profile['nace_2d'] == n2) & (size_profile['size_bucket'] == sb)]
        for _, item in subset.iterrows():
            if item['buyer_frac'] >= min_fracs['nace3'] and item['avg_spend'] >= min_spend:
                add_candidate(item['eclass'], item['buyer_frac'], item['avg_spend'], item['avg_price'], 'nace2_size', 0.65 * nw)

    if len(candidates) < 10 and is_cold:
        for _, item in universal.iterrows():
            if item['buyer_frac'] >= min_fracs['universal'] and item['avg_spend'] >= min_spend:
                add_candidate(item['eclass'], item['buyer_frac'], item['avg_spend'], item['avg_price'], 'universal', 0.2)

    return candidates


# Cold neighbor CF
print("  Building cold neighbor index...")
buyer_total_spend = df.groupby('legal_entity_id')['spend'].sum()
buyer_eclass_spend = df[df['legal_entity_id'].isin(warm_ids)].groupby(
    ['legal_entity_id', 'eclass']
)['spend'].sum().reset_index()
buyer_eclass_spend['total'] = buyer_eclass_spend['legal_entity_id'].map(buyer_total_spend)
buyer_eclass_spend['share'] = buyer_eclass_spend['spend'] / buyer_eclass_spend['total'].clip(lower=1)
buyer_eclass_spend = buyer_eclass_spend.sort_values(['legal_entity_id', 'share'], ascending=[True, False])
buyer_eclass_spend['rank'] = buyer_eclass_spend.groupby('legal_entity_id').cumcount()
buyer_eclass_top = buyer_eclass_spend[buyer_eclass_spend['rank'] < COLD_NEIGHBOR_TOP_ECLASSES]

neighbor_eclass_share = defaultdict(list)
for _, row in buyer_eclass_top.iterrows():
    neighbor_eclass_share[row['legal_entity_id']].append((row['eclass'], row['share']))

tier_index = {
    'n4_size': defaultdict(list), 'n3_size': defaultdict(list),
    'n2_size': defaultdict(list), 'n2': defaultdict(list),
}
for bid in warm_ids:
    meta = buyer_meta[buyer_meta['legal_entity_id'] == bid]
    if meta.empty:
        continue
    r = meta.iloc[0]
    sb = r['size_bucket']
    for nace in {r['nace_primary'], r['nace_secondary']}:
        if not nace:
            continue
        if len(nace) >= 4:
            tier_index['n4_size'][f"{nace[:4]}|{sb}"].append(bid)
        if len(nace) >= 3:
            tier_index['n3_size'][f"{nace[:3]}|{sb}"].append(bid)
        if len(nace) >= 2:
            tier_index['n2_size'][f"{nace[:2]}|{sb}"].append(bid)
            tier_index['n2'][nace[:2]].append(bid)

def get_cold_neighbor_candidates(nace_p, nace_s, sb, existing_eclasses):
    pool_weights = defaultdict(float)
    tier_order = [('n4_size', 1.0), ('n3_size', 0.75), ('n2_size', 0.55), ('n2', 0.35)]
    for nace in {nace_p, nace_s}:
        if not nace:
            continue
        for tier_name, weight in tier_order:
            if tier_name == 'n4_size' and len(nace) >= 4: key = f"{nace[:4]}|{sb}"
            elif tier_name == 'n3_size' and len(nace) >= 3: key = f"{nace[:3]}|{sb}"
            elif tier_name == 'n2_size' and len(nace) >= 2: key = f"{nace[:2]}|{sb}"
            elif tier_name == 'n2' and len(nace) >= 2: key = nace[:2]
            else: continue
            for bid in tier_index.get(tier_name, {}).get(key, []):
                pool_weights[bid] = max(pool_weights[bid], weight)
                if len(pool_weights) >= COLD_NEIGHBOR_POOL_CAP:
                    break
    if not pool_weights:
        return {}
    eclass_scores = defaultdict(float)
    eclass_support = defaultdict(int)
    for bid, w in pool_weights.items():
        for ec, share in neighbor_eclass_share.get(bid, []):
            if ec in existing_eclasses:
                continue
            eclass_scores[ec] += w * share
            eclass_support[ec] += 1
    result = {}
    for ec, score in eclass_scores.items():
        if eclass_support[ec] >= COLD_NEIGHBOR_MIN_SUPPORT:
            result[ec] = {
                'score': score, 'support': eclass_support[ec],
                'spend': eclass_avg_spend.get(ec, 0), 'price': eclass_avg_price.get(ec, 0),
            }
    return result


# Generate cold candidates
print("  Generating cold candidates...")
cold_rows = []
for _, buyer in test_buyers[test_buyers['legal_entity_id'].isin(cold_ids)].iterrows():
    bid = buyer['legal_entity_id']
    nace_p = normalize_nace(buyer.get('nace_code', ''))
    nace_s = normalize_nace(buyer.get('secondary_nace_code', '')) if 'secondary_nace_code' in buyer.index else ''
    sb = employee_bucket(buyer.get('estimated_number_employees'))

    nace_candidates = generate_nace_candidates(nace_p, nace_s, sb, set(), is_cold=True)
    neighbor_candidates = get_cold_neighbor_candidates(nace_p, nace_s, sb, set())
    all_eclasses = set(nace_candidates.keys()) | set(neighbor_candidates.keys())

    for eclass in all_eclasses:
        nace_info = nace_candidates.get(eclass)
        neighbor_info = neighbor_candidates.get(eclass)
        buyer_frac = nace_info['buyer_frac'] if nace_info else 0
        spend = max(nace_info['spend'] if nace_info else 0, neighbor_info['spend'] if neighbor_info else 0)
        price = max(nace_info['price'] if nace_info else 0, neighbor_info['price'] if neighbor_info else 0)
        has_nace = nace_info is not None
        has_neighbor = neighbor_info is not None
        both = has_nace and has_neighbor

        p_core = min(max(buyer_frac * 0.9, 0.05), 0.80)
        if both:
            p_core = min(p_core * 1.3, 0.85)

        ev = p_core * spend * SAVINGS_RATE - FEE_PER_ITEM
        sf_score = np.sqrt(max(price, 0.01)) * (1.0 + buyer_frac * 3.0)
        n_sources = int(has_nace) + int(has_neighbor)
        capture_score = p_core * (0.55 * sf_score + 0.45 * np.sqrt(max(spend, 1))) * (1.0 + 0.2 * (n_sources - 1))

        source = 'cold'
        if has_neighbor and not has_nace:
            source = 'cold_neighbor'

        cold_rows.append({
            'legal_entity_id': bid, 'eclass': eclass,
            'capture_score': capture_score, 'ev': ev,
            'projected_annual': spend, 'p_core': p_core,
            'source': source, 'task': 'cold start', 'n_sources': n_sources,
        })

cold_candidates = pd.DataFrame(cold_rows) if cold_rows else pd.DataFrame(
    columns=['legal_entity_id', 'eclass', 'capture_score', 'ev', 'projected_annual', 'p_core', 'source', 'task', 'n_sources']
)
print(f"Cold candidates: {len(cold_candidates):,}")


# Hybrid for warm buyers
print("\n── Hybrid ──")
hybrid_rows = []
for _, buyer in test_buyers[test_buyers['legal_entity_id'].isin(warm_ids)].iterrows():
    bid = buyer['legal_entity_id']
    nace_p = normalize_nace(buyer.get('nace_code', ''))
    nace_s = normalize_nace(buyer.get('secondary_nace_code', '')) if 'secondary_nace_code' in buyer.index else ''
    sb = employee_bucket(buyer.get('estimated_number_employees'))
    existing = warm_buyer_eclasses.get(bid, set())

    candidates = generate_nace_candidates(nace_p, nace_s, sb, existing, is_cold=False)
    ranked = sorted(candidates.items(), key=lambda x: x[1]['score'], reverse=True)
    for eclass, info in ranked[:HYBRID_MAX_PER_BUYER]:
        p_core = min(info['buyer_frac'] * 0.7, 0.60)
        price = info['price']
        spend = info['spend']
        ev = p_core * spend * SAVINGS_RATE - FEE_PER_ITEM
        n_sources = len(info['sources'])
        sf_score = np.sqrt(max(price, 0.01)) * (1.0 + info['buyer_frac'] * 3.0)
        capture_score = p_core * (0.55 * sf_score + 0.45 * np.sqrt(max(spend, 1))) * (1.0 + 0.15 * (n_sources - 1))

        hybrid_rows.append({
            'legal_entity_id': bid, 'eclass': eclass,
            'capture_score': capture_score, 'ev': ev,
            'projected_annual': spend, 'p_core': p_core,
            'source': 'hybrid', 'task': 'predict future', 'n_sources': n_sources,
        })

hybrid_candidates = pd.DataFrame(hybrid_rows) if hybrid_rows else pd.DataFrame(
    columns=['legal_entity_id', 'eclass', 'capture_score', 'ev', 'projected_annual', 'p_core', 'source', 'task', 'n_sources']
)
print(f"Hybrid candidates: {len(hybrid_candidates):,}")


# ══════════════════════════════════════════════════════════════════════
# 8. GLOBAL PORTFOLIO SELECTION
# ══════════════════════════════════════════════════════════════════════
print("\n── Portfolio Selection ──")

warm_pool = warm_candidates[['legal_entity_id', 'eclass', 'capture_score', 'ev',
                              'projected_annual', 'p_core', 'source', 'task', 'n_sources']].copy()

all_candidates = pd.concat([warm_pool, cobuy_candidates, hierarchy_candidates, cold_candidates, hybrid_candidates], ignore_index=True)

dedup_agg = all_candidates.groupby(['legal_entity_id', 'eclass']).agg(
    capture_score=('capture_score', 'max'),
    ev=('ev', 'max'),
    projected_annual=('projected_annual', 'max'),
    p_core=('p_core', 'max'),
    source=('source', 'first'),
    task=('task', 'first'),
    n_sources=('n_sources', 'sum'),
    source_count=('source', 'nunique'),
).reset_index()

dedup_agg['capture_score'] = dedup_agg['capture_score'] * (1.0 + 0.20 * (dedup_agg['source_count'] - 1).clip(lower=0))

dedup_agg['ev_floor'] = dedup_agg['source'].map(EV_FLOOR).fillna(0.0)
all_pool = dedup_agg[dedup_agg['ev'] >= dedup_agg['ev_floor']].copy()
all_pool = all_pool.sort_values('capture_score', ascending=False).reset_index(drop=True)

print(f"Total candidate pool: {len(all_pool):,}")
for src in all_pool['source'].unique():
    n = (all_pool['source'] == src).sum()
    print(f"  {src}: {n:,}")

# Portfolio construction
selected_indices = []
selected_keys = set()
per_buyer_count = defaultdict(int)

# Pass 1: base allocation
print("  Pass 1: base allocation...")
for (task, buyer_id), group in all_pool.groupby(['task', 'legal_entity_id'], sort=False):
    base = WARM_BASE_PER_BUYER if task == 'predict future' else COLD_BASE_PER_BUYER
    top_group = group.nlargest(base, 'capture_score')
    for idx in top_group.index:
        if len(selected_indices) >= TARGET_PREDICTIONS:
            break
        key = (all_pool.loc[idx, 'legal_entity_id'], all_pool.loc[idx, 'eclass'])
        if key in selected_keys:
            continue
        selected_indices.append(idx)
        selected_keys.add(key)
        per_buyer_count[buyer_id] += 1
    if len(selected_indices) >= TARGET_PREDICTIONS:
        break

print(f"  After pass 1: {len(selected_indices):,}")

# Pass 2: global fill
print("  Pass 2: global fill...")
if len(selected_indices) < TARGET_PREDICTIONS:
    for idx in all_pool.index:
        if len(selected_indices) >= TARGET_PREDICTIONS:
            break
        key = (all_pool.loc[idx, 'legal_entity_id'], all_pool.loc[idx, 'eclass'])
        if key in selected_keys:
            continue
        buyer_id = all_pool.loc[idx, 'legal_entity_id']
        task = all_pool.loc[idx, 'task']
        cap = WARM_MAX_PER_BUYER if task == 'predict future' else COLD_MAX_PER_BUYER_GLOBAL
        if per_buyer_count[buyer_id] >= cap:
            continue
        selected_indices.append(idx)
        selected_keys.add(key)
        per_buyer_count[buyer_id] += 1

selected = all_pool.loc[selected_indices]

# ── Save ──
submission = selected[['legal_entity_id', 'eclass']].rename(columns={'legal_entity_id': 'buyer_id'})
submission = submission.drop_duplicates()
submission.to_csv(OUTPUT_PATH, index=False)

warm_count = (selected['task'] == 'predict future').sum()
cold_count = (selected['task'] == 'cold start').sum()

print(f"\n{'='*60}")
print(f"SAVED {len(submission):,} predictions to {OUTPUT_PATH}")
print(f"  Warm (incl. cobuy+hybrid): {warm_count:,} | Cold: {cold_count:,}")
print(f"  Buyers: {submission['buyer_id'].nunique()}")
print(f"  Avg preds/buyer: {len(submission)/max(submission['buyer_id'].nunique(),1):.1f}")
print(f"  Max fee exposure: EUR {len(submission) * FEE_PER_ITEM:,.0f}")
print(f"\n  Source breakdown:")
for src, cnt in selected['source'].value_counts().items():
    avg_ev = selected[selected['source'] == src]['ev'].mean()
    avg_pcore = selected[selected['source'] == src]['p_core'].mean()
    print(f"    {src}: {cnt:,} preds, avg EV={avg_ev:.1f}, avg p_core={avg_pcore:.3f}")
print(f"{'='*60}")
