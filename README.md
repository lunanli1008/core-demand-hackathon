# Core Demand Prediction — Data Mining Hackathon

**Team:** minus low-key high performers

## Challenge
Predict recurring procurement needs (Core Demand) for 100 buyers to maximize Net Economic Benefit: `Score = Savings - Fees`, where each prediction costs a €10 fixed fee.

## Level 1 Results — E-Class Predictions

| Metric | Value |
|--------|-------|
| **Net Score** | **€1,479,554.49** |
| Savings | €1,969,554.49 |
| Fees | €490,000.00 |
| Hits | 27,021 |
| Spend Captured | 78.59% |

## Architecture

```
Raw Data (8.4M orders, 65k buyers, 6k eclass codes)
        ↓
 12-Step Data Cleaning
        ↓
 Feature Engineering (50+ features per buyer×eclass)
        ↓
 ┌──────────────────┐     ┌───────────────────┐
 │  WARM START (47)  │     │  COLD START (53)   │
 │  LightGBM p(core) │     │  NACE profiles     │
 │  Co-buy graph     │     │  Neighbor CF       │
 │  Eclass hierarchy │     │  Size-bucket match │
 └────────┬─────────┘     └────────┬──────────┘
          ↓                         ↓
     Portfolio Optimization (49k predictions)
          ↓
     submission.csv → €1,479,554.49 NET
```

## Repository Structure

```
├── README.md
├── requirements.txt
├── level1/
│   ├── level1_v13_best.py       # Best L1 submission (€1,480k NET)
│   ├── level1_v17_cleaned.py    # L1 with full 12-step data cleaning
│   └── data_cleaning.ipynb      # EDA & cleaning exploration
└── level2/
    └── level2_optimizations.py  # L2 eclass+manufacturer predictions
```

## Approach

### Level 1 — Warm-Start (47 buyers with history)
- 50+ features: spend, frequency, recency, price, buyer-level, eclass-level stats
- LightGBM binary classifier (label: appears in ≥2 future months)
- Co-buy candidates from set-based co-occurrence graph
- Eclass hierarchy sharing (6-digit group siblings)

### Level 1 — Cold-Start (53 buyers, no history)
- NACE industry profiles at 4d, 3d, 2d granularity
- Size-bucket stratification (NACE × employee count)
- Neighbor collaborative filtering from similar warm buyers

### Level 2 — E-Class + Manufacturer
- Manufacturer normalization (lowercase, strip suffixes, unify unknowns)
- Spend-weighted canonical manufacturer per eclass
- Brand loyalty signal: dominant manufacturer per buyer×eclass

### Portfolio Optimization
- Economic filter: `EV = P(core) × projected_savings - €10 fee`
- Two-pass selection: base allocation per buyer → global fill by capture_score
- Empirically optimal at 49,000 predictions

### Key Insight
> The scoring function rewards **breadth over precision**. At 55% hit rate, each marginal prediction returns ~€30 net. Maximizing the candidate pool matters more than aggressive pruning.

## Quick Start

```bash
pip install pandas numpy lightgbm scikit-learn
python level1/level1_v13_best.py
```
