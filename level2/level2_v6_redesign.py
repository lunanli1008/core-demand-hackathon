#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from level1_1_optimizations import make_validation_population
import level2_1_optimizations as l2
from level2_1_optimizations import (
    DEFAULT_CLEANED_TRAIN_PATH,
    DEFAULT_CONTEXT_CACHE_PATH,
    DEFAULT_FEATURES_PER_SKU_PATH,
    DEFAULT_NACE_CODES_PATH,
    DEFAULT_PAIR_DELIMITER,
    DEFAULT_SUBMISSION_BUYER_COLUMN,
    DEFAULT_SUBMISSION_CLUSTER_COLUMN,
    FUTURE_CORE_MIN_MONTHS,
    WARM_ANNUAL_HALF_LIFE_MONTHS,
    WARM_MIN_MONTHS,
    WARM_STALE_MAX_GAP_MONTHS,
    apply_lgbm_reranker,
    build_future_core_label_maps,
    build_model_views,
    clamp,
    expected_value_eur,
    fee_budget_to_target_predictions,
    load_nace_prefix_section_maps,
    load_predict_dataframe,
    load_train_dataframe,
    pbar,
    select_cold_predictions_one_buyer,
    serialize_predicted_id,
    train_lgbm_reranker,
    warm_candidate_features,
    warm_score,
)


DEFAULT_TARGET_FEE_EUR = 300_000.0

WARM_ECLASS_STALE_MAX_MONTHS = 24
WARM_ECLASS_TOP_K = 140
WARM_ECLASS_ROUTE_BRANDS_PER_ECLASS = 6
WARM_ECLASS_ROUTE_MAX_PER_BUYER = 1400
WARM_REACTIVATION_MAX_GAP_MONTHS = 8
WARM_REACTIVATION_MIN_TOTAL_MONTHS = 4
WARM_REACTIVATION_MIN_TOTAL_SPEND = 1_000.0
WARM_REACTIVATION_TOP_K = 220

WARM_BASE_PER_BUYER = 80
COLD_BASE_PER_BUYER = 45
WARM_MAX_PER_BUYER = 1_400
COLD_MAX_PER_BUYER = 450


def sqrt_freq_score(
    avg_monthly_spend: pd.Series | np.ndarray,
    n_months: pd.Series | np.ndarray,
    n_quarters: pd.Series | np.ndarray,
    recency_weight: pd.Series | np.ndarray,
) -> np.ndarray:
    spend_signal = np.sqrt(np.clip(avg_monthly_spend, 0.01, None))
    freq_signal = 0.6 * np.asarray(n_months, dtype=float) + 0.4 * np.asarray(n_quarters, dtype=float)
    return spend_signal * freq_signal * np.asarray(recency_weight, dtype=float)


def source_priority(source: str) -> int:
    if source == "warm_pair_reactivation_v6":
        return 5
    if source == "warm_eclass_route_v6":
        return 4
    if source == "warm_eclass_exact_v6":
        return 4
    if source == "warm_eclass_neighbor_v6":
        return 3
    if source == "cold_pair_neighbor_brand":
        return 3
    if source == "cold_pair_neighbor_eclass":
        return 2
    if source == "cold_pair_prior":
        return 1
    return 0


def strict_floor(source: str) -> float:
    if source == "warm_pair_reactivation_v6":
        return 8.0
    if source.startswith("warm_eclass"):
        return 2.0
    if source == "cold_pair_neighbor_brand":
        return 14.0
    if source == "cold_pair_neighbor_eclass":
        return 12.0
    if source == "cold_pair_prior":
        return 10.0
    return 0.0


def loose_floor(source: str) -> float:
    if source == "warm_pair_reactivation_v6":
        return 2.0
    if source.startswith("warm_eclass"):
        return -2.0
    if source == "cold_pair_neighbor_brand":
        return 6.0
    if source == "cold_pair_neighbor_eclass":
        return 4.0
    if source == "cold_pair_prior":
        return 2.0
    return 0.0


def compute_neighbor_pool_weights(row: Any, tier_index: dict[str, dict[str, list[str]]]) -> dict[str, float]:
    if not tier_index:
        return {}
    pool_weights: dict[str, float] = defaultdict(float)
    tier_order = [
        ("n4_size", 1.00),
        ("n3_size", 0.75),
        ("n2_size", 0.55),
        ("sec_size", 0.40),
        ("sec", 0.30),
    ]
    keys_by_tier = {
        "n4_size": {(str(row.nace_primary), str(row.size_bucket))} if str(row.nace_primary) else set(),
        "n3_size": {(str(row.nace_primary)[:3], str(row.size_bucket))} if str(row.nace_primary)[:3] else set(),
        "n2_size": {(str(row.nace_primary)[:2], str(row.size_bucket))} if str(row.nace_primary)[:2] else set(),
        "sec_size": {
            (str(getattr(row, "section_primary", "")), str(row.size_bucket)),
            (str(getattr(row, "section_secondary", "")), str(row.size_bucket)),
        }
        - {("", str(row.size_bucket))},
        "sec": {
            (str(getattr(row, "section_primary", "")), "all"),
            (str(getattr(row, "section_secondary", "")), "all"),
        }
        - {("", "all")},
    }
    for tier_name, tier_weight in tier_order:
        for key in keys_by_tier.get(tier_name, set()):
            for buyer_id in tier_index.get(tier_name, {}).get(key, []):
                if buyer_id in pool_weights:
                    pool_weights[buyer_id] += tier_weight
                elif len(pool_weights) < 150:
                    pool_weights[buyer_id] = tier_weight
        if len(pool_weights) >= 150:
            break
    ranked = sorted(pool_weights.items(), key=lambda item: item[1], reverse=True)[:150]
    return dict(ranked)


def compute_brand_neighbor_scores_from_pool(
    eclass_id: str,
    pool_weights: dict[str, float],
    buyer_brand_share: dict[str, dict[str, list[tuple[str, float]]]],
) -> tuple[dict[str, float], dict[str, int]]:
    if not pool_weights or not buyer_brand_share:
        return {}, {}
    scores: dict[str, float] = defaultdict(float)
    support: dict[str, int] = defaultdict(int)
    for buyer_id, neighbor_weight in pool_weights.items():
        for manufacturer, share in buyer_brand_share.get(buyer_id, {}).get(eclass_id, ()):
            scores[str(manufacturer)] += float(neighbor_weight) * float(share)
            support[str(manufacturer)] += 1
    return dict(scores), dict(support)


def build_warm_eclass_candidates_for_buyer(
    buyer_id: str,
    row: Any,
    views: dict[str, Any],
    enable_reactivation: bool,
) -> list[dict[str, Any]]:
    buyer_pair_stats = views["warm_pair_stats_by_buyer"].get(buyer_id, {})
    if not buyer_pair_stats:
        return []
    buyer_pair_share = views["warm_pair_share_by_buyer"].get(buyer_id, {})
    latest_month = views["latest_month"]
    recent_cutoff_month = latest_month - 11

    eclass_agg: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "total_spend": 0.0,
            "n_orders": 0,
            "month_set": set(),
            "quarter_set": set(),
            "last_month": -1,
            "recent_months": 0,
            "recent_spend": 0.0,
            "manufacturers": {},
        }
    )
    reactivation_candidates: list[dict[str, Any]] = []

    for pair_key, stats in buyer_pair_stats.items():
        eclass_id, manufacturer = pair_key
        features = warm_candidate_features(stats, recent_cutoff_month)
        total_months = int(features["total_months"])
        recency_gap = max(latest_month - int(features["last_month_ord"]), 0)
        if total_months < WARM_MIN_MONTHS or recency_gap > WARM_ECLASS_STALE_MAX_MONTHS:
            continue
        entry = eclass_agg[eclass_id]
        entry["total_spend"] += float(stats.total_spend)
        entry["n_orders"] += int(sum(stats.rows_by_month.values()))
        entry["month_set"].update(int(month) for month in stats.spend_by_month.keys())
        entry["quarter_set"].update(int((int(month) - 1) // 3) for month in stats.spend_by_month.keys())
        entry["last_month"] = max(int(entry["last_month"]), int(stats.last_month_ord))
        entry["recent_months"] += int(features["recent_months"])
        entry["recent_spend"] += float(features["recent_spend"])
        entry["manufacturers"][str(manufacturer)] = float(buyer_pair_share.get(pair_key, 0.0))

        if enable_reactivation and total_months >= WARM_REACTIVATION_MIN_TOTAL_MONTHS and recency_gap <= WARM_REACTIVATION_MAX_GAP_MONTHS:
            total_spend = float(stats.total_spend)
            if total_spend >= WARM_REACTIVATION_MIN_TOTAL_SPEND:
                pair_share = float(buyer_pair_share.get(pair_key, 0.0))
                recency_weight = float(np.exp(-np.log(2.0) * recency_gap / WARM_ANNUAL_HALF_LIFE_MONTHS))
                annualized_spend = (total_spend / max(total_months, 1)) * 12.0 * recency_weight
                expected_spend = max(float(features["recent_spend"]), 0.55 * annualized_spend)
                p_core = clamp(0.20 + 0.05 * min(total_months, 12) + 0.25 * pair_share - 0.03 * recency_gap, 0.06, 0.92)
                ev = expected_value_eur(p_core, expected_spend)
                if ev > 0:
                    reactivation_candidates.append(
                        {
                            "pair_key": pair_key,
                            "source": "warm_pair_reactivation_v6",
                            "score_proxy": total_spend,
                            "p_core": float(p_core),
                            "expected_spend": float(expected_spend),
                            "ev": float(ev),
                            "task": "predict future",
                        }
                    )

    if not eclass_agg:
        return reactivation_candidates[:WARM_REACTIVATION_TOP_K]

    manufacturer_views = views["manufacturer_views"]
    tier_index = views["cold_neighbor_views"].get("tier_index", {})
    pool_weights = compute_neighbor_pool_weights(row, tier_index)
    buyer_brand_share = manufacturer_views.get("buyer_brand_share", {})
    segment_brand_rankings = manufacturer_views.get("segment_brand_rankings", {})
    section_brand_rankings = manufacturer_views.get("section_brand_rankings", {})
    global_brand_rankings = manufacturer_views.get("global_brand_rankings", {})
    brand_share_prior = manufacturer_views.get("brand_share_prior", {})
    eclass_context_multiplier = views["eclass_context_multiplier"]
    seg_keys = views["buyer_segments"].get(buyer_id, ())
    sec_keys = views["buyer_sections"].get(buyer_id, ())

    ranked_eclasses: list[tuple[str, float, float, float]] = []
    eclass_meta: dict[str, dict[str, Any]] = {}
    for eclass_id, entry in eclass_agg.items():
        n_months = max(len(entry["month_set"]), 1)
        n_quarters = max(len(entry["quarter_set"]), 1)
        recency_gap = max(latest_month - int(entry["last_month"]), 0)
        if recency_gap > WARM_ECLASS_STALE_MAX_MONTHS:
            continue
        recency_weight = float(np.exp(-np.log(2.0) * recency_gap / WARM_ANNUAL_HALF_LIFE_MONTHS))
        avg_monthly_spend = float(entry["total_spend"]) / float(n_months)
        projected_annual = avg_monthly_spend * 12.0 * recency_weight * eclass_context_multiplier.get(eclass_id, 1.0)
        p_core = clamp(
            0.25
            + 0.20 * min(float(entry["recent_months"]) / 6.0, 1.0)
            + 0.20 * min(float(n_quarters) / 8.0, 1.0)
            + 0.15 * min(float(n_months) / 12.0, 1.0)
            + 0.10 * recency_weight
            + 0.10 * min(float(entry["n_orders"]) / 20.0, 1.0),
            0.05,
            0.95,
        )
        sf = float(
            sqrt_freq_score(
                np.array([avg_monthly_spend], dtype=float),
                np.array([n_months], dtype=float),
                np.array([n_quarters], dtype=float),
                np.array([recency_weight], dtype=float),
            )[0]
        )
        capture_score = float(p_core * (0.65 * sf + 0.35 * np.sqrt(max(projected_annual, 1.0))))
        ev = expected_value_eur(p_core, projected_annual)
        if ev <= 0:
            continue
        ranked_eclasses.append((eclass_id, capture_score, projected_annual, p_core))
        eclass_meta[eclass_id] = {
            "projected_annual": float(projected_annual),
            "p_core": float(p_core),
            "capture_score": float(capture_score),
            "existing_shares": entry["manufacturers"],
        }

    ranked_eclasses.sort(key=lambda item: item[1], reverse=True)
    output_candidates: list[dict[str, Any]] = []
    for eclass_id, eclass_capture, projected_annual, eclass_p_core in ranked_eclasses[:WARM_ECLASS_TOP_K]:
        exact_shares = eclass_meta[eclass_id]["existing_shares"]
        brand_scores: dict[str, float] = defaultdict(float)
        for manufacturer, share in exact_shares.items():
            brand_scores[str(manufacturer)] += 1.25 * float(share)
        neighbor_scores, neighbor_support = compute_brand_neighbor_scores_from_pool(
            eclass_id=eclass_id,
            pool_weights=pool_weights,
            buyer_brand_share=buyer_brand_share,
        )
        for manufacturer, score in neighbor_scores.items():
            brand_scores[str(manufacturer)] += 0.95 * float(score)
        for seg_key in seg_keys:
            for manufacturer, score in segment_brand_rankings.get((seg_key, eclass_id), []):
                brand_scores[str(manufacturer)] += 0.50 * float(score)
        for sec_key in sec_keys:
            for manufacturer, score in section_brand_rankings.get((sec_key, eclass_id), []):
                brand_scores[str(manufacturer)] += 0.30 * float(score)
        for manufacturer, score in global_brand_rankings.get(eclass_id, []):
            brand_scores[str(manufacturer)] += 0.18 * float(score)
        if not brand_scores:
            continue
        ranked_brands = sorted(brand_scores.items(), key=lambda item: item[1], reverse=True)
        total_positive_score = float(sum(max(score, 0.0) for _m, score in ranked_brands))
        if total_positive_score <= 0:
            continue

        diversity = len(exact_shares)
        max_brands = int(clamp(1.0 + np.sqrt(max(projected_annual, 1.0)) / 90.0 + min(diversity, 4) * 0.5, 1.0, float(WARM_ECLASS_ROUTE_BRANDS_PER_ECLASS)))
        for manufacturer, raw_score in ranked_brands[:max_brands]:
            normalized = float(raw_score) / total_positive_score
            exact_share = float(exact_shares.get(manufacturer, 0.0))
            neighbor_score = float(neighbor_scores.get(manufacturer, 0.0))
            share_prior = float(brand_share_prior.get((eclass_id, manufacturer), 0.0))
            expected_share = clamp(0.02 + 0.58 * normalized + 0.22 * share_prior + 0.18 * exact_share + 0.10 * neighbor_score, 0.02, 0.75)
            expected_spend = float(projected_annual) * expected_share
            p_pair = clamp(
                float(eclass_p_core) * (0.35 + 0.70 * expected_share)
                + 0.15 * exact_share
                + 0.10 * min(int(neighbor_support.get(manufacturer, 0)), 5) / 5.0,
                0.03,
                0.95,
            )
            ev = expected_value_eur(p_pair, expected_spend)
            if ev <= 0:
                continue
            if exact_share > 0:
                source = "warm_eclass_exact_v6"
            elif neighbor_score > share_prior:
                source = "warm_eclass_neighbor_v6"
            else:
                source = "warm_eclass_route_v6"
            output_candidates.append(
                {
                    "pair_key": (eclass_id, str(manufacturer)),
                    "source": source,
                    "score_proxy": float(projected_annual),
                    "p_core": float(p_pair),
                    "expected_spend": float(expected_spend),
                    "ev": float(ev),
                    "task": "predict future",
                }
            )

    if reactivation_candidates:
        reactivation_candidates.sort(key=lambda item: (item["ev"], item["score_proxy"]), reverse=True)
        output_candidates.extend(reactivation_candidates[:WARM_REACTIVATION_TOP_K])
    return output_candidates


def collect_candidate_pool(
    predict_df: pd.DataFrame,
    views: dict[str, Any],
    pair_delimiter: str,
    show_progress: bool,
    enable_reactivation: bool,
    include_legacy_warm: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in pbar(predict_df.itertuples(index=False), "Build candidate pool", len(predict_df), show_progress):
        buyer_id = str(row.legal_entity_id)
        candidate_pool: list[dict[str, Any]]
        if str(row.task) == "predict future":
            candidate_pool = build_warm_eclass_candidates_for_buyer(
                buyer_id=buyer_id,
                row=row,
                views=views,
                enable_reactivation=enable_reactivation,
            )
            if include_legacy_warm:
                candidate_pool.extend(
                    l2.select_warm_predictions_one_buyer(
                        buyer_id=buyer_id,
                        warm_pair_stats_by_buyer=views["warm_pair_stats_by_buyer"],
                        warm_pair_share_by_buyer=views["warm_pair_share_by_buyer"],
                        latest_month=views["latest_month"],
                        eclass_context_multiplier=views["eclass_context_multiplier"],
                        manufacturer_views=views["manufacturer_views"],
                        buyer_segments=views["buyer_segments"],
                        buyer_sections=views["buyer_sections"],
                    )
                )
                candidate_pool.extend(
                    l2.build_warm_neighbor_switch_candidates(
                        row=row,
                        buyer_id=buyer_id,
                        warm_pair_stats_by_buyer=views["warm_pair_stats_by_buyer"],
                        warm_pair_share_by_buyer=views["warm_pair_share_by_buyer"],
                        latest_month=views["latest_month"],
                        eclass_context_multiplier=views["eclass_context_multiplier"],
                        manufacturer_views=views["manufacturer_views"],
                        cold_neighbor_views=views["cold_neighbor_views"],
                        buyer_segments=views["buyer_segments"],
                        buyer_sections=views["buyer_sections"],
                    )
                )
        else:
            candidate_pool = select_cold_predictions_one_buyer(row, views)

        for candidate in candidate_pool:
            pair_key = tuple(candidate["pair_key"])
            source = str(candidate["source"])
            rows.append(
                {
                    "buyer_id": buyer_id,
                    "predicted_id": serialize_predicted_id(pair_key, pair_delimiter),
                    "eclass": str(pair_key[0]),
                    "manufacturer": str(pair_key[1]),
                    "source": source,
                    "task": str(candidate.get("task", row.task)),
                    "score_proxy": float(candidate.get("score_proxy", 0.0)),
                    "p_core": float(candidate.get("p_core", 0.0)),
                    "expected_spend": float(candidate.get("expected_spend", 0.0)),
                    "ev": float(candidate.get("ev", 0.0)),
                    "strict_floor": float(strict_floor(source)),
                    "loose_floor": float(loose_floor(source)),
                    "source_priority": int(source_priority(source)),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "buyer_id",
                "predicted_id",
                "eclass",
                "manufacturer",
                "source",
                "task",
                "score_proxy",
                "p_core",
                "expected_spend",
                "ev",
                "source_count",
                "capture_score",
                "strict_floor",
                "loose_floor",
                "source_priority",
                "is_warm",
            ]
        )

    raw_df = pd.DataFrame(rows)
    primary_df = raw_df.sort_values(
        ["buyer_id", "predicted_id", "source_priority", "ev", "score_proxy"],
        ascending=[True, True, False, False, False],
    ).drop_duplicates(["buyer_id", "predicted_id"], keep="first")[
        ["buyer_id", "predicted_id", "eclass", "manufacturer", "source", "task"]
    ].rename(columns={"source": "primary_source"})
    grouped = (
        raw_df.groupby(["buyer_id", "predicted_id"], observed=True)
        .agg(
            score_proxy=("score_proxy", "max"),
            p_core=("p_core", "max"),
            expected_spend=("expected_spend", "max"),
            ev=("ev", "max"),
            source_count=("source", "nunique"),
            source_priority=("source_priority", "max"),
            strict_floor=("strict_floor", "min"),
            loose_floor=("loose_floor", "min"),
        )
        .reset_index()
    )
    candidate_df = grouped.merge(primary_df, on=["buyer_id", "predicted_id"], how="left")
    candidate_df["source"] = candidate_df["primary_source"].fillna("")
    candidate_df = candidate_df.drop(columns=["primary_source"])
    candidate_df["capture_signal"] = candidate_df["p_core"] * candidate_df["expected_spend"]
    candidate_df["capture_score"] = candidate_df["capture_signal"] * (
        1.0 + 0.20 * (candidate_df["source_count"] - 1.0).clip(lower=0.0) + 0.05 * candidate_df["source_priority"]
    ) + 0.05 * candidate_df["score_proxy"]
    candidate_df["is_warm"] = (candidate_df["task"] == "predict future").astype("int8")
    return candidate_df.reset_index(drop=True)


def select_portfolio_v6(candidate_df: pd.DataFrame, target_predictions: int) -> pd.DataFrame:
    if candidate_df.empty or target_predictions <= 0:
        return candidate_df.iloc[0:0].copy()
    pool = candidate_df.copy()
    pool = pool.sort_values(["capture_score", "ev", "expected_spend"], ascending=[False, False, False]).reset_index(drop=True)
    selected_indices: list[int] = []
    selected_keys: set[tuple[str, str]] = set()
    per_buyer_count: dict[str, int] = defaultdict(int)

    for (task, buyer_id), group in pool.groupby(["task", "buyer_id"], sort=False):
        base = WARM_BASE_PER_BUYER if task == "predict future" else COLD_BASE_PER_BUYER
        for idx in group.nlargest(base, "capture_score").index:
            if len(selected_indices) >= target_predictions:
                break
            key = (str(pool.loc[idx, "buyer_id"]), str(pool.loc[idx, "predicted_id"]))
            if key in selected_keys:
                continue
            selected_indices.append(int(idx))
            selected_keys.add(key)
            per_buyer_count[str(buyer_id)] += 1
        if len(selected_indices) >= target_predictions:
            break

    if len(selected_indices) < target_predictions:
        for idx in pool.index:
            if len(selected_indices) >= target_predictions:
                break
            key = (str(pool.loc[idx, "buyer_id"]), str(pool.loc[idx, "predicted_id"]))
            if key in selected_keys:
                continue
            buyer_id = str(pool.loc[idx, "buyer_id"])
            task = str(pool.loc[idx, "task"])
            cap = WARM_MAX_PER_BUYER if task == "predict future" else COLD_MAX_PER_BUYER
            if per_buyer_count[buyer_id] >= cap:
                continue
            selected_indices.append(int(idx))
            selected_keys.add(key)
            per_buyer_count[buyer_id] += 1

    return pool.loc[selected_indices].sort_values(["buyer_id", "predicted_id"]).reset_index(drop=True)


def train_reranker_for_pipeline(
    train_df: pd.DataFrame,
    pair_delimiter: str,
    cleaned_train_path: Path,
    features_per_sku_path: Path,
    context_cache_path: Path,
    show_progress: bool,
    enable_reactivation: bool,
    include_legacy_warm: bool,
) -> Any | None:
    cutoff_date = "2025-06-30"
    hist_df = train_df[train_df["orderdate"] <= cutoff_date].copy()
    future_df = train_df[train_df["orderdate"] > cutoff_date].copy()
    if hist_df.empty or future_df.empty:
        return None
    val_predict_df = make_validation_population(hist_df, future_df, train_df)
    if val_predict_df.empty:
        return None
    val_views = build_model_views(
        hist_df,
        val_predict_df,
        cleaned_train_path=cleaned_train_path,
        features_per_sku_path=features_per_sku_path,
        context_cache_path=context_cache_path,
        show_progress=show_progress,
    )
    pool_df = collect_candidate_pool(
        predict_df=val_predict_df,
        views=val_views,
        pair_delimiter=pair_delimiter,
        show_progress=show_progress,
        enable_reactivation=enable_reactivation,
        include_legacy_warm=include_legacy_warm,
    )
    core_set, core_spend_map = build_future_core_label_maps(future_df, FUTURE_CORE_MIN_MONTHS)
    if not core_set:
        return None
    print(f"Training LightGBM reranker on {len(pool_df):,} candidate rows...")
    return train_lgbm_reranker(pool_df, core_set, core_spend_map)


def run_pipeline(
    train_path: Path,
    test_path: Path,
    output_path: Path,
    nace_codes_path: Path,
    cleaned_train_path: Path,
    features_per_sku_path: Path,
    context_cache_path: Path,
    pair_delimiter: str,
    submission_buyer_column: str,
    submission_cluster_column: str,
    target_fee_eur: float,
    show_progress: bool,
    enable_reactivation: bool,
    include_legacy_warm: bool,
    use_reranker: bool,
) -> pd.DataFrame:
    prefix_maps = load_nace_prefix_section_maps(nace_codes_path)
    train_df = load_train_dataframe(train_path, prefix_maps)
    test_df = load_predict_dataframe(test_path, prefix_maps)
    target_predictions = fee_budget_to_target_predictions(target_fee_eur)
    print(f"Portfolio target: {target_predictions:,} predictions (fee budget EUR {target_fee_eur:,.2f})")

    reranker_model = None
    l2.ENABLE_WARM_PAIR_SWITCH = include_legacy_warm
    l2.ENABLE_WARM_NEIGHBOR_SWITCH = include_legacy_warm
    if use_reranker:
        reranker_model = train_reranker_for_pipeline(
            train_df=train_df,
            pair_delimiter=pair_delimiter,
            cleaned_train_path=cleaned_train_path,
            features_per_sku_path=features_per_sku_path,
            context_cache_path=context_cache_path,
            show_progress=show_progress,
            enable_reactivation=enable_reactivation,
            include_legacy_warm=include_legacy_warm,
        )
        if reranker_model is not None:
            print("Reranker trained and enabled for ranking.")

    views = build_model_views(
        train_df,
        test_df,
        cleaned_train_path=cleaned_train_path,
        features_per_sku_path=features_per_sku_path,
        context_cache_path=context_cache_path,
        show_progress=show_progress,
    )
    candidate_df = collect_candidate_pool(
        predict_df=test_df,
        views=views,
        pair_delimiter=pair_delimiter,
        show_progress=show_progress,
        enable_reactivation=enable_reactivation,
        include_legacy_warm=include_legacy_warm,
    )
    candidate_df = apply_lgbm_reranker(candidate_df, reranker_model)
    selected = select_portfolio_v6(candidate_df, target_predictions)
    submission = selected[["buyer_id", "predicted_id"]].rename(
        columns={"buyer_id": submission_buyer_column, "predicted_id": submission_cluster_column}
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return submission


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Level 2 v6-style eclass-first redesign")
    parser.add_argument("--train-path", type=Path, default=Path("plis_training.csv"))
    parser.add_argument("--test-path", type=Path, default=Path("customer_test.csv"))
    parser.add_argument("--output-path", type=Path, default=Path("submission_level2_v6.csv"))
    parser.add_argument("--nace-codes-path", type=Path, default=DEFAULT_NACE_CODES_PATH)
    parser.add_argument("--cleaned-train-path", type=Path, default=DEFAULT_CLEANED_TRAIN_PATH)
    parser.add_argument("--features-per-sku-path", type=Path, default=DEFAULT_FEATURES_PER_SKU_PATH)
    parser.add_argument("--context-cache-path", type=Path, default=DEFAULT_CONTEXT_CACHE_PATH)
    parser.add_argument("--pair-delimiter", default=DEFAULT_PAIR_DELIMITER)
    parser.add_argument("--submission-buyer-column", default=DEFAULT_SUBMISSION_BUYER_COLUMN)
    parser.add_argument("--submission-cluster-column", default=DEFAULT_SUBMISSION_CLUSTER_COLUMN)
    parser.add_argument("--target-fee-eur", type=float, default=DEFAULT_TARGET_FEE_EUR)
    parser.add_argument("--enable-reactivation", action="store_true")
    parser.add_argument("--include-legacy-warm", action="store_true")
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    submission = run_pipeline(
        train_path=args.train_path,
        test_path=args.test_path,
        output_path=args.output_path,
        nace_codes_path=args.nace_codes_path,
        cleaned_train_path=args.cleaned_train_path,
        features_per_sku_path=args.features_per_sku_path,
        context_cache_path=args.context_cache_path,
        pair_delimiter=args.pair_delimiter,
        submission_buyer_column=args.submission_buyer_column,
        submission_cluster_column=args.submission_cluster_column,
        target_fee_eur=args.target_fee_eur,
        show_progress=not args.no_progress,
        enable_reactivation=bool(args.enable_reactivation),
        include_legacy_warm=bool(args.include_legacy_warm),
        use_reranker=not args.no_reranker,
    )
    print("=== Final Submission ===")
    print(f"rows: {len(submission)} | buyers: {submission[args.submission_buyer_column].nunique()}")
    print(f"columns: {submission.columns.tolist()}")
    print(f"duplicate rows: {int(submission.duplicated().sum())}")
    print(f"wrote: {args.output_path}")


if __name__ == "__main__":
    main()
