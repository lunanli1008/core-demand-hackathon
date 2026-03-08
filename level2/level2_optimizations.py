#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from level1_1_optimizations import (
    CSV_KWARGS,
    DEFAULT_CLEANED_TRAIN_PATH,
    DEFAULT_CUTOFF_DATE,
    DEFAULT_FEATURES_PER_SKU_PATH,
    DEFAULT_NACE_CODES_PATH,
    FEE_PER_PREDICTION_EUR,
    FUTURE_CORE_MIN_MONTHS,
    WarmPairStats,
    attach_section_columns,
    build_cold_neighbor_views,
    build_eclass_context_multiplier,
    build_target_segments,
    clamp,
    compute_buyer_totals_and_segments,
    compute_priors,
    employee_bucket,
    expected_value_eur,
    fee_budget_to_target_predictions,
    load_nace_prefix_section_maps,
    make_validation_population,
    normalize_nace,
    normalize_score_dict,
    ordered_section_keys,
    ordered_segment_keys,
    pbar,
    profile_keys_for_neighbor_tiers,
    warm_candidate_features,
    warm_score,
)


TRAIN_COLUMNS = [
    "orderdate",
    "legal_entity_id",
    "eclass",
    "manufacturer",
    "quantityvalue",
    "vk_per_item",
    "estimated_number_employees",
    "nace_code",
    "secondary_nace_code",
]
TEST_COLUMNS = [
    "legal_entity_id",
    "estimated_number_employees",
    "nace_code",
    "secondary_nace_code",
    "task",
]

# Warm-start selection knobs
WARM_MIN_MONTHS = 1
WARM_STALE_MAX_GAP_MONTHS = 12
WARM_MIN_K = 2
WARM_MAX_K = 280
WARM_COVERAGE_TARGET = 1.00
WARM_MIN_SCORE_RATIO = 0.00
WARM_SINGLE_MONTH_MIN_SHARE = 0.20
WARM_HEDGE_TOP_ECLASSES = 0
WARM_HEDGE_MAX_ADDITIONS_PER_ECLASS = 2
WARM_HEDGE_MIN_DIVERSITY_MAX_SHARE = 0.85
WARM_HEDGE_MIN_BRAND_SCORE = 0.03
WARM_AUGMENT_MAX_ADDITIONS = 40
WARM_AUGMENT_MIN_EV_EUR = 8.0

# Cold-start selection knobs
COLD_CANDIDATES_PER_SEGMENT = 40
COLD_ECLASS_TOP_K = 10
COLD_PAIR_TOP_K = 12
COLD_MAX_MANUFACTURERS_PER_ECLASS = 1
COLD_SECOND_BRAND_MIN_REL_SCORE = 0.65
COLD_THIRD_BRAND_MIN_REL_SCORE = 0.82
BRAND_CANDIDATES_PER_BUCKET = 12
MANUFACTURER_TOP_PER_BUYER_ECLASS = 5

# Cold-neighbor conservative gates
COLD_NEIGHBOR_POOL_CAP = 150
COLD_NEIGHBOR_MIN_SUPPORT = 8
COLD_NEIGHBOR_SCORE_BLEND = 0.60
BRAND_NEIGHBOR_MIN_SUPPORT = 3
BRAND_NEIGHBOR_SCORE_BLEND = 0.85
BRAND_NEIGHBOR_BONUS = 0.85

# Context blending
PAIR_BRAND_SHARE_FALLBACK = 0.15

# Output / cache defaults
DEFAULT_CONTEXT_CACHE_PATH = Path("docs/level2_eclass_context_cache.csv")
DEFAULT_MISS_REPORT_DIR = Path("docs/level2")
DEFAULT_PAIR_DELIMITER = "|"
DEFAULT_SUBMISSION_BUYER_COLUMN = "legal_entity_id"
DEFAULT_SUBMISSION_CLUSTER_COLUMN = "cluster"
DEFAULT_TARGET_FEE_EUR = 80_000.0

# Candidate guard rails
WARM_HARD_GUARD_MAX_K = 320
COLD_HARD_GUARD_MAX_K = 20

# Portfolio construction (level1_4-style)
GLOBAL_WARM_TARGET_SHARE = 0.88
GLOBAL_COLD_MIN_SHARE = 0.00
GLOBAL_MAX_WARM_PER_BUYER = 220
GLOBAL_MAX_COLD_PER_BUYER = 20
GLOBAL_MIN_WARM_PER_BUYER = 4

# Strict pass floors
STRICT_EV_FLOOR_WARM_PAIR = 0.0
STRICT_EV_FLOOR_WARM_PAIR_HEDGE = 12.0
STRICT_EV_FLOOR_COLD_PAIR_PRIOR = 10.0
STRICT_EV_FLOOR_COLD_PAIR_NEIGHBOR_ECLASS = 12.0
STRICT_EV_FLOOR_COLD_PAIR_NEIGHBOR_BRAND = 14.0

# Loose backfill floors
LOOSE_EV_FLOOR_WARM_PAIR = -8.0
LOOSE_EV_FLOOR_WARM_PAIR_HEDGE = 5.0
LOOSE_EV_FLOOR_COLD_PAIR_PRIOR = 2.0
LOOSE_EV_FLOOR_COLD_PAIR_NEIGHBOR_ECLASS = 4.0
LOOSE_EV_FLOOR_COLD_PAIR_NEIGHBOR_BRAND = 6.0

PairKey = tuple[str, str]


def normalize_manufacturer(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def manufacturer_normalization_key(value: object) -> str:
    text = normalize_manufacturer(value)
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def canonicalize_manufacturer_series(manufacturer: pd.Series, spend: pd.Series) -> pd.Series:
    raw = manufacturer.map(normalize_manufacturer)
    norm_key = raw.map(manufacturer_normalization_key)
    nonempty = (raw != "") & (norm_key != "")
    if not nonempty.any():
        return raw.astype("string")

    alias_df = pd.DataFrame(
        {
            "manufacturer_raw": raw[nonempty].astype("string"),
            "manufacturer_key": norm_key[nonempty].astype("string"),
            "spend": spend[nonempty].astype("float64"),
        }
    )
    alias_rank = (
        alias_df.groupby(["manufacturer_key", "manufacturer_raw"], observed=True)["spend"]
        .agg(total_spend="sum", rows="size")
        .reset_index()
    )
    alias_rank = alias_rank.sort_values(
        ["manufacturer_key", "total_spend", "rows", "manufacturer_raw"],
        ascending=[True, False, False, True],
    )
    canonical = alias_rank.drop_duplicates("manufacturer_key", keep="first")
    canonical_map = {
        str(row.manufacturer_key): str(row.manufacturer_raw) for row in canonical.itertuples(index=False)
    }

    canonicalized = raw.copy()
    canonicalized.loc[nonempty] = norm_key.loc[nonempty].map(canonical_map).fillna(raw.loc[nonempty])
    return canonicalized.astype("string")


def serialize_predicted_id(pair_key: PairKey, pair_delimiter: str) -> str:
    return f"{pair_key[0]}{pair_delimiter}{pair_key[1]}"


def strict_ev_floor_for_source(source: str) -> float:
    if source == "warm_pair":
        return STRICT_EV_FLOOR_WARM_PAIR
    if source == "warm_pair_hedge":
        return STRICT_EV_FLOOR_WARM_PAIR_HEDGE
    if source == "cold_pair_prior":
        return STRICT_EV_FLOOR_COLD_PAIR_PRIOR
    if source == "cold_pair_neighbor_eclass":
        return STRICT_EV_FLOOR_COLD_PAIR_NEIGHBOR_ECLASS
    if source == "cold_pair_neighbor_brand":
        return STRICT_EV_FLOOR_COLD_PAIR_NEIGHBOR_BRAND
    if source.startswith("warm_augment"):
        return 8.0
    return 0.0


def loose_ev_floor_for_source(source: str) -> float:
    if source == "warm_pair":
        return LOOSE_EV_FLOOR_WARM_PAIR
    if source == "warm_pair_hedge":
        return LOOSE_EV_FLOOR_WARM_PAIR_HEDGE
    if source == "cold_pair_prior":
        return LOOSE_EV_FLOOR_COLD_PAIR_PRIOR
    if source == "cold_pair_neighbor_eclass":
        return LOOSE_EV_FLOOR_COLD_PAIR_NEIGHBOR_ECLASS
    if source == "cold_pair_neighbor_brand":
        return LOOSE_EV_FLOOR_COLD_PAIR_NEIGHBOR_BRAND
    if source.startswith("warm_augment"):
        return 2.0
    return 0.0


def source_priority(source: str) -> int:
    if source == "warm_pair":
        return 4
    if source == "warm_pair_hedge":
        return 1
    if source.startswith("warm_augment"):
        return 2
    if source == "cold_pair_neighbor_brand":
        return 2
    if source == "cold_pair_neighbor_eclass":
        return 1
    if source == "cold_pair_prior":
        return 0
    return 0


def buyer_cap_for_task(task: str) -> int:
    if task == "predict future":
        return GLOBAL_MAX_WARM_PER_BUYER
    return GLOBAL_MAX_COLD_PER_BUYER


def load_train_dataframe(train_path: Path, prefix_maps: dict[int, dict[str, str]]) -> pd.DataFrame:
    train_df = pd.read_csv(train_path, usecols=TRAIN_COLUMNS, **CSV_KWARGS)
    train_df["legal_entity_id"] = train_df["legal_entity_id"].astype("string")
    train_df["eclass"] = train_df["eclass"].fillna("").astype("string").str.strip()
    train_df["manufacturer"] = train_df["manufacturer"].map(normalize_manufacturer).astype("string")

    qty = pd.to_numeric(train_df["quantityvalue"], errors="coerce").fillna(0.0)
    price = pd.to_numeric(train_df["vk_per_item"], errors="coerce").fillna(0.0)
    train_df["spend"] = qty * price
    train_df["manufacturer"] = canonicalize_manufacturer_series(train_df["manufacturer"], train_df["spend"])

    dates = train_df["orderdate"].fillna("").astype("string")
    train_df["month_ord"] = (
        dates.str.slice(0, 4).astype("int32") * 12 + dates.str.slice(5, 7).astype("int32") - 1
    )
    train_df["nace_primary"] = train_df["nace_code"].map(normalize_nace)
    train_df["nace_secondary"] = train_df["secondary_nace_code"].map(normalize_nace)
    train_df["size_bucket"] = train_df["estimated_number_employees"].map(employee_bucket)
    attach_section_columns(train_df, prefix_maps)
    return train_df


def load_predict_dataframe(test_path: Path, prefix_maps: dict[int, dict[str, str]]) -> pd.DataFrame:
    predict_df = pd.read_csv(test_path, usecols=TEST_COLUMNS, **CSV_KWARGS)
    predict_df["legal_entity_id"] = predict_df["legal_entity_id"].astype("string")
    predict_df["nace_primary"] = predict_df["nace_code"].map(normalize_nace)
    predict_df["nace_secondary"] = predict_df["secondary_nace_code"].map(normalize_nace)
    predict_df["size_bucket"] = predict_df["estimated_number_employees"].map(employee_bucket)
    attach_section_columns(predict_df, prefix_maps)
    return predict_df


def compute_warm_pair_stats(
    train_df: pd.DataFrame,
    warm_buyers: set[str],
    show_progress: bool,
) -> tuple[dict[str, dict[PairKey, WarmPairStats]], dict[str, dict[PairKey, float]], int]:
    warm_rows = train_df[
        train_df["legal_entity_id"].isin(warm_buyers)
        & (train_df["eclass"] != "")
        & (train_df["manufacturer"] != "")
    ]
    warm_monthly = (
        warm_rows.groupby(["legal_entity_id", "eclass", "manufacturer", "month_ord"], observed=True)["spend"]
        .agg(month_spend="sum", row_count="size")
        .reset_index()
    )

    warm_stats_by_buyer: dict[str, dict[PairKey, WarmPairStats]] = defaultdict(dict)
    for row in pbar(warm_monthly.itertuples(index=False), "Build warm pair stats", len(warm_monthly), show_progress):
        buyer_id = str(row.legal_entity_id)
        pair_key = (str(row.eclass), str(row.manufacturer))
        stats = warm_stats_by_buyer[buyer_id].get(pair_key)
        if stats is None:
            stats = WarmPairStats()
            warm_stats_by_buyer[buyer_id][pair_key] = stats

        month_value = int(row.month_ord)
        month_spend = float(row.month_spend)
        row_count = int(row.row_count)

        stats.total_spend += month_spend
        stats.spend_by_month[month_value] += month_spend
        stats.rows_by_month[month_value] += row_count
        if month_value > stats.last_month_ord:
            stats.last_month_ord = month_value

    pair_totals = (
        warm_monthly.groupby(["legal_entity_id", "eclass", "manufacturer"], observed=True)["month_spend"]
        .sum()
        .reset_index(name="pair_spend")
    )
    eclass_totals = (
        pair_totals.groupby(["legal_entity_id", "eclass"], observed=True)["pair_spend"]
        .sum()
        .reset_index(name="eclass_spend")
    )
    pair_totals = pair_totals.merge(eclass_totals, on=["legal_entity_id", "eclass"], how="left")
    pair_totals["pair_share"] = pair_totals["pair_spend"] / pair_totals["eclass_spend"].where(
        pair_totals["eclass_spend"] > 0,
        1.0,
    )

    warm_pair_share_by_buyer: dict[str, dict[PairKey, float]] = defaultdict(dict)
    for row in pbar(pair_totals.itertuples(index=False), "Build warm pair shares", len(pair_totals), show_progress):
        buyer_id = str(row.legal_entity_id)
        pair_key = (str(row.eclass), str(row.manufacturer))
        warm_pair_share_by_buyer[buyer_id][pair_key] = float(row.pair_share)

    latest_month = int(train_df["month_ord"].max())
    return warm_stats_by_buyer, warm_pair_share_by_buyer, latest_month


def _empty_brand_views() -> dict[str, Any]:
    return {
        "segment_brand_rankings": {},
        "section_brand_rankings": {},
        "global_brand_rankings": {},
        "pair_spend_prior": {},
        "brand_share_prior": {},
        "buyer_brand_share": {},
    }


def build_manufacturer_views(
    train_df: pd.DataFrame,
    buyer_segments: dict[str, tuple[tuple[str, str], ...]],
    buyer_sections: dict[str, tuple[tuple[str, str], ...]],
    show_progress: bool,
) -> dict[str, Any]:
    pair_rows = (
        train_df[(train_df["eclass"] != "") & (train_df["manufacturer"] != "")]
        .groupby(["legal_entity_id", "eclass", "manufacturer"], observed=True)["spend"]
        .sum()
        .reset_index()
    )
    if pair_rows.empty:
        return _empty_brand_views()

    pair_spend_prior_series = (
        pair_rows.groupby(["eclass", "manufacturer"], observed=True)["spend"].median().astype(float)
    )
    pair_spend_prior = {
        (str(eclass_id), str(manufacturer)): float(spend)
        for (eclass_id, manufacturer), spend in pair_spend_prior_series.items()
    }

    eclass_totals = (
        pair_rows.groupby(["legal_entity_id", "eclass"], observed=True)["spend"].sum().reset_index(name="eclass_spend")
    )
    pair_rows = pair_rows.merge(eclass_totals, on=["legal_entity_id", "eclass"], how="left")
    pair_rows["share_within_eclass"] = pair_rows["spend"] / pair_rows["eclass_spend"].where(
        pair_rows["eclass_spend"] > 0,
        1.0,
    )
    pair_rows = pair_rows.sort_values(
        ["legal_entity_id", "eclass", "share_within_eclass", "spend"],
        ascending=[True, True, False, False],
    )
    pair_rows["rank"] = pair_rows.groupby(["legal_entity_id", "eclass"], observed=True).cumcount()
    top_pair_rows = pair_rows[pair_rows["rank"] < MANUFACTURER_TOP_PER_BUYER_ECLASS].copy()

    buyer_brand_share: dict[str, dict[str, list[tuple[str, float]]]] = defaultdict(lambda: defaultdict(list))
    for row in pbar(
        top_pair_rows.itertuples(index=False),
        "Build buyer brand shares",
        len(top_pair_rows),
        show_progress,
    ):
        buyer_brand_share[str(row.legal_entity_id)][str(row.eclass)].append(
            (str(row.manufacturer), float(row.share_within_eclass))
        )

    eclass_buyer_counts = (
        eclass_totals.groupby("eclass", observed=True)["legal_entity_id"].nunique().astype(int).to_dict()
    )

    global_brand_df = (
        top_pair_rows.groupby(["eclass", "manufacturer"], observed=True)["share_within_eclass"]
        .sum()
        .reset_index(name="share_sum")
    )
    global_brand_df["buyer_count"] = global_brand_df["eclass"].map(eclass_buyer_counts).clip(lower=1)
    global_brand_df["score"] = global_brand_df["share_sum"] / global_brand_df["buyer_count"]
    global_brand_df = global_brand_df.sort_values(
        ["eclass", "score", "share_sum"],
        ascending=[True, False, False],
    )
    brand_share_prior = {
        (str(row.eclass), str(row.manufacturer)): float(row.score) for row in global_brand_df.itertuples(index=False)
    }
    global_brand_df["rank"] = global_brand_df.groupby("eclass", observed=True).cumcount()
    global_brand_df = global_brand_df[global_brand_df["rank"] < BRAND_CANDIDATES_PER_BUCKET]

    global_brand_rankings: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in pbar(
        global_brand_df.itertuples(index=False),
        "Build global brand rankings",
        len(global_brand_df),
        show_progress,
    ):
        global_brand_rankings[str(row.eclass)].append((str(row.manufacturer), float(row.score)))

    segment_brand_rankings: dict[tuple[tuple[str, str], str], list[tuple[str, float]]] = defaultdict(list)
    segment_rows = [(buyer_id, seg_key) for buyer_id, segs in buyer_segments.items() for seg_key in segs]
    if segment_rows:
        buyer_segment_df = pd.DataFrame(segment_rows, columns=["legal_entity_id", "seg_key"])
        seg_eclass_counts = (
            eclass_totals[["legal_entity_id", "eclass"]]
            .merge(buyer_segment_df, on="legal_entity_id", how="inner")
            .groupby(["seg_key", "eclass"], observed=True)["legal_entity_id"]
            .nunique()
            .reset_index(name="buyer_count")
        )
        seg_brand_grouped = (
            top_pair_rows[["legal_entity_id", "eclass", "manufacturer", "share_within_eclass"]]
            .merge(buyer_segment_df, on="legal_entity_id", how="inner")
            .groupby(["seg_key", "eclass", "manufacturer"], observed=True)["share_within_eclass"]
            .sum()
            .reset_index(name="share_sum")
        )
        seg_brand_grouped = seg_brand_grouped.merge(seg_eclass_counts, on=["seg_key", "eclass"], how="left")
        seg_brand_grouped["buyer_count"] = seg_brand_grouped["buyer_count"].fillna(1).clip(lower=1)
        seg_brand_grouped["score"] = seg_brand_grouped["share_sum"] / seg_brand_grouped["buyer_count"]
        seg_brand_grouped = seg_brand_grouped.sort_values(
            ["seg_key", "eclass", "score", "share_sum"],
            ascending=[True, True, False, False],
        )
        seg_brand_grouped["rank"] = seg_brand_grouped.groupby(["seg_key", "eclass"], observed=True).cumcount()
        seg_brand_grouped = seg_brand_grouped[seg_brand_grouped["rank"] < BRAND_CANDIDATES_PER_BUCKET]

        for row in pbar(
            seg_brand_grouped.itertuples(index=False),
            "Build segment brand rankings",
            len(seg_brand_grouped),
            show_progress,
        ):
            segment_brand_rankings[(row.seg_key, str(row.eclass))].append((str(row.manufacturer), float(row.score)))

    section_brand_rankings: dict[tuple[tuple[str, str], str], list[tuple[str, float]]] = defaultdict(list)
    section_rows = [(buyer_id, sec_key) for buyer_id, secs in buyer_sections.items() for sec_key in secs]
    if section_rows:
        buyer_section_df = pd.DataFrame(section_rows, columns=["legal_entity_id", "sec_key"])
        sec_eclass_counts = (
            eclass_totals[["legal_entity_id", "eclass"]]
            .merge(buyer_section_df, on="legal_entity_id", how="inner")
            .groupby(["sec_key", "eclass"], observed=True)["legal_entity_id"]
            .nunique()
            .reset_index(name="buyer_count")
        )
        sec_brand_grouped = (
            top_pair_rows[["legal_entity_id", "eclass", "manufacturer", "share_within_eclass"]]
            .merge(buyer_section_df, on="legal_entity_id", how="inner")
            .groupby(["sec_key", "eclass", "manufacturer"], observed=True)["share_within_eclass"]
            .sum()
            .reset_index(name="share_sum")
        )
        sec_brand_grouped = sec_brand_grouped.merge(sec_eclass_counts, on=["sec_key", "eclass"], how="left")
        sec_brand_grouped["buyer_count"] = sec_brand_grouped["buyer_count"].fillna(1).clip(lower=1)
        sec_brand_grouped["score"] = sec_brand_grouped["share_sum"] / sec_brand_grouped["buyer_count"]
        sec_brand_grouped = sec_brand_grouped.sort_values(
            ["sec_key", "eclass", "score", "share_sum"],
            ascending=[True, True, False, False],
        )
        sec_brand_grouped["rank"] = sec_brand_grouped.groupby(["sec_key", "eclass"], observed=True).cumcount()
        sec_brand_grouped = sec_brand_grouped[sec_brand_grouped["rank"] < BRAND_CANDIDATES_PER_BUCKET]

        for row in pbar(
            sec_brand_grouped.itertuples(index=False),
            "Build section brand rankings",
            len(sec_brand_grouped),
            show_progress,
        ):
            section_brand_rankings[(row.sec_key, str(row.eclass))].append((str(row.manufacturer), float(row.score)))

    return {
        "segment_brand_rankings": segment_brand_rankings,
        "section_brand_rankings": section_brand_rankings,
        "global_brand_rankings": global_brand_rankings,
        "pair_spend_prior": pair_spend_prior,
        "brand_share_prior": brand_share_prior,
        "buyer_brand_share": {
            buyer_id: {eclass_id: values for eclass_id, values in eclass_map.items()}
            for buyer_id, eclass_map in buyer_brand_share.items()
        },
    }


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
    keys_by_tier = profile_keys_for_neighbor_tiers(
        row.nace_primary,
        row.nace_secondary,
        row.section_primary,
        row.section_secondary,
        row.size_bucket,
    )

    for tier_name, tier_weight in tier_order:
        keys = keys_by_tier.get(tier_name, set())
        for key in keys:
            for buyer_id in tier_index.get(tier_name, {}).get(key, []):
                if buyer_id in pool_weights:
                    pool_weights[buyer_id] += tier_weight
                elif len(pool_weights) < COLD_NEIGHBOR_POOL_CAP:
                    pool_weights[buyer_id] = tier_weight
        if len(pool_weights) >= COLD_NEIGHBOR_POOL_CAP:
            break

    if len(pool_weights) > COLD_NEIGHBOR_POOL_CAP:
        ranked = sorted(pool_weights.items(), key=lambda item: item[1], reverse=True)[:COLD_NEIGHBOR_POOL_CAP]
        return dict(ranked)
    return dict(pool_weights)


def compute_eclass_neighbor_scores_from_pool(
    pool_weights: dict[str, float],
    buyer_eclass_share: dict[str, list[tuple[str, float]]],
    eclass_context_multiplier: dict[str, float],
) -> tuple[dict[str, float], dict[str, int]]:
    if not pool_weights or not buyer_eclass_share:
        return {}, {}

    scores: dict[str, float] = defaultdict(float)
    support: dict[str, int] = defaultdict(int)
    for buyer_id, neighbor_weight in pool_weights.items():
        for eclass_id, share in buyer_eclass_share.get(buyer_id, []):
            scores[eclass_id] += float(neighbor_weight) * float(share) * eclass_context_multiplier.get(eclass_id, 1.0)
            support[eclass_id] += 1

    filtered_scores: dict[str, float] = {}
    filtered_support: dict[str, int] = {}
    for eclass_id, score in scores.items():
        count = support.get(eclass_id, 0)
        if count >= COLD_NEIGHBOR_MIN_SUPPORT:
            filtered_scores[eclass_id] = float(score)
            filtered_support[eclass_id] = int(count)
    return filtered_scores, filtered_support


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
            scores[manufacturer] += float(neighbor_weight) * float(share)
            support[manufacturer] += 1

    filtered_scores: dict[str, float] = {}
    filtered_support: dict[str, int] = {}
    for manufacturer, score in scores.items():
        count = support.get(manufacturer, 0)
        if count >= BRAND_NEIGHBOR_MIN_SUPPORT:
            filtered_scores[manufacturer] = float(score)
            filtered_support[manufacturer] = int(count)
    return filtered_scores, filtered_support


def build_model_views(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    cleaned_train_path: Path,
    features_per_sku_path: Path,
    context_cache_path: Path,
    show_progress: bool,
) -> dict[str, Any]:
    warm_buyers = set(predict_df.loc[predict_df["task"] == "predict future", "legal_entity_id"].astype(str))
    target_segments = build_target_segments(predict_df, show_progress)
    (
        buyer_total_spend_series,
        buyer_total_spend,
        buyer_segments,
        segment_buyer_counts,
        buyer_sections,
        section_buyer_counts,
    ) = compute_buyer_totals_and_segments(train_df, target_segments, show_progress)

    warm_pair_stats_by_buyer, warm_pair_share_by_buyer, latest_month = compute_warm_pair_stats(
        train_df,
        warm_buyers,
        show_progress,
    )
    segment_rankings, section_rankings, global_rankings, eclass_spend_prior = compute_priors(
        train_df,
        buyer_total_spend_series,
        buyer_total_spend,
        buyer_segments,
        segment_buyer_counts,
        buyer_sections,
        section_buyer_counts,
        show_progress,
    )
    eclass_context_multiplier = build_eclass_context_multiplier(
        train_df=train_df,
        cleaned_train_path=cleaned_train_path,
        features_per_sku_path=features_per_sku_path,
        cache_path=context_cache_path,
        show_progress=show_progress,
    )
    cold_neighbor_views = build_cold_neighbor_views(
        train_df=train_df,
        buyer_total_spend_series=buyer_total_spend_series,
        show_progress=show_progress,
        enabled=True,
    )
    manufacturer_views = build_manufacturer_views(
        train_df=train_df,
        buyer_segments=buyer_segments,
        buyer_sections=buyer_sections,
        show_progress=show_progress,
    )

    return {
        "warm_pair_stats_by_buyer": warm_pair_stats_by_buyer,
        "warm_pair_share_by_buyer": warm_pair_share_by_buyer,
        "latest_month": latest_month,
        "segment_rankings": segment_rankings,
        "section_rankings": section_rankings,
        "global_rankings": global_rankings,
        "eclass_spend_prior": eclass_spend_prior,
        "eclass_context_multiplier": eclass_context_multiplier,
        "cold_neighbor_views": cold_neighbor_views,
        "manufacturer_views": manufacturer_views,
    }


def select_warm_predictions_one_buyer(
    buyer_id: str,
    warm_pair_stats_by_buyer: dict[str, dict[PairKey, WarmPairStats]],
    warm_pair_share_by_buyer: dict[str, dict[PairKey, float]],
    latest_month: int,
    eclass_context_multiplier: dict[str, float],
    manufacturer_views: dict[str, Any],
) -> list[dict[str, Any]]:
    recent_cutoff_month = latest_month - 11
    candidates: list[tuple[PairKey, float, float, float]] = []
    eclass_recent_spend: dict[str, float] = defaultdict(float)
    eclass_top_score: dict[str, float] = defaultdict(float)
    eclass_manufacturers: dict[str, set[str]] = defaultdict(set)
    eclass_max_pair_share: dict[str, float] = defaultdict(float)

    buyer_pair_stats = warm_pair_stats_by_buyer.get(buyer_id, {})
    buyer_pair_share = warm_pair_share_by_buyer.get(buyer_id, {})
    for pair_key, stats in buyer_pair_stats.items():
        eclass_id, manufacturer = pair_key
        features = warm_candidate_features(stats, recent_cutoff_month)
        if features["recent_months"] < WARM_MIN_MONTHS:
            continue
        if latest_month - int(features["last_month_ord"]) > WARM_STALE_MAX_GAP_MONTHS:
            continue

        pair_share = float(buyer_pair_share.get(pair_key, 0.0))
        if features["recent_months"] == 1 and pair_share < WARM_SINGLE_MONTH_MIN_SHARE:
            continue

        recent_spend = float(features["recent_spend"])
        eclass_recent_spend[eclass_id] += recent_spend
        eclass_manufacturers[eclass_id].add(manufacturer)
        if pair_share > eclass_max_pair_share[eclass_id]:
            eclass_max_pair_share[eclass_id] = pair_share

        context_mult = eclass_context_multiplier.get(eclass_id, 1.0)
        score = warm_score(features, latest_month) * context_mult * (0.65 + 0.70 * pair_share)
        score_float = float(score)
        candidates.append((pair_key, score_float, recent_spend, float(pair_share)))
        if score_float > eclass_top_score[eclass_id]:
            eclass_top_score[eclass_id] = score_float

    if not candidates:
        return []

    ranked_base = sorted(candidates, key=lambda item: item[1], reverse=True)
    top_score = ranked_base[0][1]
    eligible = [item for item in ranked_base if item[1] >= top_score * WARM_MIN_SCORE_RATIO]
    recent_spend_total = sum(item[2] for item in eligible if item[2] > 0)

    output_candidates: list[dict[str, Any]] = []
    covered = 0.0
    for pair_key, score_proxy, recent_spend, pair_share in eligible:
        if len(output_candidates) >= WARM_MAX_K:
            break

        stats = buyer_pair_stats[pair_key]
        features = warm_candidate_features(stats, recent_cutoff_month)
        recency_gap = max(latest_month - int(features["last_month_ord"]), 0)
        p_core = clamp(
            0.14 * features["recent_months"]
            + 0.08 * features["recent_quarters"]
            + 0.04 * min(features["total_months"], 12)
            + 0.45 * pair_share
            - 0.05 * recency_gap,
            0.02,
            0.95,
        )
        expected_spend = max(float(recent_spend), 0.0)
        ev = expected_value_eur(p_core, expected_spend)
        output_candidates.append(
            {
                "pair_key": pair_key,
                "source": "warm_pair",
                "score_proxy": float(score_proxy),
                "p_core": float(p_core),
                "expected_spend": float(expected_spend),
                "ev": float(ev),
            }
        )
        covered += recent_spend
        if len(output_candidates) >= WARM_MIN_K and recent_spend_total > 0:
            if covered / recent_spend_total >= WARM_COVERAGE_TARGET:
                break

    # Add hedge brands for warm eclasses with brand diversity.
    existing_pairs = {tuple(candidate["pair_key"]) for candidate in output_candidates}
    existing_brands_by_eclass: dict[str, set[str]] = defaultdict(set)
    for eclass_id, manufacturer in existing_pairs:
        existing_brands_by_eclass[eclass_id].add(manufacturer)

    global_brand_rankings = manufacturer_views.get("global_brand_rankings", {})
    brand_share_prior = manufacturer_views.get("brand_share_prior", {})
    ranked_eclasses = sorted(eclass_recent_spend.items(), key=lambda item: item[1], reverse=True)
    for eclass_id, recent_total in ranked_eclasses[:WARM_HEDGE_TOP_ECLASSES]:
        if recent_total <= 0:
            continue
        buyer_brand_count = len(eclass_manufacturers.get(eclass_id, set()))
        max_share = float(eclass_max_pair_share.get(eclass_id, 1.0))
        if buyer_brand_count <= 1 and max_share >= WARM_HEDGE_MIN_DIVERSITY_MAX_SHARE:
            continue

        brand_entries = global_brand_rankings.get(eclass_id, [])
        if not brand_entries:
            continue
        top_brand_score = float(brand_entries[0][1]) if brand_entries else 0.0
        additions = 0
        for manufacturer, brand_score in brand_entries:
            brand_score_f = float(brand_score)
            if brand_score_f < WARM_HEDGE_MIN_BRAND_SCORE:
                continue
            if top_brand_score > 0 and (brand_score_f / top_brand_score) < 0.45:
                continue
            if manufacturer in existing_brands_by_eclass.get(eclass_id, set()):
                continue

            pair_key = (eclass_id, str(manufacturer))
            brand_share = float(brand_share_prior.get(pair_key, brand_score_f))
            expected_spend = recent_total * clamp(brand_share, 0.05, 0.22)
            base_p_core = clamp(0.10 + 0.35 * max_share + 0.18 * brand_score_f, 0.02, 0.72)
            ev = expected_value_eur(base_p_core, expected_spend)
            output_candidates.append(
                {
                    "pair_key": pair_key,
                    "source": "warm_pair_hedge",
                    "score_proxy": float(eclass_top_score.get(eclass_id, 0.0) * (0.30 + 0.70 * brand_score_f)),
                    "p_core": float(base_p_core),
                    "expected_spend": float(expected_spend),
                    "ev": float(ev),
                }
            )
            existing_brands_by_eclass[eclass_id].add(str(manufacturer))
            additions += 1
            if additions >= WARM_HEDGE_MAX_ADDITIONS_PER_ECLASS:
                break

    output_candidates.sort(key=lambda item: (item["ev"], item["score_proxy"]), reverse=True)
    return output_candidates[:WARM_HARD_GUARD_MAX_K]


def select_cold_predictions_one_buyer(row: Any, views: dict[str, Any]) -> list[dict[str, Any]]:
    eclass_context_multiplier = views["eclass_context_multiplier"]
    manufacturer_views = views["manufacturer_views"]

    base_scores: dict[str, float] = defaultdict(float)
    for seg_key, weight in ordered_segment_keys(row.nace_primary, row.nace_secondary, row.size_bucket):
        entries = views["global_rankings"] if seg_key == ("all", "all") else views["segment_rankings"].get(seg_key, [])
        for eclass_id, base_score in entries[:COLD_CANDIDATES_PER_SEGMENT]:
            base_scores[eclass_id] += weight * base_score * eclass_context_multiplier.get(eclass_id, 1.0)

    for sec_key, weight in ordered_section_keys(row.section_primary, row.section_secondary, row.size_bucket):
        entries = views["section_rankings"].get(sec_key, [])
        for eclass_id, base_score in entries[:COLD_CANDIDATES_PER_SEGMENT]:
            base_scores[eclass_id] += weight * base_score * eclass_context_multiplier.get(eclass_id, 1.0)

    if not base_scores:
        for eclass_id, base_score in views["global_rankings"][:COLD_CANDIDATES_PER_SEGMENT]:
            base_scores[eclass_id] = float(base_score) * eclass_context_multiplier.get(eclass_id, 1.0)

    if not base_scores:
        return []

    cold_neighbor_views = views["cold_neighbor_views"]
    tier_index = cold_neighbor_views.get("tier_index", {})
    buyer_eclass_share = cold_neighbor_views.get("buyer_eclass_share", {})
    pool_weights = compute_neighbor_pool_weights(row, tier_index)
    neighbor_scores, neighbor_support = compute_eclass_neighbor_scores_from_pool(
        pool_weights=pool_weights,
        buyer_eclass_share=buyer_eclass_share,
        eclass_context_multiplier=eclass_context_multiplier,
    )

    combined_eclass_scores = dict(base_scores)
    for eclass_id, score in neighbor_scores.items():
        combined_eclass_scores[eclass_id] = combined_eclass_scores.get(eclass_id, 0.0) + COLD_NEIGHBOR_SCORE_BLEND * score

    base_norm = normalize_score_dict(base_scores)
    neighbor_norm = normalize_score_dict(neighbor_scores)
    ranked_eclass = sorted(combined_eclass_scores.items(), key=lambda item: item[1], reverse=True)

    output_candidates: list[dict[str, Any]] = []
    seen_pairs: set[PairKey] = set()
    for eclass_id, eclass_score in ranked_eclass[:COLD_ECLASS_TOP_K]:
        brand_scores: dict[str, float] = defaultdict(float)
        for seg_key, weight in ordered_segment_keys(row.nace_primary, row.nace_secondary, row.size_bucket):
            for manufacturer, base_score in manufacturer_views["segment_brand_rankings"].get((seg_key, eclass_id), []):
                brand_scores[manufacturer] += weight * base_score

        for sec_key, weight in ordered_section_keys(row.section_primary, row.section_secondary, row.size_bucket):
            for manufacturer, base_score in manufacturer_views["section_brand_rankings"].get((sec_key, eclass_id), []):
                brand_scores[manufacturer] += weight * base_score

        if not brand_scores:
            for manufacturer, base_score in manufacturer_views["global_brand_rankings"].get(eclass_id, []):
                brand_scores[manufacturer] += base_score

        if not brand_scores:
            continue

        brand_neighbor_scores, brand_neighbor_support = compute_brand_neighbor_scores_from_pool(
            eclass_id=eclass_id,
            pool_weights=pool_weights,
            buyer_brand_share=manufacturer_views["buyer_brand_share"],
        )
        combined_brand_scores = dict(brand_scores)
        for manufacturer, score in brand_neighbor_scores.items():
            combined_brand_scores[manufacturer] = combined_brand_scores.get(manufacturer, 0.0) + (
                BRAND_NEIGHBOR_SCORE_BLEND * score
            )
        if not combined_brand_scores:
            continue

        brand_norm = normalize_score_dict(brand_scores)
        brand_neighbor_norm = normalize_score_dict(brand_neighbor_scores)
        eclass_p_core = clamp(
            0.65 * float(base_norm.get(eclass_id, 0.0))
            + 0.35 * float(neighbor_norm.get(eclass_id, 0.0)),
            0.02,
            0.90,
        )
        ranked_brands = sorted(combined_brand_scores.items(), key=lambda item: item[1], reverse=True)
        if not ranked_brands:
            continue

        top_brand_score = float(ranked_brands[0][1])
        for brand_rank, (manufacturer, combined_brand_score) in enumerate(ranked_brands):
            if brand_rank >= COLD_MAX_MANUFACTURERS_PER_ECLASS:
                break

            rel_score = float(combined_brand_score) / max(top_brand_score, 1e-9)
            if brand_rank == 1 and rel_score < COLD_SECOND_BRAND_MIN_REL_SCORE:
                continue
            if brand_rank >= 2 and rel_score < COLD_THIRD_BRAND_MIN_REL_SCORE:
                continue

            pair_key = (eclass_id, str(manufacturer))
            if pair_key in seen_pairs:
                continue

            brand_conf = max(
                float(brand_norm.get(manufacturer, 0.0)),
                BRAND_NEIGHBOR_BONUS * float(brand_neighbor_norm.get(manufacturer, 0.0)),
            )
            rank_penalty = 1.0 if brand_rank == 0 else (0.86 if brand_rank == 1 else 0.74)
            p_core = clamp((0.72 * eclass_p_core + 0.28 * brand_conf) * rank_penalty, 0.02, 0.90)
            expected_spend = max(
                float(manufacturer_views["pair_spend_prior"].get(pair_key, 0.0)),
                float(views["eclass_spend_prior"].get(eclass_id, 0.0))
                * max(float(manufacturer_views["brand_share_prior"].get(pair_key, 0.0)), PAIR_BRAND_SHARE_FALLBACK)
                * rank_penalty,
            )
            ev = expected_value_eur(p_core, expected_spend)

            source = "cold_pair_prior"
            if brand_neighbor_norm.get(manufacturer, 0.0) >= brand_norm.get(manufacturer, 0.0) and brand_neighbor_support.get(
                manufacturer,
                0,
            ) >= BRAND_NEIGHBOR_MIN_SUPPORT:
                source = "cold_pair_neighbor_brand"
            elif neighbor_support.get(eclass_id, 0) >= COLD_NEIGHBOR_MIN_SUPPORT:
                source = "cold_pair_neighbor_eclass"

            output_candidates.append(
                {
                    "pair_key": pair_key,
                    "source": source,
                    "score_proxy": float(eclass_score * (0.45 + 0.55 * max(brand_conf, 0.05)) * rank_penalty),
                    "p_core": float(p_core),
                    "expected_spend": float(expected_spend),
                    "ev": float(ev),
                    "brand_score_proxy": float(combined_brand_score),
                }
            )
            seen_pairs.add(pair_key)

    output_candidates.sort(key=lambda item: (item["ev"], item["score_proxy"]), reverse=True)
    return output_candidates[: min(COLD_HARD_GUARD_MAX_K, COLD_PAIR_TOP_K)]


def collect_population_candidates(
    predict_df: pd.DataFrame,
    views: dict[str, Any],
    pair_delimiter: str,
    show_progress: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in pbar(predict_df.itertuples(index=False), "Build candidate pool", len(predict_df), show_progress):
        buyer_id = str(row.legal_entity_id)
        candidate_pool: list[dict[str, Any]] = []

        if row.task == "predict future":
            warm_candidates = select_warm_predictions_one_buyer(
                buyer_id=buyer_id,
                warm_pair_stats_by_buyer=views["warm_pair_stats_by_buyer"],
                warm_pair_share_by_buyer=views["warm_pair_share_by_buyer"],
                latest_month=views["latest_month"],
                eclass_context_multiplier=views["eclass_context_multiplier"],
                manufacturer_views=views["manufacturer_views"],
            )
            candidate_pool.extend(warm_candidates)

            # Structural recall expansion: for warm buyers, also add a few high-EV
            # pair candidates from the cold-style model when they introduce new eclasses.
            augment_candidates = select_cold_predictions_one_buyer(row, views)
            warm_pairs = {tuple(c["pair_key"]) for c in warm_candidates}
            warm_eclasses = {pair_key[0] for pair_key in warm_pairs}
            augment_added = 0
            for candidate in augment_candidates:
                pair_key = tuple(candidate["pair_key"])
                if pair_key in warm_pairs:
                    continue
                if pair_key[0] in warm_eclasses:
                    continue
                if float(candidate.get("ev", 0.0)) < WARM_AUGMENT_MIN_EV_EUR:
                    continue
                source = str(candidate.get("source", "cold_pair_prior"))
                candidate_pool.append(
                    {
                        "pair_key": pair_key,
                        "source": f"warm_augment_{source}",
                        "score_proxy": float(candidate.get("score_proxy", 0.0)),
                        "p_core": float(candidate.get("p_core", 0.0)),
                        "expected_spend": float(candidate.get("expected_spend", 0.0)),
                        "ev": float(candidate.get("ev", 0.0)),
                    }
                )
                augment_added += 1
                if augment_added >= WARM_AUGMENT_MAX_ADDITIONS:
                    break

        if not candidate_pool:
            candidate_pool.extend(select_cold_predictions_one_buyer(row, views))

        for candidate in candidate_pool:
            pair_key = candidate["pair_key"]
            source = str(candidate.get("source", "unknown"))
            rows.append(
                {
                    "buyer_id": buyer_id,
                    "predicted_id": serialize_predicted_id(pair_key, pair_delimiter),
                    "eclass": str(pair_key[0]),
                    "manufacturer": str(pair_key[1]),
                    "source": source,
                    "task": str(row.task),
                    "score_proxy": float(candidate.get("score_proxy", 0.0)),
                    "p_core": float(candidate.get("p_core", 0.0)),
                    "expected_spend": float(candidate.get("expected_spend", 0.0)),
                    "ev": float(candidate.get("ev", 0.0)),
                    "strict_floor": float(strict_ev_floor_for_source(source)),
                    "loose_floor": float(loose_ev_floor_for_source(source)),
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
                "strict_floor",
                "loose_floor",
                "source_priority",
                "is_warm",
            ]
        )

    candidate_df = pd.DataFrame(rows).sort_values(
        ["buyer_id", "predicted_id", "ev", "source_priority", "score_proxy"],
        ascending=[True, True, False, False, False],
    )
    candidate_df = candidate_df.drop_duplicates(["buyer_id", "predicted_id"], keep="first")
    candidate_df["is_warm"] = (candidate_df["task"] == "predict future").astype("int8")
    return candidate_df.reset_index(drop=True)


def select_portfolio_from_candidates(candidate_df: pd.DataFrame, target_predictions: int) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df

    target_predictions = max(int(target_predictions), 0)
    if target_predictions == 0:
        return candidate_df.iloc[0:0].copy()

    selected_indices: list[int] = []
    selected_keys: set[tuple[str, str]] = set()
    buyer_counts: dict[str, int] = defaultdict(int)
    selected_task_counts: dict[str, int] = defaultdict(int)

    def append_from_rows(rows_df: pd.DataFrame, limit: int | None = None) -> None:
        for idx, row in rows_df.iterrows():
            if len(selected_indices) >= target_predictions:
                break
            if limit is not None and len(selected_indices) >= limit:
                break
            key = (str(row["buyer_id"]), str(row["predicted_id"]))
            if key in selected_keys:
                continue
            buyer_id = str(row["buyer_id"])
            task = str(row["task"])
            if buyer_counts[buyer_id] >= buyer_cap_for_task(task):
                continue
            selected_indices.append(int(idx))
            selected_keys.add(key)
            buyer_counts[buyer_id] += 1
            selected_task_counts[task] += 1

    def append_cold_until_target(rows_df: pd.DataFrame, cold_target: int) -> None:
        if cold_target <= 0:
            return
        for idx, row in rows_df.iterrows():
            if len(selected_indices) >= target_predictions:
                break
            if selected_task_counts.get("cold start", 0) >= cold_target:
                break
            key = (str(row["buyer_id"]), str(row["predicted_id"]))
            if key in selected_keys:
                continue
            buyer_id = str(row["buyer_id"])
            task = str(row["task"])
            if task != "cold start":
                continue
            if buyer_counts[buyer_id] >= buyer_cap_for_task(task):
                continue
            selected_indices.append(int(idx))
            selected_keys.add(key)
            buyer_counts[buyer_id] += 1
            selected_task_counts[task] += 1

    warm_target = min(
        int(target_predictions * GLOBAL_WARM_TARGET_SHARE),
        int((candidate_df["task"] == "predict future").sum()),
    )
    cold_target = min(
        int(target_predictions * GLOBAL_COLD_MIN_SHARE),
        int((candidate_df["task"] == "cold start").sum()),
    )

    strict_df = candidate_df[candidate_df["ev"] >= candidate_df["strict_floor"]].copy()
    strict_df = strict_df.sort_values(
        ["is_warm", "ev", "source_priority", "score_proxy"],
        ascending=[False, False, False, False],
    )
    strict_warm_df = strict_df[strict_df["task"] == "predict future"]
    strict_cold_df = strict_df[strict_df["task"] == "cold start"]
    append_from_rows(strict_warm_df, limit=warm_target)
    append_cold_until_target(strict_cold_df, cold_target=cold_target)
    append_from_rows(strict_df)

    if len(selected_indices) < target_predictions:
        loose_df = candidate_df[candidate_df["ev"] >= candidate_df["loose_floor"]].copy()
        loose_df = loose_df.sort_values(
            ["is_warm", "ev", "source_priority", "score_proxy"],
            ascending=[False, False, False, False],
        )
        loose_warm_df = loose_df[loose_df["task"] == "predict future"]
        loose_cold_df = loose_df[loose_df["task"] == "cold start"]
        if len(selected_indices) < warm_target:
            append_from_rows(loose_warm_df, limit=warm_target)
        if selected_task_counts.get("cold start", 0) < cold_target:
            append_cold_until_target(loose_cold_df, cold_target=cold_target)
        append_from_rows(loose_df)

    if len(selected_indices) < target_predictions:
        rescue_df = candidate_df.sort_values(
            ["is_warm", "ev", "source_priority", "score_proxy"],
            ascending=[False, False, False, False],
        )
        append_from_rows(rescue_df)

    selected = candidate_df.loc[selected_indices].copy() if selected_indices else candidate_df.iloc[0:0].copy()
    if selected.empty:
        return selected

    warm_df = candidate_df[candidate_df["task"] == "predict future"]
    if not warm_df.empty:
        selected_warm_counts = selected[selected["task"] == "predict future"].groupby("buyer_id").size().to_dict()
        selected_keys = {(str(r.buyer_id), str(r.predicted_id)) for r in selected.itertuples(index=False)}
        add_rows: list[pd.Series] = []
        for buyer_id, buyer_rows in warm_df.groupby("buyer_id", observed=True):
            current = int(selected_warm_counts.get(str(buyer_id), 0))
            needed = max(GLOBAL_MIN_WARM_PER_BUYER - current, 0)
            if needed <= 0:
                continue
            if len(selected) + len(add_rows) >= target_predictions:
                break
            buyer_rows = buyer_rows.sort_values(
                ["ev", "source_priority", "score_proxy"],
                ascending=[False, False, False],
            )
            for row in buyer_rows.itertuples(index=False):
                key = (str(row.buyer_id), str(row.predicted_id))
                if key in selected_keys:
                    continue
                add_rows.append(pd.Series(row._asdict()))
                selected_keys.add(key)
                needed -= 1
                if needed <= 0 or len(selected) + len(add_rows) >= target_predictions:
                    break
        if add_rows:
            selected = pd.concat([selected, pd.DataFrame(add_rows)], ignore_index=True)

    return selected.sort_values(["buyer_id", "predicted_id"]).reset_index(drop=True)


def predict_for_population_with_sources(
    predict_df: pd.DataFrame,
    views: dict[str, Any],
    pair_delimiter: str,
    target_predictions: int,
    show_progress: bool,
) -> pd.DataFrame:
    candidate_df = collect_population_candidates(
        predict_df=predict_df,
        views=views,
        pair_delimiter=pair_delimiter,
        show_progress=show_progress,
    )
    selected = select_portfolio_from_candidates(candidate_df, target_predictions=target_predictions)
    if selected.empty:
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
            ]
        )
    return selected[
        ["buyer_id", "predicted_id", "eclass", "manufacturer", "source", "task", "score_proxy", "p_core", "expected_spend", "ev"]
    ].copy()


def predict_for_population(
    predict_df: pd.DataFrame,
    views: dict[str, Any],
    pair_delimiter: str,
    target_predictions: int,
    submission_buyer_column: str,
    submission_cluster_column: str,
    show_progress: bool,
) -> pd.DataFrame:
    pred_with_source = predict_for_population_with_sources(
        predict_df,
        views,
        pair_delimiter=pair_delimiter,
        target_predictions=target_predictions,
        show_progress=show_progress,
    )
    if pred_with_source.empty:
        return pd.DataFrame(columns=[submission_buyer_column, submission_cluster_column])
    return pred_with_source[["buyer_id", "predicted_id"]].rename(
        columns={
            "buyer_id": submission_buyer_column,
            "predicted_id": submission_cluster_column,
        }
    )


def offline_score(
    pred_with_source_df: pd.DataFrame,
    hist_df: pd.DataFrame,
    future_df: pd.DataFrame,
    future_core_min_months: int,
) -> dict[str, float]:
    fut_nonempty = future_df[(future_df["eclass"] != "") & (future_df["manufacturer"] != "")].copy()
    core_future = (
        fut_nonempty.groupby(["legal_entity_id", "eclass", "manufacturer"], observed=True)["month_ord"]
        .nunique()
        .reset_index(name="future_active_months")
    )
    core_future = core_future[core_future["future_active_months"] >= future_core_min_months]
    core_set = {
        (str(row.legal_entity_id), str(row.eclass), str(row.manufacturer)) for row in core_future.itertuples(index=False)
    }

    hist_nonempty = hist_df[(hist_df["eclass"] != "") & (hist_df["manufacturer"] != "")]
    hist_spend = (
        hist_nonempty.groupby(["legal_entity_id", "eclass", "manufacturer"], observed=True)["spend"].sum().astype(float).to_dict()
    )

    pred_set = {
        (str(row.buyer_id), str(row.eclass), str(row.manufacturer)) for row in pred_with_source_df.itertuples(index=False)
    }
    hits = pred_set & core_set

    matched_hist_spend = float(sum(hist_spend.get(key, 0.0) for key in hits))
    total_core_hist_spend = float(sum(hist_spend.get(key, 0.0) for key in core_set))
    total_savings = 0.10 * matched_hist_spend
    total_fees = FEE_PER_PREDICTION_EUR * len(pred_set)
    total_score = total_savings - total_fees
    spend_capture_rate = (matched_hist_spend / total_core_hist_spend) if total_core_hist_spend > 0 else 0.0

    return {
        "Total Score": total_score,
        "Total Savings": total_savings,
        "Total Fees": total_fees,
        "Num Hits": float(len(hits)),
        "Spend Capture Rate": spend_capture_rate,
        "Predictions": float(len(pred_set)),
        "Core Truth Pairs (proxy)": float(len(core_set)),
    }


def write_miss_reports(
    pred_with_source_df: pd.DataFrame,
    predict_population_df: pd.DataFrame,
    hist_df: pd.DataFrame,
    future_df: pd.DataFrame,
    output_dir: Path,
    future_core_min_months: int,
    pair_delimiter: str,
) -> None:
    fut_nonempty = future_df[(future_df["eclass"] != "") & (future_df["manufacturer"] != "")]
    core_future = (
        fut_nonempty.groupby(["legal_entity_id", "eclass", "manufacturer"], observed=True)["month_ord"]
        .nunique()
        .reset_index(name="future_active_months")
    )
    core_future = core_future[core_future["future_active_months"] >= future_core_min_months]
    core_set = {
        (str(row.legal_entity_id), str(row.eclass), str(row.manufacturer)) for row in core_future.itertuples(index=False)
    }

    hist_nonempty = hist_df[(hist_df["eclass"] != "") & (hist_df["manufacturer"] != "")]
    hist_spend = (
        hist_nonempty.groupby(["legal_entity_id", "eclass", "manufacturer"], observed=True)["spend"].sum().astype(float).to_dict()
    )

    pred_set = {
        (str(row.buyer_id), str(row.eclass), str(row.manufacturer)) for row in pred_with_source_df.itertuples(index=False)
    }
    false_negatives = core_set - pred_set
    false_positives = pred_set - core_set

    fn_rows = []
    for buyer_id, eclass_id, manufacturer in false_negatives:
        spend_value = float(hist_spend.get((buyer_id, eclass_id, manufacturer), 0.0))
        fn_rows.append(
            {
                "buyer_id": buyer_id,
                "predicted_id": serialize_predicted_id((eclass_id, manufacturer), pair_delimiter),
                "eclass": eclass_id,
                "manufacturer": manufacturer,
                "hist_spend": spend_value,
                "missed_savings": 0.10 * spend_value,
            }
        )
    fn_df = pd.DataFrame(
        fn_rows,
        columns=["buyer_id", "predicted_id", "eclass", "manufacturer", "hist_spend", "missed_savings"],
    )
    if not fn_df.empty:
        fn_df = fn_df.sort_values("missed_savings", ascending=False).head(20)

    task_by_buyer = (
        predict_population_df[["legal_entity_id", "task"]]
        .drop_duplicates("legal_entity_id")
        .set_index("legal_entity_id")["task"]
        .to_dict()
    )
    fp_rows = []
    for buyer_id, eclass_id, manufacturer in false_positives:
        fp_rows.append(
            {
                "buyer_id": buyer_id,
                "predicted_id": serialize_predicted_id((eclass_id, manufacturer), pair_delimiter),
                "eclass": eclass_id,
                "manufacturer": manufacturer,
                "task": task_by_buyer.get(buyer_id, "unknown"),
                "wasted_fee": FEE_PER_PREDICTION_EUR,
            }
        )
    fp_df = pd.DataFrame(
        fp_rows,
        columns=["buyer_id", "predicted_id", "eclass", "manufacturer", "task", "wasted_fee"],
    )
    if not fp_df.empty:
        fp_df = (
            fp_df.groupby(["predicted_id", "eclass", "manufacturer", "task"], observed=True)["wasted_fee"]
            .sum()
            .reset_index()
            .sort_values("wasted_fee", ascending=False)
            .head(20)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    fn_path = output_dir / "miss_report_fn.csv"
    fp_path = output_dir / "miss_report_fp.csv"
    fn_df.to_csv(fn_path, index=False)
    fp_df.to_csv(fp_path, index=False)
    print(f"Miss report written: {fn_path}")
    print(f"Miss report written: {fp_path}")


def write_source_attribution_report(
    pred_with_source_df: pd.DataFrame,
    hist_df: pd.DataFrame,
    future_df: pd.DataFrame,
    output_dir: Path,
    future_core_min_months: int,
) -> None:
    if pred_with_source_df.empty:
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            columns=[
                "source",
                "predictions",
                "hits",
                "hit_rate",
                "matched_hist_spend",
                "spend_capture_share",
                "savings_eur",
                "fees_eur",
                "net_eur",
                "avg_savings_per_prediction",
                "avg_savings_per_hit",
            ]
        ).to_csv(output_dir / "source_attribution_summary.csv", index=False)
        return

    fut_nonempty = future_df[(future_df["eclass"] != "") & (future_df["manufacturer"] != "")]
    core_future = (
        fut_nonempty.groupby(["legal_entity_id", "eclass", "manufacturer"], observed=True)["month_ord"]
        .nunique()
        .reset_index(name="future_active_months")
    )
    core_future = core_future[core_future["future_active_months"] >= future_core_min_months]
    core_set = {
        (str(row.legal_entity_id), str(row.eclass), str(row.manufacturer)) for row in core_future.itertuples(index=False)
    }

    hist_nonempty = hist_df[(hist_df["eclass"] != "") & (hist_df["manufacturer"] != "")]
    hist_spend = (
        hist_nonempty.groupby(["legal_entity_id", "eclass", "manufacturer"], observed=True)["spend"].sum().astype(float).to_dict()
    )
    total_core_hist_spend = float(sum(hist_spend.get(key, 0.0) for key in core_set))

    pred = pred_with_source_df.drop_duplicates(["buyer_id", "predicted_id"], keep="first").copy()
    keys = list(zip(pred["buyer_id"].astype(str), pred["eclass"].astype(str), pred["manufacturer"].astype(str)))
    pred["is_hit"] = [key in core_set for key in keys]
    pred["hist_spend"] = [float(hist_spend.get(key, 0.0)) for key in keys]
    pred["matched_hist_spend"] = np.where(pred["is_hit"], pred["hist_spend"], 0.0)
    pred["savings_eur"] = 0.10 * pred["matched_hist_spend"]
    pred["fees_eur"] = FEE_PER_PREDICTION_EUR
    pred["net_eur"] = pred["savings_eur"] - pred["fees_eur"]

    summary = (
        pred.groupby("source", observed=True)
        .agg(
            predictions=("predicted_id", "count"),
            hits=("is_hit", "sum"),
            matched_hist_spend=("matched_hist_spend", "sum"),
            savings_eur=("savings_eur", "sum"),
            fees_eur=("fees_eur", "sum"),
            net_eur=("net_eur", "sum"),
        )
        .reset_index()
    )
    summary["hit_rate"] = np.where(summary["predictions"] > 0, summary["hits"] / summary["predictions"], 0.0)
    summary["spend_capture_share"] = np.where(
        total_core_hist_spend > 0,
        summary["matched_hist_spend"] / total_core_hist_spend,
        0.0,
    )
    summary["avg_savings_per_prediction"] = np.where(
        summary["predictions"] > 0,
        summary["savings_eur"] / summary["predictions"],
        0.0,
    )
    summary["avg_savings_per_hit"] = np.where(summary["hits"] > 0, summary["savings_eur"] / summary["hits"], 0.0)
    summary = summary.sort_values("net_eur", ascending=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "source_attribution_summary.csv"
    by_task = (
        pred.groupby(["source", "task"], observed=True)
        .agg(predictions=("predicted_id", "count"), hits=("is_hit", "sum"), savings_eur=("savings_eur", "sum"))
        .reset_index()
    )
    by_task_path = output_dir / "source_attribution_by_task.csv"
    summary.to_csv(summary_path, index=False)
    by_task.to_csv(by_task_path, index=False)
    print(f"Source attribution written: {summary_path}")
    print(f"Source attribution written: {by_task_path}")


def run_pipeline(
    train_path: Path,
    test_path: Path,
    output_path: Path,
    nace_codes_path: Path,
    cleaned_train_path: Path,
    features_per_sku_path: Path,
    context_cache_path: Path,
    miss_report_dir: Path,
    pair_delimiter: str,
    submission_buyer_column: str,
    submission_cluster_column: str,
    cutoff_date: str = DEFAULT_CUTOFF_DATE,
    target_fee_eur: float = DEFAULT_TARGET_FEE_EUR,
    run_validation: bool = True,
    write_miss_report: bool = True,
    write_source_report: bool = True,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, dict[str, float] | None]:
    print("Loading NACE section mappings...")
    prefix_maps = load_nace_prefix_section_maps(nace_codes_path)
    print("NACE prefixes loaded:", {size: len(prefix_maps[size]) for size in sorted(prefix_maps)})

    train_df = load_train_dataframe(train_path, prefix_maps)
    test_df = load_predict_dataframe(test_path, prefix_maps)
    target_predictions = fee_budget_to_target_predictions(target_fee_eur)
    print(
        f"Portfolio target: {target_predictions:,} predictions "
        f"(fee budget EUR {target_fee_eur:,.2f})"
    )

    metrics: dict[str, float] | None = None
    if run_validation:
        hist_df = train_df[train_df["orderdate"] <= cutoff_date].copy()
        future_df = train_df[train_df["orderdate"] > cutoff_date].copy()

        val_predict_df = make_validation_population(hist_df, future_df, train_df)
        val_views = build_model_views(
            hist_df,
            val_predict_df,
            cleaned_train_path=cleaned_train_path,
            features_per_sku_path=features_per_sku_path,
            context_cache_path=context_cache_path,
            show_progress=show_progress,
        )
        val_pred_with_source = predict_for_population_with_sources(
            val_predict_df,
            val_views,
            pair_delimiter=pair_delimiter,
            target_predictions=target_predictions,
            show_progress=show_progress,
        )
        metrics = offline_score(val_pred_with_source, hist_df, future_df, FUTURE_CORE_MIN_MONTHS)
        if write_miss_report:
            write_miss_reports(
                pred_with_source_df=val_pred_with_source,
                predict_population_df=val_predict_df,
                hist_df=hist_df,
                future_df=future_df,
                output_dir=miss_report_dir,
                future_core_min_months=FUTURE_CORE_MIN_MONTHS,
                pair_delimiter=pair_delimiter,
            )
        if write_source_report:
            write_source_attribution_report(
                pred_with_source_df=val_pred_with_source,
                hist_df=hist_df,
                future_df=future_df,
                output_dir=miss_report_dir,
                future_core_min_months=FUTURE_CORE_MIN_MONTHS,
            )

    final_views = build_model_views(
        train_df,
        test_df,
        cleaned_train_path=cleaned_train_path,
        features_per_sku_path=features_per_sku_path,
        context_cache_path=context_cache_path,
        show_progress=show_progress,
    )
    submission = predict_for_population(
        test_df,
        final_views,
        pair_delimiter=pair_delimiter,
        target_predictions=target_predictions,
        submission_buyer_column=submission_buyer_column,
        submission_cluster_column=submission_cluster_column,
        show_progress=show_progress,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return submission, metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Level 2 pipeline predicting recurring (eclass, manufacturer) demand pairs"
    )
    parser.add_argument("--train-path", type=Path, default=Path("plis_training.csv"))
    parser.add_argument("--test-path", type=Path, default=Path("customer_test.csv"))
    parser.add_argument("--output-path", type=Path, default=Path("submission_level2.csv"))
    parser.add_argument("--nace-codes-path", type=Path, default=DEFAULT_NACE_CODES_PATH)
    parser.add_argument("--cleaned-train-path", type=Path, default=DEFAULT_CLEANED_TRAIN_PATH)
    parser.add_argument("--features-per-sku-path", type=Path, default=DEFAULT_FEATURES_PER_SKU_PATH)
    parser.add_argument("--context-cache-path", type=Path, default=DEFAULT_CONTEXT_CACHE_PATH)
    parser.add_argument("--miss-report-dir", type=Path, default=DEFAULT_MISS_REPORT_DIR)
    parser.add_argument("--pair-delimiter", default=DEFAULT_PAIR_DELIMITER)
    parser.add_argument("--submission-buyer-column", default=DEFAULT_SUBMISSION_BUYER_COLUMN)
    parser.add_argument("--submission-cluster-column", default=DEFAULT_SUBMISSION_CLUSTER_COLUMN)
    parser.add_argument("--cutoff-date", default=DEFAULT_CUTOFF_DATE)
    parser.add_argument("--target-fee-eur", type=float, default=DEFAULT_TARGET_FEE_EUR)
    parser.add_argument("--no-miss-report", action="store_true")
    parser.add_argument("--no-source-report", action="store_true")
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    submission, metrics = run_pipeline(
        train_path=args.train_path,
        test_path=args.test_path,
        output_path=args.output_path,
        nace_codes_path=args.nace_codes_path,
        cleaned_train_path=args.cleaned_train_path,
        features_per_sku_path=args.features_per_sku_path,
        context_cache_path=args.context_cache_path,
        miss_report_dir=args.miss_report_dir,
        pair_delimiter=args.pair_delimiter,
        submission_buyer_column=args.submission_buyer_column,
        submission_cluster_column=args.submission_cluster_column,
        cutoff_date=args.cutoff_date,
        target_fee_eur=args.target_fee_eur,
        run_validation=not args.no_validation,
        write_miss_report=not args.no_miss_report,
        write_source_report=not args.no_source_report,
        show_progress=not args.no_progress,
    )

    if metrics is not None:
        print("=== Offline Validation Metrics (proxy) ===")
        for key, value in metrics.items():
            if key in {"Num Hits", "Predictions", "Core Truth Pairs (proxy)"}:
                print(f"{key}: {int(value):,}")
            else:
                print(f"{key}: {value:,.4f}")

    print("=== Final Submission ===")
    print(f"rows: {len(submission)} | buyers: {submission[args.submission_buyer_column].nunique()}")
    print(f"columns: {submission.columns.tolist()}")
    print(f"duplicate rows: {int(submission.duplicated().sum())}")
    buyer_counts = submission.groupby(args.submission_buyer_column).size()
    if buyer_counts.empty:
        print("rows per buyer (min/median/max): 0 / 0.0 / 0")
    else:
        print(
            "rows per buyer (min/median/max): "
            f"{int(buyer_counts.min())} / {float(buyer_counts.median()):.1f} / {int(buyer_counts.max())}"
        )
    print(f"pair delimiter: {args.pair_delimiter!r}")
    print(f"wrote: {args.output_path}")


if __name__ == "__main__":
    main()
