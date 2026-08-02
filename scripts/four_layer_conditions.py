#!/usr/bin/env python3
"""Point-in-time four-layer conditions for conditional trajectory diffusion.

The output is always:
  values:       48 floats in [-1, 1]
  observed_mask: 48 integers (1=observed, 0=missing)

Missing values are serialized as 0 only for tensor compatibility; the mask must
be passed to the model so that "missing" is not confused with "neutral".

LLM-produced events are inputs to this module, not final factors.  This module
performs timestamp filtering, decay, aggregation and validation deterministically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


COMPANY_FEATURES = [
    "revenue_surprise", "profit_surprise", "eps_surprise", "revenue_growth",
    "profit_growth", "margin_change", "roe_change", "cash_quality",
    "eps_revision", "fund_flow", "margin_financing_change", "dilution_pressure",
    "event_earnings", "event_guidance", "event_order_contract",
    "event_product_technology", "event_management", "event_regulation_litigation",
    "event_capital_action", "event_supply_chain",
]
SECTOR_FEATURES = [
    "sector_return_1d", "sector_return_5d", "sector_relative_20d",
    "sector_breadth", "sector_fund_flow", "sector_volume",
    "peer_earnings_breadth", "supply_demand", "pricing_power",
    "inventory_cycle", "policy_support", "competition_intensity",
]
MACRO_FEATURES = [
    "market_return", "market_volatility", "market_breadth", "limit_sentiment",
    "market_fund_flow", "option_iv", "rate_regime", "fx_pressure",
    "growth_surprise", "policy_risk",
]
META_FEATURES = [
    "source_quality", "official_ratio", "information_coverage",
    "novelty", "conflict", "freshness",
]
FEATURE_NAMES = COMPANY_FEATURES + SECTOR_FEATURES + MACRO_FEATURES + META_FEATURES
assert len(FEATURE_NAMES) == 48

EVENT_TO_FEATURE = {
    "earnings": "event_earnings",
    "earnings_guidance": "event_guidance",
    "guidance": "event_guidance",
    "order": "event_order_contract",
    "contract": "event_order_contract",
    "order_contract": "event_order_contract",
    "product": "event_product_technology",
    "technology": "event_product_technology",
    "product_technology": "event_product_technology",
    "management": "event_management",
    "regulation": "event_regulation_litigation",
    "litigation": "event_regulation_litigation",
    "regulation_litigation": "event_regulation_litigation",
    "merger": "event_capital_action",
    "buyback": "event_capital_action",
    "placement": "event_capital_action",
    "capital_action": "event_capital_action",
    "supply_chain": "event_supply_chain",
    "supply_demand": "supply_demand",
    "pricing_power": "pricing_power",
    "inventory_cycle": "inventory_cycle",
    "sector_policy": "policy_support",
    "competition": "competition_intensity",
    "policy_risk": "policy_risk",
}
OFFICIAL_TYPES = {"exchange_announcement", "official_policy", "company_ir"}


def _parse_time(value: str | datetime) -> datetime:
    result = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError(f"time must include timezone: {value!r}")
    return result


def _bounded(value: float) -> float:
    return float(np.clip(value, -1.0, 1.0))


def robust_score(value: float, history: Sequence[float]) -> float | None:
    """Past-only robust normalization, compressed to [-1, 1]."""
    clean = np.asarray([x for x in history if x is not None and np.isfinite(x)], dtype=float)
    if not np.isfinite(value) or len(clean) < 20:
        return None
    median = float(np.median(clean))
    mad = float(np.median(np.abs(clean - median)))
    scale = max(1.4826 * mad, 1e-12)
    return _bounded((value - median) / scale / 3.0)


def bounded_surprise(actual: float | None, expected: float | None,
                     scale_floor: float = 0.05) -> float | None:
    if actual is None or expected is None:
        return None
    denominator = max(abs(expected), scale_floor)
    return float(math.tanh((actual - expected) / denominator))


@dataclass(frozen=True)
class Event:
    event_type: str
    layer: str
    direction: float
    magnitude: float
    exposure: float
    horizon_days: int
    novelty: float
    source_quality: float
    evidence_confidence: float
    published_at: datetime
    source_type: str
    source_ids: tuple[str, ...]
    evidence_text: str | None = None
    actual: float | None = None
    expected: float | None = None

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "Event":
        event = cls(
            event_type=str(row["event_type"]),
            layer=str(row.get("layer", "company")),
            direction=float(row["direction"]),
            magnitude=float(row["magnitude"]),
            exposure=float(row.get("exposure", 1.0)),
            horizon_days=int(row["horizon_days"]),
            novelty=float(row["novelty"]),
            source_quality=float(row["source_quality"]),
            evidence_confidence=float(row["evidence_confidence"]),
            published_at=_parse_time(row["published_at"]),
            source_type=str(row["source_type"]),
            source_ids=tuple(str(x) for x in row.get("source_ids", [])),
            evidence_text=row.get("evidence_text"),
            actual=row.get("actual"),
            expected=row.get("expected"),
        )
        event.validate()
        return event

    def validate(self) -> None:
        if self.layer not in {"company", "sector", "macro"}:
            raise ValueError(f"invalid layer: {self.layer}")
        if self.event_type not in EVENT_TO_FEATURE:
            raise ValueError(f"unsupported event_type: {self.event_type}")
        if not -1 <= self.direction <= 1:
            raise ValueError("direction must be in [-1, 1]")
        for name in ("magnitude", "exposure", "novelty", "source_quality",
                     "evidence_confidence"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 1 <= self.horizon_days <= 730:
            raise ValueError("horizon_days must be in [1, 730]")
        if not self.source_ids:
            raise ValueError("at least one source_id is required")

    @property
    def identity(self) -> str:
        payload = "|".join((self.event_type, self.layer, *sorted(self.source_ids)))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class ConditionVector:
    ticker: str
    decision_time: str
    feature_names: list[str]
    values: list[float]
    observed_mask: list[int]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def filter_point_in_time(events: Iterable[Event], decision_time: datetime) -> list[Event]:
    """Reject future, undated and duplicated evidence before aggregation."""
    decision_time = _parse_time(decision_time)
    unique: dict[str, Event] = {}
    for event in events:
        if event.published_at > decision_time:
            continue
        previous = unique.get(event.identity)
        if previous is None or event.evidence_confidence > previous.evidence_confidence:
            unique[event.identity] = event
    return list(unique.values())


def aggregate_events(events: Sequence[Event], decision_time: datetime) -> dict[str, float]:
    totals: dict[str, float] = {}
    for event in events:
        age_days = max((decision_time - event.published_at).total_seconds() / 86400.0, 0.0)
        decay = math.exp(-math.log(2) * age_days / event.horizon_days)
        score = (event.direction * event.magnitude * event.exposure *
                 event.novelty * event.source_quality *
                 event.evidence_confidence * decay)
        feature = EVENT_TO_FEATURE[event.event_type]
        totals[feature] = totals.get(feature, 0.0) + score
    return {name: float(math.tanh(score)) for name, score in totals.items()}


def meta_features(events: Sequence[Event], expected_slots: int = 13) -> dict[str, float]:
    if not events:
        return {}
    qualities = np.asarray([e.source_quality for e in events])
    novelties = np.asarray([e.novelty for e in events])
    official = np.asarray([e.source_type in OFFICIAL_TYPES for e in events])
    ages = np.asarray([max((max(x.published_at for x in events) - e.published_at)
                           .total_seconds() / 86400, 0) for e in events])

    # Direction disagreement within the same semantic feature.
    groups: dict[str, list[float]] = {}
    for event in events:
        groups.setdefault(EVENT_TO_FEATURE[event.event_type], []).append(event.direction)
    disagreements = [np.std(v) for v in groups.values() if len(v) > 1]
    conflict = min(float(np.mean(disagreements)) if disagreements else 0.0, 1.0)
    coverage = min(len(groups) / expected_slots, 1.0)
    freshness = float(np.mean(np.exp(-math.log(2) * ages / 30.0)))
    return {
        "source_quality": float(np.mean(qualities)),
        "official_ratio": float(np.mean(official)),
        "information_coverage": coverage,
        "novelty": float(np.mean(novelties)),
        # Higher means more conflict; the condition encoder can learn to gate down.
        "conflict": conflict,
        "freshness": freshness,
    }


class LocalDataProvider:
    """Read existing per-day caches without looking beyond decision_date."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def _days(self, table: str, through: str) -> list[Path]:
        return [p for p in sorted((self.data_dir / table).glob("*.pkl"))
                if p.stem <= through]

    @staticmethod
    def _row(path: Path, ticker: str) -> pd.Series | None:
        frame = pd.read_pickle(path)
        rows = frame.loc[frame["stock_id"].eq(ticker)]
        return None if rows.empty else rows.iloc[-1]

    def company_hard(self, ticker: str, date: str) -> dict[str, float]:
        result: dict[str, float] = {}
        flow_paths = self._days("stock_money_flow", date)[-21:]
        flow = [self._row(p, ticker) for p in flow_paths]
        flow = [x for x in flow if x is not None]
        if flow:
            current = float(flow[-1]["net_inflow_rate_value"])
            history = [float(x["net_inflow_rate_value"]) for x in flow[:-1]
                       if pd.notna(x["net_inflow_rate_value"])]
            score = robust_score(current, history)
            if score is not None:
                result["fund_flow"] = score

        equity_paths = self._days("stock_equity", date)[-22:]
        equity = [self._row(p, ticker) for p in equity_paths]
        equity = [x for x in equity if x is not None and pd.notna(x["float_a_share"])]
        if len(equity) >= 2:
            old, new = float(equity[0]["float_a_share"]), float(equity[-1]["float_a_share"])
            if old > 0:
                # Share expansion is dilution pressure, hence negative.
                result["dilution_pressure"] = _bounded(-math.tanh((new / old - 1) * 20))
        return result

    def market_sector_hard(self, ticker: str, date: str) -> tuple[dict[str, float], dict[str, float]]:
        market_paths = self._days("stock_market", date)[-21:]
        status_paths = {p.stem: p for p in self._days("stock_status", date)[-21:]}
        rows: list[tuple[pd.DataFrame, pd.Series, str]] = []
        for path in market_paths:
            frame = pd.read_pickle(path)
            stock = frame.loc[frame.stock_id.eq(ticker)]
            if not stock.empty:
                rows.append((frame, stock.iloc[-1], path.stem))
        if not rows:
            return {}, {}

        macro: dict[str, float] = {}
        latest_frame = rows[-1][0]
        index = latest_frame.loc[latest_frame.stock_id.eq("000300.SH")]
        if not index.empty and pd.notna(index.iloc[-1]["pct_chg"]):
            macro["market_return"] = _bounded(float(index.iloc[-1]["pct_chg"]) / 3.0)
        all_a = latest_frame.loc[latest_frame.stock_id.str.endswith((".SH", ".SZ"), na=False)]
        changes = pd.to_numeric(all_a["pct_chg"], errors="coerce").dropna()
        if len(changes):
            macro["market_breadth"] = _bounded((changes.gt(0).mean() - changes.lt(0).mean()))
            macro["market_volatility"] = _bounded(changes.std() / 5.0)
            up = changes.ge(9.5).mean()
            down = changes.le(-9.5).mean()
            macro["limit_sentiment"] = _bounded((up - down) * 20)

        sector: dict[str, float] = {}
        sector_returns: list[float] = []
        index_returns: list[float] = []
        latest_sector_changes: pd.Series | None = None
        for frame, _, row_date in rows:
            status_path = status_paths.get(row_date)
            if status_path is None:
                continue
            status = pd.read_pickle(status_path)
            target = status.loc[status.stock_id.eq(ticker)]
            industry_cols = [c for c in status.columns if c.startswith("zx_")]
            active = [c for c in industry_cols
                      if not target.empty and target.iloc[-1].get(c) == 1]
            if not active:
                continue
            members = status.loc[status[active[0]].eq(1), "stock_id"]
            changes = pd.to_numeric(
                frame.loc[frame.stock_id.isin(members), "pct_chg"], errors="coerce"
            ).dropna()
            if changes.empty:
                continue
            sector_returns.append(float(changes.mean()) / 100)
            latest_sector_changes = changes
            idx = frame.loc[frame.stock_id.eq("000300.SH"), "pct_chg"]
            if not idx.empty and pd.notna(idx.iloc[-1]):
                index_returns.append(float(idx.iloc[-1]) / 100)

        if latest_sector_changes is not None:
            sector["sector_return_1d"] = _bounded(float(latest_sector_changes.mean()) / 3.0)
            sector["sector_breadth"] = _bounded(
                latest_sector_changes.gt(0).mean() - latest_sector_changes.lt(0).mean())
        if len(sector_returns) >= 5:
            sector["sector_return_5d"] = _bounded(
                (np.prod(1 + np.asarray(sector_returns[-5:])) - 1) / 0.10)
        if len(sector_returns) >= 20 and len(index_returns) >= 20:
            sector_20 = np.prod(1 + np.asarray(sector_returns[-20:])) - 1
            market_20 = np.prod(1 + np.asarray(index_returns[-20:])) - 1
            sector["sector_relative_20d"] = _bounded((sector_20 - market_20) / 0.20)
        return sector, macro


def build_condition_vector(
    ticker: str,
    decision_time: str | datetime,
    data_dir: Path,
    event_rows: Sequence[Mapping[str, Any]] = (),
    supplied_features: Mapping[str, float | None] | None = None,
) -> ConditionVector:
    decision = _parse_time(decision_time)
    date = decision.date().isoformat()
    provider = LocalDataProvider(data_dir)
    company = provider.company_hard(ticker, date)
    sector, macro = provider.market_sector_hard(ticker, date)

    parsed = [Event.from_mapping(x) for x in event_rows]
    accepted = filter_point_in_time(parsed, decision)
    semantic = aggregate_events(accepted, decision)
    meta = meta_features(accepted)
    features: dict[str, float] = {**company, **sector, **macro, **semantic, **meta}
    if supplied_features:
        unknown = set(supplied_features) - set(FEATURE_NAMES)
        if unknown:
            raise ValueError(f"unknown supplied features: {sorted(unknown)}")
        features.update({k: float(v) for k, v in supplied_features.items() if v is not None})

    values = [_bounded(features.get(name, 0.0)) for name in FEATURE_NAMES]
    mask = [int(name in features) for name in FEATURE_NAMES]
    return ConditionVector(
        ticker=ticker,
        decision_time=decision.isoformat(),
        feature_names=list(FEATURE_NAMES),
        values=values,
        observed_mask=mask,
        provenance={
            "accepted_event_ids": [e.identity for e in accepted],
            "rejected_event_count": len(parsed) - len(accepted),
            "observed_count": sum(mask),
            "data_cutoff": date,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True, help="e.g. 000001.SZ")
    parser.add_argument("--decision-time", required=True,
                        help="timezone-aware ISO timestamp, e.g. 2026-07-10T15:00:00+08:00")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--events", type=Path, help="JSON file: a list, or {events: [...]}")
    parser.add_argument("--features", type=Path, help="optional JSON hard/macro feature overrides")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    event_rows: Sequence[Mapping[str, Any]] = []
    if args.events:
        payload = json.loads(args.events.read_text())
        event_rows = payload["events"] if isinstance(payload, dict) else payload
    supplied = json.loads(args.features.read_text()) if args.features else None
    result = build_condition_vector(
        args.ticker, args.decision_time, args.data_dir, event_rows, supplied)
    text = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
