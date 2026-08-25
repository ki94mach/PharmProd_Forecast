"""Missing-month vs explicit-zero data-quality audit for TS V2.

Analyzes **source** monthly sales extracts before changing ``missing_month_policy``
or model behavior. Calendar gaps (no warehouse row) are kept distinct from
explicit observed zeros (row present, ``sales == 0``).

Optional inventory cross-tabs are **diagnostic only** — they do not feed the
univariate V2 baseline.

See ``docs/ts_v2_gap_audit.md``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd

from pkg.benchmark.calendar import shamsi_add_months, shamsi_month_start_gregorian
from pkg.ts_v2.config import DEFAULT_CONFIG, TSForecastConfig
from pkg.ts_v2.data import filter_training_frame, product_monthly_sales
from pkg.ts_v2.dates import make_forecast_window, validate_shamsi_yyyymm
from pkg.ts_v2.intermittency import intermittency_stats

MonthState = Literal["missing", "explicit_zero", "positive"]

LIMITATIONS = (
    "Flat_Fact_Sale exposes only aggregated monthly rows grouped by product title "
    "and dimensions. The warehouse does not label whether a missing month is "
    "true zero demand, delayed reporting, SKU inactive, or stockout.\n"
    "An absent row is classified here as **missing** (unknown), not as zero.\n"
    "A present row with SUM(DQTY)=0 is **explicit_zero** (observed zero shipment).\n"
    "Distributor/factory inventory (F3C) is daily stock on hand / in transit; "
    "zero inventory does not prove zero demand and non-zero inventory does not "
    "prove a missing sales row should have been positive. Inventory joins use "
    "ProductTitleEN and month-end snapshot dates — coverage gaps exist.\n"
    "This audit cannot distinguish stockout vs true zero demand vs data latency "
    "without additional operational fields not present in the sales extract."
)


@dataclass(frozen=True)
class ProductGapAuditRow:
    """Per-SKU gap / zero diagnostics on a fixed monthly calendar."""

    product: str
    calendar_start: int
    calendar_end: int
    first_observed_month: Optional[int]
    last_observed_month: Optional[int]
    first_positive_month: Optional[int]
    last_positive_month: Optional[int]
    n_expected_months: int
    n_observed_months: int
    n_missing_months: int
    n_explicit_zero_months: int
    n_positive_months: int
    pct_explicit_zero_observed: Optional[float]
    pct_missing_expected: Optional[float]
    pct_positive_expected: Optional[float]
    longest_explicit_zero_run: int
    longest_missing_run: int
    longest_non_positive_run_if_gaps_filled_as_zero: int
    average_inter_demand_interval: Optional[float]
    n_demand_months: int
    activity_start_applied: bool
    first_active_month: Optional[int]


@dataclass
class GapAuditReport:
    """Portfolio-level gap audit."""

    products: pd.DataFrame
    portfolio: dict[str, Any]
    limitations: str = LIMITATIONS
    inventory_summary: Optional[pd.DataFrame] = None
    month_detail: Optional[pd.DataFrame] = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "portfolio": self.portfolio,
            "limitations": self.limitations,
            "n_products": int(len(self.products)),
        }
        if self.inventory_summary is not None:
            out["inventory_summary"] = self.inventory_summary.to_dict(orient="records")
        return out


def _month_range_inclusive(start_ym: int, end_ym: int) -> list[int]:
    start = validate_shamsi_yyyymm(start_ym)
    end = validate_shamsi_yyyymm(end_ym)
    if start > end:
        return []
    out = [start]
    cur = start
    while cur < end:
        cur = shamsi_add_months(cur, 1)
        out.append(cur)
    return out


def _longest_true_run(mask: pd.Series) -> int:
    if mask.empty or not bool(mask.any()):
        return 0
    grouped = (mask != mask.shift(fill_value=False)).cumsum()
    return int(mask.groupby(grouped).sum().max())


def _inventory_month_end(ym: int) -> pd.Timestamp:
    origin_start = shamsi_month_start_gregorian(int(ym))
    return pd.Timestamp(origin_start - timedelta(days=1))


def _first_active_month(
    observed: pd.Series,
    threshold: Optional[float],
) -> Optional[int]:
    if observed.empty:
        return None
    if threshold is None:
        return int(observed.index.min())
    active = observed.loc[observed > float(threshold)]
    if active.empty:
        return None
    return int(active.index.min())


def classify_month_states(
    observed: pd.Series,
    calendar: Sequence[int],
) -> pd.DataFrame:
    """Label each calendar month as missing, explicit_zero, or positive."""
    idx = [validate_shamsi_yyyymm(int(x)) for x in calendar]
    rows: list[dict[str, Any]] = []
    for ym in idx:
        if ym not in observed.index:
            state: MonthState = "missing"
            sales = np.nan
        else:
            sales = float(observed.loc[ym])
            if sales > 0.0:
                state = "positive"
            else:
                state = "explicit_zero"
        rows.append({"date": ym, "sales": sales, "month_state": state})
    return pd.DataFrame(rows)


def audit_product_gaps(
    observed: pd.Series,
    *,
    product: Optional[str] = None,
    calendar_end: Optional[int] = None,
    activity_start_min_sales: Optional[float] = None,
    apply_activity_start: bool = False,
) -> Optional[ProductGapAuditRow]:
    """Audit one SKU's monthly calendar against aggregated source sales."""
    if observed is None or observed.empty:
        return None

    product_key = str(product or observed.name or "")
    observed = observed.sort_index()
    observed.index = observed.index.map(lambda x: validate_shamsi_yyyymm(int(x)))
    observed = observed.astype(float)

    first_obs = int(observed.index.min())
    last_obs = int(observed.index.max())
    end = validate_shamsi_yyyymm(calendar_end) if calendar_end is not None else last_obs
    if end < first_obs:
        return None

    first_active: Optional[int] = first_obs
    if apply_activity_start:
        first_active = _first_active_month(observed, activity_start_min_sales)
        if first_active is None or first_active > end:
            return None

    calendar = _month_range_inclusive(first_active, end)
    if not calendar:
        return None

    detail = classify_month_states(observed, calendar)
    missing_mask = detail["month_state"] == "missing"
    zero_mask = detail["month_state"] == "explicit_zero"
    pos_mask = detail["month_state"] == "positive"

    n_expected = len(detail)
    n_missing = int(missing_mask.sum())
    n_zero = int(zero_mask.sum())
    n_pos = int(pos_mask.sum())
    n_obs = n_zero + n_pos

    pos_months = detail.loc[pos_mask, "date"].astype(int)
    first_pos = int(pos_months.min()) if not pos_months.empty else None
    last_pos = int(pos_months.max()) if not pos_months.empty else None

    # ADI on calendar positions where demand > 0 (gaps excluded).
    demand_values = detail["sales"].where(pos_mask, other=np.nan)
    stats = intermittency_stats(demand_values)

    # V1-compatible view: gaps filled as zero for longest non-positive run only.
    filled_as_zero = detail["sales"].copy()
    filled_as_zero[missing_mask] = 0.0
    non_positive = filled_as_zero.fillna(0.0) <= 0.0

    return ProductGapAuditRow(
        product=product_key,
        calendar_start=int(calendar[0]),
        calendar_end=int(calendar[-1]),
        first_observed_month=first_obs,
        last_observed_month=last_obs,
        first_positive_month=first_pos,
        last_positive_month=last_pos,
        n_expected_months=n_expected,
        n_observed_months=n_obs,
        n_missing_months=n_missing,
        n_explicit_zero_months=n_zero,
        n_positive_months=n_pos,
        pct_explicit_zero_observed=(n_zero / n_obs) if n_obs else None,
        pct_missing_expected=(n_missing / n_expected) if n_expected else None,
        pct_positive_expected=(n_pos / n_expected) if n_expected else None,
        longest_explicit_zero_run=_longest_true_run(zero_mask),
        longest_missing_run=_longest_true_run(missing_mask),
        longest_non_positive_run_if_gaps_filled_as_zero=_longest_true_run(non_positive),
        average_inter_demand_interval=stats.average_inter_demand_interval,
        n_demand_months=stats.n_demand_months,
        activity_start_applied=bool(apply_activity_start),
        first_active_month=int(first_active) if first_active is not None else None,
    )


def _rows_to_frame(rows: list[ProductGapAuditRow]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[f.name for f in ProductGapAuditRow.__dataclass_fields__.values()]
        )
    return pd.DataFrame([r.__dict__ for r in rows])


def run_gap_audit(
    sales: pd.DataFrame,
    *,
    products: Optional[Sequence[str]] = None,
    origin: Optional[int] = None,
    config: Optional[TSForecastConfig] = None,
    product_col: str = "product",
    date_col: str = "date",
    sales_col: str = "sales",
    apply_activity_start: bool = False,
    include_month_detail: bool = False,
    distributor_inventory: Optional[pd.DataFrame] = None,
    factory_inventory: Optional[pd.DataFrame] = None,
) -> GapAuditReport:
    """Run per-SKU gap audit on a sales extract.

    Parameters
    ----------
    origin:
        When set, keep ``date < origin`` and extend each SKU calendar through
        ``origin - 1`` (matches V2 ``training_end``).
    apply_activity_start:
        When True, trim leading months using ``config.activity_start_min_sales``
        (mirrors ``prepare_monthly_series`` span, not its gap fill).
    distributor_inventory / factory_inventory:
        Optional F3C-style daily inventory frames (``product``, ``snapshot_date``,
        ``distributor_inventory_qty`` / ``factory_inventory_qty``). Used only for
        exploratory cross-tabs; not added to TS models.
    """
    cfg = config or DEFAULT_CONFIG
    work = sales
    calendar_end_by_product: dict[str, int] = {}

    if origin is not None:
        window = make_forecast_window(int(origin), config=cfg)
        work = filter_training_frame(work, window, date_col=date_col, config=cfg)
        training_end = int(window.training_end)
    else:
        training_end = None

    if work is None or work.empty:
        return GapAuditReport(
            products=pd.DataFrame(),
            portfolio={"n_products": 0, "message": "empty sales input"},
        )

    if products is None:
        product_list = sorted(work[product_col].astype(str).unique(), key=str.casefold)
    else:
        product_list = [str(p) for p in products]

    if training_end is not None:
        for p in product_list:
            calendar_end_by_product[p] = training_end
    else:
        for p in product_list:
            sub = work.loc[work[product_col].astype(str) == p, date_col]
            if sub.empty:
                continue
            calendar_end_by_product[p] = int(
                max(validate_shamsi_yyyymm(int(x)) for x in sub)
            )

    rows: list[ProductGapAuditRow] = []
    detail_frames: list[pd.DataFrame] = []

    for product in product_list:
        observed = product_monthly_sales(
            work,
            product,
            product_col=product_col,
            date_col=date_col,
            sales_col=sales_col,
        )
        observed.name = product
        end = calendar_end_by_product.get(product)
        row = audit_product_gaps(
            observed,
            product=product,
            calendar_end=end,
            activity_start_min_sales=cfg.activity_start_min_sales,
            apply_activity_start=apply_activity_start,
        )
        if row is None:
            continue
        rows.append(row)
        if include_month_detail:
            cal = _month_range_inclusive(row.calendar_start, row.calendar_end)
            detail = classify_month_states(observed, cal)
            detail.insert(0, "product", product)
            detail_frames.append(detail)

    products_df = _rows_to_frame(rows)
    portfolio = summarize_gap_audit(products_df)

    month_detail = (
        pd.concat(detail_frames, ignore_index=True) if detail_frames else None
    )
    inv_summary = None
    if month_detail is not None and (
        distributor_inventory is not None or factory_inventory is not None
    ):
        inv_summary = audit_inventory_relationship(
            month_detail,
            distributor_inventory=distributor_inventory,
            factory_inventory=factory_inventory,
        )

    return GapAuditReport(
        products=products_df,
        portfolio=portfolio,
        inventory_summary=inv_summary,
        month_detail=month_detail,
    )


def summarize_gap_audit(products_df: pd.DataFrame) -> dict[str, Any]:
    """Portfolio aggregates from per-SKU audit rows."""
    if products_df is None or products_df.empty:
        return {"n_products": 0}

    def _q(col: str, q: float) -> float:
        s = pd.to_numeric(products_df[col], errors="coerce").dropna()
        return float(s.quantile(q)) if len(s) else float("nan")

    total_expected = int(products_df["n_expected_months"].sum())
    total_missing = int(products_df["n_missing_months"].sum())
    total_zero = int(products_df["n_explicit_zero_months"].sum())
    total_pos = int(products_df["n_positive_months"].sum())

    only_missing_zeros = products_df.loc[
        (products_df["n_explicit_zero_months"] == 0)
        & (products_df["n_missing_months"] > 0)
    ]

    return {
        "n_products": int(len(products_df)),
        "total_expected_months": total_expected,
        "total_missing_months": total_missing,
        "total_explicit_zero_months": total_zero,
        "total_positive_months": total_pos,
        "portfolio_pct_missing": total_missing / total_expected if total_expected else None,
        "portfolio_pct_explicit_zero": total_zero / total_expected if total_expected else None,
        "portfolio_pct_positive": total_pos / total_expected if total_expected else None,
        "n_skus_with_any_missing": int((products_df["n_missing_months"] > 0).sum()),
        "n_skus_with_any_explicit_zero": int((products_df["n_explicit_zero_months"] > 0).sum()),
        "n_skus_zero_explicit_zeros_only_gaps": int(len(only_missing_zeros)),
        "median_pct_missing_expected": _q("pct_missing_expected", 0.5),
        "p90_pct_missing_expected": _q("pct_missing_expected", 0.9),
        "median_longest_missing_run": _q("longest_missing_run", 0.5),
        "max_longest_missing_run": float(products_df["longest_missing_run"].max()),
        "median_longest_explicit_zero_run": _q("longest_explicit_zero_run", 0.5),
        "max_longest_explicit_zero_run": float(
            products_df["longest_explicit_zero_run"].max()
        ),
    }


def audit_inventory_relationship(
    month_detail: pd.DataFrame,
    *,
    distributor_inventory: Optional[pd.DataFrame] = None,
    factory_inventory: Optional[pd.DataFrame] = None,
    product_col: str = "product",
    date_col: str = "date",
    state_col: str = "month_state",
) -> pd.DataFrame:
    """Exploratory cross-tab of sales month state vs month-end inventory.

    Returns one row per ``month_state`` with inventory lookup coverage and
    conditional shares (e.g. share with distributor qty == 0). Does **not**
    establish causal stockout labels.
    """
    if month_detail is None or month_detail.empty:
        return pd.DataFrame()

    work = month_detail.copy()
    work[product_col] = work[product_col].astype(str)
    work[date_col] = work[date_col].map(lambda x: validate_shamsi_yyyymm(int(x)))
    work["inventory_month_end"] = work[date_col].map(_inventory_month_end)

    if distributor_inventory is not None and not distributor_inventory.empty:
        dist = distributor_inventory.copy()
        dist["product"] = dist["product"].astype(str)
        dist["snapshot_date"] = pd.to_datetime(dist["snapshot_date"]).dt.normalize()
        dist = dist.rename(
            columns={"distributor_inventory_qty": "dist_qty"}
        )
        work = work.merge(
            dist[["product", "snapshot_date", "dist_qty"]],
            left_on=[product_col, "inventory_month_end"],
            right_on=["product", "snapshot_date"],
            how="left",
        )
    else:
        work["dist_qty"] = np.nan

    if factory_inventory is not None and not factory_inventory.empty:
        fact = factory_inventory.copy()
        fact["product"] = fact["product"].astype(str)
        fact["snapshot_date"] = pd.to_datetime(fact["snapshot_date"]).dt.normalize()
        fact = fact.rename(columns={"factory_inventory_qty": "fact_qty"})
        work = work.merge(
            fact[["product", "snapshot_date", "fact_qty"]],
            left_on=[product_col, "inventory_month_end"],
            right_on=["product", "snapshot_date"],
            how="left",
            suffixes=("", "_fact"),
        )
    else:
        work["fact_qty"] = np.nan

    rows: list[dict[str, Any]] = []
    for state, grp in work.groupby(state_col, sort=False):
        n = int(len(grp))
        dist_present = grp["dist_qty"].notna()
        fact_present = grp["fact_qty"].notna()
        row = {
            "month_state": state,
            "n_product_months": n,
            "n_with_distributor_record": int(dist_present.sum()),
            "n_with_factory_record": int(fact_present.sum()),
            "pct_distributor_record": float(dist_present.mean()) if n else None,
            "pct_factory_record": float(fact_present.mean()) if n else None,
        }
        if dist_present.any():
            dist_vals = grp.loc[dist_present, "dist_qty"].astype(float)
            row["pct_distributor_qty_eq0"] = float((dist_vals == 0).mean())
            row["median_distributor_qty"] = float(dist_vals.median())
        else:
            row["pct_distributor_qty_eq0"] = None
            row["median_distributor_qty"] = None
        if fact_present.any():
            fact_vals = grp.loc[fact_present, "fact_qty"].astype(float)
            row["pct_factory_qty_eq0"] = float((fact_vals == 0).mean())
            row["median_factory_qty"] = float(fact_vals.median())
        else:
            row["pct_factory_qty_eq0"] = None
            row["median_factory_qty"] = None
        rows.append(row)

    out = pd.DataFrame(rows)
    out["note"] = (
        "Exploratory only: inventory presence/qty does not prove demand state "
        "for missing or zero sales months."
    )
    return out


def write_gap_audit_report(
    report: GapAuditReport,
    out_dir: Union[str, Path],
    *,
    prefix: str = "gap_audit",
) -> dict[str, Path]:
    """Write CSV artifacts and a markdown summary."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    products_path = root / f"{prefix}_by_product.csv"
    report.products.to_csv(products_path, index=False)
    paths["products"] = products_path

    if report.month_detail is not None and not report.month_detail.empty:
        detail_path = root / f"{prefix}_month_detail.csv"
        report.month_detail.to_csv(detail_path, index=False)
        paths["month_detail"] = detail_path

    if report.inventory_summary is not None and not report.inventory_summary.empty:
        inv_path = root / f"{prefix}_inventory_cross_tab.csv"
        report.inventory_summary.to_csv(inv_path, index=False)
        paths["inventory"] = inv_path

    summary_path = root / f"{prefix}_summary.md"
    lines = [
        "# TS V2 gap audit summary",
        "",
        "## Portfolio",
        "",
    ]
    for key, val in report.portfolio.items():
        lines.append(f"- **{key}**: {val}")
    lines.extend(["", "## Limitations", "", report.limitations, ""])
    if report.inventory_summary is not None and not report.inventory_summary.empty:
        lines.extend(["", "## Inventory cross-tab (exploratory)", ""])
        lines.append(report.inventory_summary.to_markdown(index=False))
        lines.append("")
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    paths["summary"] = summary_path
    return paths


def audit_from_sales_loader(
    loader,
    *,
    origin: Optional[int] = None,
    **audit_kwargs: Any,
) -> GapAuditReport:
    """Convenience wrapper: ``loader(**engine_kwargs)`` → :func:`run_gap_audit`."""
    sales = loader()
    return run_gap_audit(sales, origin=origin, **audit_kwargs)
