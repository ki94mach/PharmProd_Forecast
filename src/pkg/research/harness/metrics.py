"""Pair metrics vs a control on the frozen PRIMARY evaluation keys."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pkg.benchmark.config import HORIZON_BUCKETS
from pkg.benchmark.dataset import horizon_bucket
from pkg.benchmark.evaluate import BacktestResult, wmape

ROW_KEYS = ("product", "qrt", "target_date", "test_origin")

CONCENTRATION_ONE_PRODUCT = 0.25
CONCENTRATION_TOP5 = 0.50


def row_key_frame(preds: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in ROW_KEYS if c not in preds.columns]
    if missing:
        raise KeyError(f"predictions missing keys {missing}")
    return (
        preds[list(ROW_KEYS)]
        .assign(
            product=lambda d: d["product"].astype(str),
            qrt=lambda d: d["qrt"].astype(str),
            target_date=lambda d: d["target_date"].astype(int),
            test_origin=lambda d: d["test_origin"].astype(int),
        )
        .sort_values(list(ROW_KEYS))
        .reset_index(drop=True)
    )


def assert_same_eval_rows(baseline: BacktestResult, candidate: BacktestResult) -> None:
    """Candidate experiment must score exactly the same TEST identities as F0."""
    a = row_key_frame(baseline.predictions)
    b = row_key_frame(candidate.predictions)
    if len(a) != len(b):
        raise AssertionError(
            f"eval row count mismatch: F0 n={len(a)} candidate n={len(b)}"
        )
    if not a.equals(b):
        merged = a.merge(b, on=list(ROW_KEYS), how="outer", indicator=True)
        bad = merged.loc[merged["_merge"] != "both"]
        raise AssertionError(
            f"eval row keys differ from F0 (diff_rows={len(bad)}). "
            f"sample:\n{bad.head(10)}"
        )


def rel_wmape(base: float, new: float) -> float:
    if base == 0 or not np.isfinite(base):
        return float("nan")
    return float((base - new) / base * 100.0)


def origins_improved(base: BacktestResult, cand: BacktestResult) -> tuple[int, int]:
    b = base.by_origin.set_index("origin")["wmape"]
    c = cand.by_origin.set_index("origin")["wmape"]
    common = sorted(set(b.index) & set(c.index))
    improved = sum(1 for o in common if c.loc[o] < b.loc[o])
    return improved, len(common)


def product_stats(base: BacktestResult, cand: BacktestResult) -> dict:
    """Product win rate / median improvement (n>=3), F1-style."""
    b = base.predictions[
        ["product", "qrt", "target_date", "test_origin", "actual", "prediction"]
    ].rename(columns={"prediction": "pred_f0"})
    c = cand.predictions[
        ["product", "qrt", "target_date", "test_origin", "prediction"]
    ].rename(columns={"prediction": "pred_new"})
    m = b.merge(c, on=list(ROW_KEYS), how="inner")
    rows = []
    for product, g in m.groupby("product"):
        if len(g) < 3:
            continue
        w0 = wmape(g["actual"], g["pred_f0"])
        w1 = wmape(g["actual"], g["pred_new"])
        rows.append(
            {
                "product": product,
                "n": len(g),
                "wmape_f0": w0,
                "wmape_new": w1,
                "rel_improvement_pct": rel_wmape(w0, w1),
            }
        )
    if not rows:
        return {
            "product_win_rate": float("nan"),
            "median_product_improvement_pct": float("nan"),
            "n_products": 0,
        }
    pdf = pd.DataFrame(rows)
    return {
        "product_win_rate": float((pdf["rel_improvement_pct"] > 0).mean()),
        "median_product_improvement_pct": float(pdf["rel_improvement_pct"].median()),
        "n_products": int(len(pdf)),
    }


def product_stats_full(base: BacktestResult, cand: BacktestResult) -> dict:
    """F2-style product stats (n>=3) plus quartiles and a table."""
    b = base.predictions[
        ["product", "qrt", "target_date", "test_origin", "actual", "prediction"]
    ].rename(columns={"prediction": "pred_f0"})
    c = cand.predictions[
        ["product", "qrt", "target_date", "test_origin", "prediction"]
    ].rename(columns={"prediction": "pred_new"})
    m = b.merge(c, on=list(ROW_KEYS), how="inner")
    rows = []
    for product, g in m.groupby("product"):
        if len(g) < 3:
            continue
        w0 = wmape(g["actual"], g["pred_f0"])
        w1 = wmape(g["actual"], g["pred_new"])
        rows.append(
            {
                "product": product,
                "n": len(g),
                "wmape_f0": w0,
                "wmape_new": w1,
                "rel_improvement_pct": rel_wmape(w0, w1),
            }
        )
    if not rows:
        return {
            "product_win_rate": float("nan"),
            "median_product_improvement_pct": float("nan"),
            "p25_product_improvement_pct": float("nan"),
            "p75_product_improvement_pct": float("nan"),
            "n_products": 0,
            "table": pd.DataFrame(),
        }
    pdf = pd.DataFrame(rows)
    return {
        "product_win_rate": float((pdf["rel_improvement_pct"] > 0).mean()),
        "median_product_improvement_pct": float(pdf["rel_improvement_pct"].median()),
        "p25_product_improvement_pct": float(pdf["rel_improvement_pct"].quantile(0.25)),
        "p75_product_improvement_pct": float(pdf["rel_improvement_pct"].quantile(0.75)),
        "n_products": int(len(pdf)),
        "table": pdf,
    }


def product_pair_table(control: BacktestResult, cand: BacktestResult) -> pd.DataFrame:
    """All-product table with control/candidate names (F3A-style, no n>=3 filter)."""
    b = control.predictions[
        ["product", "qrt", "target_date", "test_origin", "actual", "prediction"]
    ].rename(columns={"prediction": "pred_control"})
    c = cand.predictions[
        ["product", "qrt", "target_date", "test_origin", "prediction"]
    ].rename(columns={"prediction": "pred_candidate"})
    m = b.merge(c, on=list(ROW_KEYS), how="inner")
    rows = []
    for product, g in m.groupby("product"):
        w0 = wmape(g["actual"], g["pred_control"])
        w1 = wmape(g["actual"], g["pred_candidate"])
        ae0 = (g["actual"] - g["pred_control"]).abs().sum()
        ae1 = (g["actual"] - g["pred_candidate"]).abs().sum()
        rows.append(
            {
                "product": product,
                "actual_volume": float(np.abs(g["actual"]).sum()),
                "n": int(len(g)),
                "wmape_control": w0,
                "wmape_candidate": w1,
                "relative_improvement_pct": rel_wmape(w0, w1),
                "delta_absolute_error": float(ae1 - ae0),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("actual_volume", ascending=False)
        .reset_index(drop=True)
    )


def product_summary(pdf: pd.DataFrame) -> dict:
    if pdf is None or pdf.empty:
        return {
            "product_win_rate": float("nan"),
            "median_product_improvement_pct": float("nan"),
            "p25_product_improvement_pct": float("nan"),
            "p75_product_improvement_pct": float("nan"),
            "n_products": 0,
        }
    rel = pdf["relative_improvement_pct"].to_numpy(dtype=float)
    finite = rel[np.isfinite(rel)]
    if len(finite) == 0:
        win = med = p25 = p75 = float("nan")
    else:
        win = float((finite > 0).mean())
        med = float(np.median(finite))
        p25 = float(np.quantile(finite, 0.25))
        p75 = float(np.quantile(finite, 0.75))
    return {
        "product_win_rate": win,
        "median_product_improvement_pct": med,
        "p25_product_improvement_pct": p25,
        "p75_product_improvement_pct": p75,
        "n_products": int(len(pdf)),
    }


def origin_pair_table(control: BacktestResult, cand: BacktestResult) -> pd.DataFrame:
    b = control.by_origin.set_index("origin")
    c = cand.by_origin.set_index("origin")
    common = sorted(set(b.index) & set(c.index))
    rows = []
    for o in common:
        w0 = float(b.loc[o, "wmape"])
        w1 = float(c.loc[o, "wmape"])
        rows.append(
            {
                "origin": int(o),
                "n": int(c.loc[o, "n"]),
                "wmape_control": w0,
                "wmape_candidate": w1,
                "relative_improvement_pct": rel_wmape(w0, w1),
            }
        )
    return pd.DataFrame(rows)


def origin_summary(odf: pd.DataFrame) -> dict:
    if odf is None or odf.empty:
        return {
            "origins_improved": 0,
            "origins_total": 0,
            "median_origin_improvement": float("nan"),
        }
    rel = odf["relative_improvement_pct"].to_numpy(dtype=float)
    return {
        "origins_improved": int((rel > 0).sum()),
        "origins_total": int(len(odf)),
        "median_origin_improvement": float(np.median(rel)) if len(rel) else float("nan"),
    }


def horizon_bucket_table(base: BacktestResult, cand: BacktestResult) -> pd.DataFrame:
    b = base.predictions[
        ["product", "qrt", "target_date", "test_origin", "horizon", "actual", "prediction"]
    ].rename(columns={"prediction": "pred_f0"})
    c = cand.predictions[
        ["product", "qrt", "target_date", "test_origin", "prediction"]
    ].rename(columns={"prediction": "pred_new"})
    m = b.merge(c, on=list(ROW_KEYS), how="inner")
    m["horizon_bucket"] = m["horizon"].map(horizon_bucket)
    rows = []
    for name, lo, hi in HORIZON_BUCKETS:
        g = m.loc[m["horizon_bucket"] == name]
        if g.empty:
            continue
        w0 = wmape(g["actual"], g["pred_f0"])
        w1 = wmape(g["actual"], g["pred_new"])
        rows.append(
            {
                "horizon_bucket": name,
                "n": len(g),
                "wmape_f0": w0,
                "wmape_new": w1,
                "rel_wmape_vs_f0_pct": rel_wmape(w0, w1),
            }
        )
    return pd.DataFrame(rows)


def merge_ae(f0: BacktestResult, cand: BacktestResult) -> pd.DataFrame:
    b = f0.predictions[
        ["product", "qrt", "target_date", "test_origin", "horizon", "actual", "prediction"]
    ].rename(columns={"prediction": "pred_f0"})
    c = cand.predictions[
        ["product", "qrt", "target_date", "test_origin", "prediction"]
    ].rename(columns={"prediction": "pred_cand"})
    m = b.merge(c, on=list(ROW_KEYS), how="inner")
    m["ae_f0"] = (m["actual"] - m["pred_f0"]).abs()
    m["ae_cand"] = (m["actual"] - m["pred_cand"]).abs()
    m["delta_ae"] = m["ae_cand"] - m["ae_f0"]
    m["horizon_bucket"] = m["horizon"].map(horizon_bucket)
    return m


def error_concentration(m: pd.DataFrame, experiment: str, anchor: str) -> dict:
    net = float(m["delta_ae"].sum())
    det = m.loc[m["delta_ae"] > 0, "delta_ae"].sum()
    imp = -m.loc[m["delta_ae"] < 0, "delta_ae"].sum()
    by_p = (
        m.groupby("product")
        .agg(
            delta_ae=("delta_ae", "sum"),
            actual_volume=("actual", lambda s: float(np.abs(s).sum())),
            n=("delta_ae", "count"),
        )
        .reset_index()
        .sort_values("delta_ae", ascending=False)
    )
    det_sorted = by_p.loc[by_p["delta_ae"] > 0].sort_values("delta_ae", ascending=False)
    imp_sorted = by_p.loc[by_p["delta_ae"] < 0].sort_values("delta_ae")
    top5_det_share = (
        float(det_sorted.head(5)["delta_ae"].sum() / det) if det > 0 else 0.0
    )
    top10_det_share = (
        float(det_sorted.head(10)["delta_ae"].sum() / det) if det > 0 else 0.0
    )
    top5_imp_share = (
        float((-imp_sorted.head(5)["delta_ae"]).sum() / imp) if imp > 0 else 0.0
    )
    top1_share_of_net_det = 0.0
    if det > 0 and len(det_sorted):
        top1_share_of_net_det = float(det_sorted.iloc[0]["delta_ae"] / det)

    flags = []
    if top1_share_of_net_det > CONCENTRATION_ONE_PRODUCT:
        flags.append("one_product_gt_25pct_deterioration")
    if top5_det_share > CONCENTRATION_TOP5:
        flags.append("top5_gt_50pct_deterioration")

    return {
        "net_delta_ae": net,
        "total_deterioration": float(det),
        "total_improvement": float(imp),
        "top5_deterioration_share": top5_det_share,
        "top10_deterioration_share": top10_det_share,
        "top5_improvement_share": top5_imp_share,
        "top1_deterioration_share": top1_share_of_net_det,
        "flags": flags,
        "by_product": by_p,
        "top5_det": det_sorted.head(5),
        "top10_det": det_sorted.head(10),
        "top5_imp": imp_sorted.head(5),
    }
