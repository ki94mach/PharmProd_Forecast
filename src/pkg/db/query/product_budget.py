"""Line Budget (human) forecast queries from VW_Product_Budget."""
import pandas as pd

from pkg.db.client import read_sql
from pkg.db.query.constants import GENERIC_EN_IN

# All editions for TARGET_GENERIC_EN Line Budgets (no quarter allowlist).
LINE_BUDGET_FORECASTS = f"""
    SELECT
        p.[ProductTitleEN] AS product,
        p.[GenericEN] AS generic,
        CAST(b.[FK_Date_ID] AS bigint) AS fk_date_id,
        CAST(b.[BudgetQty] AS float) AS forecast,
        CAST(b.[Version] AS bigint) AS version
    FROM [Iris_DW].[Fact].[VW_Product_Budget] b
    INNER JOIN [Iris_DW].[Dim].[Product] p
        ON b.[FK_Product_ID] = p.[ID]
    WHERE b.[Budget_Type] = 'Line Budget'
        AND p.[ProductTitleEN] IS NOT NULL
        AND p.[GenericEN] IN ({GENERIC_EN_IN})
"""


def version_to_qrt(version: int) -> str:
    """Map Version YYYYQQEE to qrt label, e.g. 14040401 -> 1404Q4."""
    v = int(version)
    year = v // 10000
    quarter = (v // 100) % 100
    return f"{year}Q{quarter}"


def version_edition(version: int) -> int:
    """Edition number EE from Version YYYYQQEE."""
    return int(version) % 100


def select_earliest_edition(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the earliest edition per (product, qrt); prefer edition 01 when present.

    Maximizes historical quarter coverage when some qrts lack a first edition.
    """
    if df.empty:
        return df
    out = df.copy()
    out["edition"] = out["version"].map(version_edition)
    # Smallest edition number within each product-qrt
    idx = out.groupby(["product", "qrt"], sort=False)["edition"].idxmin()
    return out.loc[idx].drop(columns=["edition"]).reset_index(drop=True)


def load_line_budget_forecasts(
    earliest_edition_only: bool = True, **engine_kwargs
) -> pd.DataFrame:
    """Load Line Budget forecasts for TARGET_GENERIC_EN products (all historical qrts).

    Version encoding is YYYYQQEE (e.g. 14040401 = 1404Q4 edition 1).
    FK_Date_ID (YYYYMMDD) is converted to Shamsi YYYYMM. BudgetQty is the forecast.

    By default keeps the earliest edition per (product, qrt) so quarters without
    edition 01 are still included. Pass earliest_edition_only=False to keep every
    edition as separate version rows.

    Returns columns: product, generic, date, forecast, version, qrt.
    """
    df = read_sql(LINE_BUDGET_FORECASTS, **engine_kwargs)
    empty_cols = ["product", "generic", "date", "forecast", "version", "qrt"]
    if df.empty:
        return pd.DataFrame(columns=empty_cols)

    df["version"] = df["version"].astype(int)
    df["date"] = (df["fk_date_id"].astype(int) // 100).astype(int)
    df["forecast"] = pd.to_numeric(df["forecast"], errors="coerce")
    df["qrt"] = df["version"].map(version_to_qrt)
    df["product"] = df["product"].astype(str)
    df = df[empty_cols].copy()

    if earliest_edition_only:
        # Apply after expanding date rows: pick earliest edition's version per product-qrt,
        # then keep all date rows for that version.
        edition_map = (
            df.assign(edition=df["version"].map(version_edition))
            .groupby(["product", "qrt"], sort=False)["edition"]
            .min()
            .reset_index()
        )
        df = df.assign(edition=df["version"].map(version_edition)).merge(
            edition_map, on=["product", "qrt", "edition"], how="inner"
        )
        df = df.drop(columns=["edition"])

    return df.reset_index(drop=True)
