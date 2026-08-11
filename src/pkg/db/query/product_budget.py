"""Line Budget (human) forecast queries from VW_Product_Budget."""
import pandas as pd

from pkg.db.client import read_sql
from pkg.db.query.constants import GENERIC_EN_IN

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


def load_line_budget_forecasts(
    *, earliest_edition_only: bool = True, **engine_kwargs
) -> pd.DataFrame:
    """Load Line Budget forecasts for TARGET_GENERIC_EN products.

    Version encoding is YYYYQQEE (e.g. 14040401 = 1404Q4 edition 1).
    FK_Date_ID (YYYYMMDD) is converted to Shamsi YYYYMM. BudgetQty is the
    forecast quantity.

    If ``earliest_edition_only`` is True (default), keep only the earliest
    ``Version`` per (product, qrt) — typically edition 01 when present.

    Returns columns: product, generic, date, forecast, version, qrt.
    """
    df = read_sql(LINE_BUDGET_FORECASTS, **engine_kwargs)
    if df.empty:
        return pd.DataFrame(
            columns=["product", "generic", "date", "forecast", "version", "qrt"]
        )

    df["version"] = df["version"].astype(int)
    df["date"] = (df["fk_date_id"].astype(int) // 100).astype(int)
    df["forecast"] = pd.to_numeric(df["forecast"], errors="coerce")
    df["qrt"] = df["version"].map(version_to_qrt)
    df["product"] = df["product"].astype(str)

    if earliest_edition_only:
        earliest = (
            df.groupby(["product", "qrt"], as_index=False)["version"]
            .min()
            .rename(columns={"version": "_earliest_version"})
        )
        df = df.merge(earliest, on=["product", "qrt"], how="inner")
        df = df.loc[df["version"] == df["_earliest_version"]].drop(
            columns=["_earliest_version"]
        )

    return df[["product", "generic", "date", "forecast", "version", "qrt"]].copy()
