"""F3E peer-demand SQL query.

Joins Flat_Fact_Sale to Dim.Product in the database so that all F3E-relevant
product dimension columns are returned alongside aggregated monthly sales.

The peer universe is ALL products passing the WHERE filter (not restricted to
MVP benchmark products).  Negative monthly_dqty sums are retained — the
clipping / representation policy for negative aggregate peer demand is
deferred to F3E Step 2.

Filtering semantics deliberately match the canonical pipeline:
  - ProductTitleEN IS NOT NULL
  - Field != '-'
  - Group by product × Shamsi month (same granularity as load_sales_data)
"""
from pkg.db.client import read_sql

PEER_SALES_QUERY = """
SELECT
    p.[ProductTitleEN]          AS product,
    s.[ShamsiYearMonth]         AS date,
    SUM(s.[DQty])               AS monthly_dqty,
    p.[FKGeneric],
    p.[Field],
    p.[Unit]                    AS unit_ratio,
    p.[PatientConsumeType],
    p.[PatientConsumePerPeriod],
    p.[ID_INT]                  AS product_id
FROM [DWOrchid].[dbo].[Flat_Fact_Sale] s WITH (NOLOCK)
INNER JOIN [Iris_DW].[Dim].[Product] p
    ON s.[FKProduct] = p.[ID_INT]
WHERE p.[ProductTitleEN] IS NOT NULL
  AND p.[Field] != '-'
GROUP BY
    p.[ProductTitleEN],
    s.[ShamsiYearMonth],
    p.[FKGeneric],
    p.[Field],
    p.[Unit],
    p.[PatientConsumeType],
    p.[PatientConsumePerPeriod],
    p.[ID_INT]
"""


def load_peer_sales(**engine_kwargs) -> "pd.DataFrame":
    """Load aggregated monthly peer sales with product-dimension columns.

    Returns a DataFrame with columns:
        product, date, monthly_dqty,
        FKGeneric, Field, unit_ratio,
        PatientConsumeType, PatientConsumePerPeriod, product_id
    """
    import pandas as pd  # local import to keep module importable without pandas
    df = read_sql(PEER_SALES_QUERY, **engine_kwargs)
    df["date"] = pd.to_numeric(df["date"], errors="coerce").astype("Int64")
    df["monthly_dqty"] = pd.to_numeric(df["monthly_dqty"], errors="coerce")
    df["unit_ratio"] = pd.to_numeric(df["unit_ratio"], errors="coerce")
    df["PatientConsumePerPeriod"] = pd.to_numeric(
        df["PatientConsumePerPeriod"], errors="coerce"
    )
    df["product"] = df["product"].astype(str)
    return df
