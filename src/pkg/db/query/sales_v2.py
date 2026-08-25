"""Proposed V2 sales query — FK join on ``ID_INT`` (not used by V1).

See ``docs/ts_v2_product_identity.md``. Wire through ``pkg.ts_v2.data.load_monthly_sales``
only after ``run_product_identity_audit`` passes on live extracts.
"""
from pkg.db.client import read_sql

SALES_BY_PRODUCT_MONTH_V2 = """
    SELECT
        p.[ID_INT] AS product_id,
        p.[ProductTitleEN] AS product_title,
        s.[ProductTitle] AS product_fa,
        s.[ShamsiYearMonth] AS date,
        s.[GenericProvider] AS provider,
        s.[GenericField] AS dep,
        s.[mappedBoxQuantity] AS boxq,
        SUM(s.[DQTY]) AS sales,
        s.[FKProduct] AS fk_product,
        s.[ProductTitleEN] AS fact_product_title,
        p.[ProductTitleEN] AS dim_product_title
    FROM [DWOrchid].[dbo].[Flat_Fact_Sale] s WITH (NOLOCK)
    INNER JOIN [Iris_DW].[Dim].[Product] p
        ON s.[FKProduct] = p.[ID_INT]
    WHERE p.[ProductTitleEN] IS NOT NULL
      AND s.[GenericField] != '-'
    GROUP BY
        p.[ID_INT],
        p.[ProductTitleEN],
        s.[ProductTitle],
        s.[ShamsiYearMonth],
        s.[GenericProvider],
        s.[GenericField],
        s.[mappedBoxQuantity],
        s.[FKProduct],
        s.[ProductTitleEN]
    ORDER BY
        p.[ProductTitleEN],
        s.[ShamsiYearMonth],
        sales DESC
"""


def load_sales_data_v2(**engine_kwargs):
    """Load monthly sales keyed by ``ID_INT`` (proposed V2 path; not production)."""
    return read_sql(SALES_BY_PRODUCT_MONTH_V2, **engine_kwargs)
