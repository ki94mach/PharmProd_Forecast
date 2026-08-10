"""Sales-related SQL queries."""
from pkg.db.client import read_sql

SALES_BY_PRODUCT_MONTH = """
    SELECT [ProductTitle] AS product_fa,
    [ProductTitleEN] AS product,
    [ShamsiYearMonth] AS date,
    [GenericProvider] AS provider,
    [GenericField] AS dep,
    [mappedBoxQuantity] AS boxq,
    SUM([DQTY]) as sales
    FROM [DWOrchid].[dbo].[Flat_Fact_Sale] WITH (NOLOCK)
    WHERE ProductTitleEN IS NOT NULL AND [GenericField] != '-'
    GROUP BY [ProductTitle],
    [ProductTitleEN],
    [ShamsiYearMonth],
    [GenericProvider],
    [GenericField],
    [mappedBoxQuantity]
    ORDER BY [ProductTitleEn], [ShamsiYearMonth], sales Desc
"""


def load_sales_data(**engine_kwargs):
    """Load aggregated monthly sales by product from Flat_Fact_Sale."""
    return read_sql(SALES_BY_PRODUCT_MONTH, **engine_kwargs)
