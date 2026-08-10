"""Visit activity SQL queries."""
from pkg.db.client import read_sql
from pkg.db.query.constants import GENERIC_EN_IN


def _activity_count_by_product_sql(start_ym: int, end_ym: int) -> str:
    start = int(start_ym)
    end = int(end_ym)
    return f"""
    SELECT
        p.[ProductTitleEN] AS product,
        p.[GenericEN] AS generic,
        COUNT(DISTINCT a.[ActivityID]) AS activity_count
    FROM [Iris_DW].[Fact].[Activity] a
    INNER JOIN [Iris_DW].[Dim].[Date] dt
        ON CAST(a.[FK_Date_ID] AS date) = dt.[DateID]
    INNER JOIN [Iris_DW].[Dim].[Product] p
        ON a.[BundleGenericID] = p.[FKGeneric]
    WHERE a.[Position] = 'MedRep'
        AND a.[AcceptedVisits] = 'Accepted'
        AND a.[BundleGenericID] IS NOT NULL
        AND p.[GenericEN] IN ({GENERIC_EN_IN})
        AND LTRIM(RTRIM(dt.[ShamsiYearMonth])) BETWEEN '{start}' AND '{end}'
    GROUP BY
        p.[ProductTitleEN],
        p.[GenericEN]
    ORDER BY
        p.[GenericEN],
        p.[ProductTitleEN]
"""


def load_activity_count_by_product(start_ym, end_ym, **engine_kwargs):
    """Load unique accepted MedRep activity counts per product in a Shamsi window.

    Activities are at generic grain via BundleGenericID; each matching product
    under that FKGeneric receives the activity count.

    Args:
        start_ym: Inclusive Shamsi year-month start (e.g. 140407).
        end_ym: Inclusive Shamsi year-month end (e.g. 140412).
    """
    return read_sql(
        _activity_count_by_product_sql(start_ym, end_ym),
        **engine_kwargs,
    )
