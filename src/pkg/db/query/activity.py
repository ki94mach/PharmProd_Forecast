"""Visit activity SQL queries."""
from pkg.db.client import read_sql
from pkg.db.query.constants import GENERIC_EN_IN

ACTIVITY_COUNT_BY_PRODUCT = f"""
    SELECT
        p.[ProductTitleEN] AS product,
        p.[GenericEN] AS generic,
        COUNT(DISTINCT a.[ActivityID]) AS activity_count
    FROM [Iris_DW].[Fact].[Activity] a
    INNER JOIN [Iris_DW].[Dim].[Product] p
        ON a.[BundleGenericID] = p.[FKGeneric]
    WHERE a.[Position] = 'MedRep'
        AND a.[AcceptedVisits] = 'Accepted'
        AND a.[BundleGenericID] IS NOT NULL
        AND p.[GenericEN] IN ({GENERIC_EN_IN})
    GROUP BY
        p.[ProductTitleEN],
        p.[GenericEN]
    ORDER BY
        p.[GenericEN],
        p.[ProductTitleEN]
"""


def load_activity_count_by_product(**engine_kwargs):
    """Load unique accepted MedRep activity counts per product.

    Activities are at generic grain via BundleGenericID; each matching product
    under that FKGeneric receives the activity count.
    """
    return read_sql(ACTIVITY_COUNT_BY_PRODUCT, **engine_kwargs)
