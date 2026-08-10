"""Event profile product SQL queries."""
from pkg.db.client import read_sql
from pkg.db.query.constants import GENERIC_EN_IN


def _event_count_by_product_sql(start_ym: int, end_ym: int) -> str:
    start = int(start_ym)
    end = int(end_ym)
    return f"""
    WITH event_products AS (
        -- Events with an assigned product (Generic_ID)
        SELECT
            a.[op_id],
            b.[ProductTitleEN],
            b.[GenericEN]
        FROM [Iris_DW].[Fact].[VW_Eventprofile_Product] a
        INNER JOIN [Iris_DW].[Dim].[Product] b
            ON a.[Generic_ID] = b.[ID]
        WHERE a.[Event_Statecode] = 'Active'
            AND a.[Generic_ID] IS NOT NULL
            AND b.[GenericEN] IN ({GENERIC_EN_IN})
            AND LEFT(LTRIM(RTRIM(CAST(a.[ShamsiDate] AS varchar(20)))), 6)
                BETWEEN '{start}' AND '{end}'

        UNION

        -- Events with no product: attribute to all products in the department
        SELECT
            a.[op_id],
            p.[ProductTitleEN],
            p.[GenericEN]
        FROM [Iris_DW].[Fact].[VW_Eventprofile_Product] a
        INNER JOIN [Iris_DW].[Dim].[Department] d
            ON a.[DepartmentID] = d.[DepartmentID]
        INNER JOIN [Iris_DW].[Dim].[Product] p
            ON d.[Department] = p.[Field]
        WHERE a.[Event_Statecode] = 'Active'
            AND a.[Generic_ID] IS NULL
            AND p.[GenericEN] IN ({GENERIC_EN_IN})
            AND LEFT(LTRIM(RTRIM(CAST(a.[ShamsiDate] AS varchar(20)))), 6)
                BETWEEN '{start}' AND '{end}'
    )
    SELECT
        [ProductTitleEN] AS product,
        [GenericEN] AS generic,
        COUNT(DISTINCT [op_id]) AS event_count
    FROM event_products
    GROUP BY
        [ProductTitleEN],
        [GenericEN]
    ORDER BY
        [GenericEN],
        [ProductTitleEN]
"""


def load_event_count_by_product(start_ym, end_ym, **engine_kwargs):
    """Load unique active event counts per product in a Shamsi YYYYMM window.

    Events with Generic_ID are counted for that product. Events with null
    Generic_ID are counted for every product whose Field matches the event's
    department.

    Args:
        start_ym: Inclusive Shamsi year-month start (e.g. 140407).
        end_ym: Inclusive Shamsi year-month end (e.g. 140412).
    """
    return read_sql(_event_count_by_product_sql(start_ym, end_ym), **engine_kwargs)
