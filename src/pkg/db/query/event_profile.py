"""Event profile product SQL queries."""
from pkg.db.client import read_sql
from pkg.db.query.constants import GENERIC_EN_IN

EVENT_COUNT_BY_PRODUCT = f"""
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


def load_event_count_by_product(**engine_kwargs):
    """Load unique active event counts per product from VW_Eventprofile_Product.

    Events with Generic_ID are counted for that product. Events with null
    Generic_ID are counted for every product whose Field matches the event's
    department.
    """
    return read_sql(EVENT_COUNT_BY_PRODUCT, **engine_kwargs)
