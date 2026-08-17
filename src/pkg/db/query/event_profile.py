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


def _product_launch_date_sql() -> str:
    return f"""
    WITH launch_events AS (
        SELECT
            a.[Generic_ID],
            a.[ShamsiDate]
        FROM [Iris_DW].[Fact].[VW_Eventprofile_Product] a
        WHERE a.[EventType] LIKE N'%لانچ دارو Drug Launch%'
            AND a.[Generic_ID] IS NOT NULL
            AND a.[ShamsiDate] IS NOT NULL
    ),
    date_counts AS (
        SELECT
            [Generic_ID],
            [ShamsiDate],
            COUNT(*) AS n_event_rows,
            ROW_NUMBER() OVER (
                PARTITION BY [Generic_ID]
                ORDER BY COUNT(*) DESC, [ShamsiDate] ASC
            ) AS rn
        FROM launch_events
        GROUP BY
            [Generic_ID],
            [ShamsiDate]
    )
    SELECT
        p.[ProductTitleEN] AS product,
        p.[GenericEN] AS generic,
        CAST(LEFT(LTRIM(RTRIM(CAST(d.[ShamsiDate] AS varchar(20)))), 6) AS int) AS date,
        d.[ShamsiDate] AS launch_date,
        d.[n_event_rows]
    FROM date_counts d
    INNER JOIN [Iris_DW].[Dim].[Product] p
        ON d.[Generic_ID] = p.[ID]
    WHERE d.[rn] = 1
        AND p.[ProductTitleEN] IS NOT NULL
        AND p.[GenericEN] IN ({GENERIC_EN_IN})
    ORDER BY
        p.[GenericEN],
        p.[ProductTitleEN]
"""


def load_product_launch_dates(**engine_kwargs):
    """Load the modal Drug Launch ShamsiDate per Generic_ID as product YYYYMM.

    For each Generic_ID, the ShamsiDate with the most event rows is kept
    (earliest date on ties). Joined to Dim.Product so each ProductTitleEN
    has a ``date`` column compatible with sales (Shamsi YYYYMM).
    """
    return read_sql(_product_launch_date_sql(), **engine_kwargs)
