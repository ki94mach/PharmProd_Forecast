SELECT
    CAST(fi.FKDate AS date) AS snapshot_date,

    fi.FKProduct AS fk_product,

    p.ID AS product_id,
    p.ProductTitleEN AS product,
    p.GenericEN AS generic,
    p.FKGeneric AS generic_id,

    SUM(
        CASE
            WHEN fi.[Status] IN (N'موجودی', N'موجودي')
            THEN CAST(fi.DQty AS decimal(38, 6))
            ELSE 0
        END
    ) AS distributor_on_hand_qty,

    SUM(
        CASE
            WHEN fi.[Status] = N'در راه'
            THEN CAST(fi.DQty AS decimal(38, 6))
            ELSE 0
        END
    ) AS distributor_in_transit_qty,

    SUM(
        CASE
            WHEN fi.[Status] IN (
                N'موجودی',
                N'موجودي',
                N'در راه'
            )
            THEN CAST(fi.DQty AS decimal(38, 6))
            ELSE 0
        END
    ) AS distributor_inventory_qty,

    SUM(
        CASE
            WHEN fi.[Status] = N'بلوکه'
            THEN CAST(fi.DQty AS decimal(38, 6))
            ELSE 0
        END
    ) AS blocked_inventory_qty,

    SUM(
        CASE
            WHEN fi.[Status] NOT IN (
                N'موجودی',
                N'موجودي',
                N'در راه',
                N'بلوکه'
            )
            THEN 1
            ELSE 0
        END
    ) AS n_unknown_status_rows,

    COUNT_BIG(*) AS source_row_count,

    COUNT(DISTINCT fi.FKDistributor) AS n_distributors,
    COUNT(DISTINCT fi.FKCenter) AS n_centers

FROM [DWOrchid].[dbo].[FactInventoryHistorical] fi

LEFT JOIN [Iris_DW].[Dim].[Product] p
    ON p.ID_INT = fi.FKProduct

WHERE
    fi.FKDate IS NOT NULL
    AND fi.FKDistributor IS NOT NULL
    AND fi.FKCenter IS NOT NULL
    AND fi.FKProduct IS NOT NULL
    AND fi.DQty IS NOT NULL

GROUP BY
    CAST(fi.FKDate AS date),
    fi.FKProduct,
    p.ID,
    p.ProductTitleEN,
    p.GenericEN,
    p.FKGeneric

ORDER BY
    snapshot_date,
    product;
