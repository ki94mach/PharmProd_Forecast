SELECT
    CAST(fi.FKDate AS date) AS snapshot_date,

    fi.FKProduct AS fk_product,

    p.ID AS product_id,
    p.ProductTitleEN AS product,
    p.GenericEN AS generic,
    p.FKGeneric AS generic_id,

    SUM(
        CAST(fi.DQty AS decimal(38, 6))
    ) AS factory_inventory_qty,

    COUNT_BIG(*) AS source_row_count,

    COUNT(DISTINCT fi.FkProvider) AS n_factories,
    COUNT(DISTINCT fi.FkStore) AS n_stores,

    SUM(
        CASE
            WHEN fi.DQty = 0 THEN 1
            ELSE 0
        END
    ) AS n_zero_source_rows,

    SUM(
        CASE
            WHEN fi.DQty < 0 THEN 1
            ELSE 0
        END
    ) AS n_negative_source_rows

FROM [DWOrchid].[dbo].[FactInventory] fi

LEFT JOIN [Iris_DW].[Dim].[Product] p
    ON p.ID_INT = fi.FKProduct

WHERE
    fi.FKDate IS NOT NULL
    AND fi.FkProvider IS NOT NULL
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
