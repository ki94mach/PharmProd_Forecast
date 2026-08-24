"""Sales-related SQL queries."""
import pandas as pd

from pkg.db.client import read_sql

BASKET_PRODUCT_COLUMNS = [
    "ProductTitleEN",
    "Title",
    "Provider",
    "Field",
    "OrchidBoxQuantity",
    "BoxQuantity",
]

DIM_PRODUCT = """
    SELECT  [ID]
        ,[ID_INT]
        ,[Title]
        ,[Provider]
        ,[OrchidPharmed]
        ,[Generic]
        ,[BrandGroup]
        ,[RituximabBase]
        ,[Field]
        ,[Weight]
        ,[Unit]
        ,[ProductTitleEN]
        ,[GenericEN]
        ,[FKGeneric]
        ,[LastPriceUnit]
        ,[UnitNormalize]
        ,[ProductBasket]
        ,[ProviderOrder]
        ,[PatientConsumeType]
        ,[CurePeriod]
        ,[PatientConsumePerPeriod]
        ,[PatientConsumePerPeriodUnit]
        ,[FinanceCode]
        ,[StatusCode]
        ,[GenericBasket]
        ,[ProductType]
        ,[ProductForm]
        ,[OrchidBoxQuantity]
        ,[BoxQuantity]
    FROM [Iris_DW].[Dim].[Product]
"""


def load_dim_product(**engine_kwargs):
    """Load dim product from Dim.Product."""
    return read_sql(DIM_PRODUCT, **engine_kwargs)


def select_basket_products(dim_df: pd.DataFrame) -> pd.DataFrame:
    """Keep active ProductBasket=1 rows with a real Field and non-blank ProductTitleEN; dedupe."""
    if dim_df is None or dim_df.empty:
        return pd.DataFrame(columns=BASKET_PRODUCT_COLUMNS)

    work = dim_df.copy()
    titles = work["ProductTitleEN"].astype("string").str.strip()
    work = work.loc[titles.notna() & titles.ne("") & titles.ne("nan")].copy()
    work["ProductTitleEN"] = (
        work["ProductTitleEN"].astype("string").str.strip().astype(str)
    )
    work = work.loc[
        pd.to_numeric(work["ProductBasket"], errors="coerce") == 1
    ].copy()
    fields = work["Field"].astype("string").str.strip()
    work = work.loc[fields.notna() & fields.ne("") & fields.ne("-")].copy()
    if "StatusCode" not in work.columns:
        return pd.DataFrame(columns=BASKET_PRODUCT_COLUMNS)
    status = work["StatusCode"].astype("string").str.strip()
    work = work.loc[status == "Active"].copy()
    work = work.drop_duplicates(subset=["ProductTitleEN"], keep="first")
    work = work.sort_values(
        "ProductTitleEN", key=lambda s: s.str.casefold(), kind="mergesort"
    )
    missing = [c for c in BASKET_PRODUCT_COLUMNS if c not in work.columns]
    for col in missing:
        work[col] = pd.NA
    return work.loc[:, BASKET_PRODUCT_COLUMNS].reset_index(drop=True)


def load_basket_products(**engine_kwargs) -> pd.DataFrame:
    """Dim.Product rows in the commercial basket (ProductBasket=1, Field != '-', StatusCode=Active)."""
    return select_basket_products(load_dim_product(**engine_kwargs))

