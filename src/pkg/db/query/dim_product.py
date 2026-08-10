"""Sales-related SQL queries."""
from pkg.db.client import read_sql

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
