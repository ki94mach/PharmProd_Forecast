import logging
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from tqdm import tqdm
from pkg.forecast import SalesForecast
from pkg.utils import define_path, setup_forecast_file, update_department_info, pivot_and_format_data, manage_excel

class SalesForecasting:
    def __init__(self, curr_qrt):
        self.server = 'op-db1-srv'
        self.database = 'DWOrchid'
        self.curr_qrt = curr_qrt
        self.forecasts = define_path(curr_qrt)
        self.headers = ['product', 'product_fa', 'date', 'provider', 'model', 'dep', 'status', 'forecast']
        logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
        setup_forecast_file(self.forecasts, self.headers)

    def load_sales_data(self):
        query ="""
            SELECT [ProductTitle] AS product_fa,
            [ProductTitleEN] AS product,
            [ShamsiYearMonth] AS date,
            [GenericProvider] AS provider,
            [GenericField] AS dep,
            [mappedBoxQuantity] AS boxq,
            SUM([DQTY]) as sales
            FROM [DWOrchid].[dbo].[Flat_Fact_Sale]
            WHERE ProductTitleEN IS NOT NULL AND [GenericField] != '-'
            GROUP BY [ProductTitle],
            [ProductTitleEN],
            [ShamsiYearMonth],
            [GenericProvider],
            [GenericField],
            [mappedBoxQuantity]
            ORDER BY [ProductTitleEn], [ShamsiYearMonth], sales Desc
            """
        connection_url = URL.create(
            "mssql+pyodbc",
            query={
                "odbc_connect": f"DRIVER={{SQL Server}};SERVER={self.server};DATABASE={self.database};Trusted_Connection=yes;"
            }
        )
        engine = create_engine(connection_url)
        with engine.connect() as connection:
            sale_df_total = pd.read_sql(query, connection)
        return sale_df_total

    def load_forecast_data(self):
        return pd.read_csv(self.forecasts)

    def process_sales_data(self, sale_df_total, forecast_df, forecast_start_date):
        sale_df_total['date'] = sale_df_total['date'].astype(int)
        products_fr = pd.unique(forecast_df['product'])
        products = pd.unique(sale_df_total['product'])
        sale_df_total.date += 62100

        for product in tqdm(products, desc="Processing products", unit="product"):
                if product not in products_fr:
                    print(f'\n{product} is in progress!')
                    sale_df = sale_df_total[sale_df_total['product'] == product]
                    prod_fr = SalesForecast(product, sale_df, self.forecasts)

                    if (prod_fr.product in ["Solariba", "Suliba 100" ,"Tabinoz"]):
                        strat_month = pd.to_datetime(forecast_start_date + 62100, format='%Y%m') 
                        prod_fr.forecast_index = pd.date_range(strat_month, periods=15, freq='MS')
                        prod_fr.forecast = np.zeros(15)
                        prod_fr.save_csv()
                        continue

                    prod_fr.preprocess_data()

                    if (
                        (prod_fr.sale_series == 0).all() | 
                        (prod_fr.prophet_df['y'] == 0).all() | 
                        (prod_fr.prophet_df.ds.max() < np.datetime64('2021-01-01'))):
                        continue

                    if (len(prod_fr.sale_series) < 4):
                        prod_fr.forecast = np.zeros(15)
                        prod_fr.save_csv()
                        continue

                    prod_fr.model_selection()
                    try:
                        prod_fr.predict()
                        prod_fr.redistribute_smoothing()
                        prod_fr.save_csv()
                    except ValueError:
                        prod_fr.forecast = np.zeros(15)
                        prod_fr.save_csv()
                        continue    
                else:
                    continue

        forecast_total_df = pd.read_csv(self.forecasts)
        return forecast_total_df

    def run(self, forecast_start_date):
        sale_df_total = self.load_sales_data()
        forecast_df = self.load_forecast_data()

        forecast_total_df = self.process_sales_data(sale_df_total, forecast_df, forecast_start_date)
        updated_dep_dict = update_department_info( self.curr_qrt)

        forecast_total_df['sales'] = forecast_total_df['forecast']
        forecast_total_df['type'] = 'forecast'
        sale_df_total['type'] = 'actual'

        # temp = pd.concat([sale_df_total, forecast_total_df])
        # forecast_total_df_mod = replace_negative_sales(temp)

        pivot = pivot_and_format_data(forecast_total_df, updated_dep_dict, forecast_start_date)
        manage_excel(pivot, directory=f"data/results/{self.curr_qrt}")
