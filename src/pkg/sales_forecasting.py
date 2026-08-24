import csv
import logging
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm
from pkg.db.query.dim_product import load_basket_products
from pkg.db.query.sales import load_sales_data as fetch_sales_data
from pkg.forecast import SalesForecast
from pkg.utils import (
    DATA_DIR,
    define_path,
    drop_unmapped_departments,
    setup_forecast_file,
    update_department_info,
    pivot_and_format_data,
    manage_excel,
)
from dotenv import load_dotenv
import os

load_dotenv()

class SalesForecasting:
    def __init__(self, curr_qrt):
        self.curr_qrt = curr_qrt
        self.forecasts = define_path(curr_qrt)
        self.headers = ['product', 'product_fa', 'date', 'provider', 'model', 'dep', 'status', 'forecast']
        logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
        setup_forecast_file(self.forecasts, self.headers)

    def load_sales_data(self):
        return fetch_sales_data()

    def load_forecast_data(self):
        return pd.read_csv(self.forecasts)

    @staticmethod
    def truncate_sales_before_origin(sale_df, forecast_start_date):
        """Keep Shamsi months strictly before the forecast origin (as-of vintage).

        Leaves ``[:-1]`` incomplete-month handling in SalesForecast unchanged.
        When the warehouse last month is origin-1, live training still ends at origin-2.
        """
        if sale_df is None or sale_df.empty:
            return sale_df
        work = sale_df.copy()
        work["date"] = work["date"].astype(int)
        origin = int(forecast_start_date)
        before = len(work)
        work = work[work["date"] < origin].copy()
        print(
            f"vintage sales cutoff: {before} -> {len(work)} rows "
            f"(date < {origin}, {work['product'].nunique() if not work.empty else 0} products)"
        )
        return work

    def reset_forecast_csv(self, force=False):
        """Backup (if needed) and rewrite the forecast CSV to headers only."""
        path = Path(self.forecasts)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > 0:
            if not force:
                raise SystemExit(
                    f"Refusing to overwrite existing {path}. Re-run with --force "
                    "(vintage must start from an empty CSV so all basket SKUs are processed)."
                )
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = path.with_name(f"{path.stem}.bak_{stamp}{path.suffix}")
            shutil.copy2(path, backup_path)
            print(f"backed up existing CSV -> {backup_path}")
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
        print(f"reset forecast CSV (headers only): {path}")

    @staticmethod
    def _stub_sale_df(product, attrs, forecast_start_date):
        """One-row sale frame from Dim.Product attrs for SKUs with no history."""
        orchid = attrs.get("OrchidBoxQuantity")
        boxq = orchid if pd.notna(orchid) else attrs.get("BoxQuantity")
        return pd.DataFrame(
            [
                {
                    "product": product,
                    "product_fa": attrs.get("Title"),
                    "provider": attrs.get("Provider"),
                    "dep": attrs.get("Field"),
                    "boxq": boxq,
                    "date": int(forecast_start_date) + 62100,
                    "sales": 0,
                }
            ]
        )

    @staticmethod
    def _write_zero_forecast(prod_fr, forecast_start_date):
        strat_month = pd.to_datetime(forecast_start_date + 62100, format="%Y%m")
        prod_fr.forecast_index = pd.date_range(strat_month, periods=15, freq="MS")
        prod_fr.forecast = np.zeros(15)
        prod_fr.save_csv()

    def process_sales_data(
        self,
        sale_df_total,
        forecast_df,
        forecast_start_date,
        skip_forecast=False,
        basket_df=None,
    ):
        products_fr = (
            set(pd.unique(forecast_df["product"].astype(str)))
            if forecast_df is not None and forecast_df.shape[0] > 0
            else set()
        )

        if sale_df_total is None or sale_df_total.empty:
            sale_df_total = pd.DataFrame(
                columns=[
                    "product",
                    "product_fa",
                    "date",
                    "provider",
                    "dep",
                    "boxq",
                    "sales",
                ]
            )
        else:
            sale_df_total = sale_df_total.copy()
            sale_df_total["product"] = sale_df_total["product"].astype(str)
            sale_df_total["date"] = sale_df_total["date"].astype(int)
            sale_df_total["date"] = sale_df_total["date"] + 62100

        if basket_df is not None:
            # Production path: Dim.Product ProductBasket=1 (may be empty)
            products = (
                basket_df["ProductTitleEN"].astype(str).tolist()
                if not basket_df.empty
                else []
            )
            attrs_by_product = (
                basket_df.assign(ProductTitleEN=basket_df["ProductTitleEN"].astype(str))
                .set_index("ProductTitleEN")
                .to_dict("index")
                if not basket_df.empty
                else {}
            )
            if not products:
                print("Warning: basket product universe is empty (ProductBasket=1).")
        else:
            # Legacy callers (e.g. backfill_vintage): sales-only universe
            products = list(pd.unique(sale_df_total["product"])) if not sale_df_total.empty else []
            attrs_by_product = {}

        products = sorted(products, key=str.casefold)

        for product in tqdm(products, desc="Processing products", unit="product"):
                if product not in products_fr:
                    if not skip_forecast:
                        print(f'\n{product} is in progress!')
                    sale_df = sale_df_total[sale_df_total['product'] == product]
                    if sale_df.empty:
                        if product not in attrs_by_product:
                            print(
                                f"Skipping {product}: no sales and no Dim.Product attrs."
                            )
                            continue
                        sale_df = self._stub_sale_df(
                            product, attrs_by_product[product], forecast_start_date
                        )
                        prod_fr = SalesForecast(product, sale_df, self.forecasts)
                        self._write_zero_forecast(prod_fr, forecast_start_date)
                        continue

                    prod_fr = SalesForecast(product, sale_df, self.forecasts)

                    # Output-only mode: write zero forecast for every product, no model run
                    if skip_forecast:
                        self._write_zero_forecast(prod_fr, forecast_start_date)
                        continue

                    ZERO_FORECAST_PRODUCTS = os.getenv('ZERO_FORECAST_PRODUCTS')
                    if ZERO_FORECAST_PRODUCTS and (prod_fr.product in ZERO_FORECAST_PRODUCTS):
                        self._write_zero_forecast(prod_fr, forecast_start_date)
                        continue

                    prod_fr.preprocess_data()

                    # Stale vs requested origin (not a fixed calendar date): zero if last
                    # sale is more than 3 months before forecast_start_date.
                    origin_greg = pd.to_datetime(forecast_start_date + 62100, format='%Y%m')
                    stale_before = origin_greg - pd.DateOffset(months=3)
                    if (
                        (prod_fr.sale_series == 0).all() |
                        (prod_fr.prophet_df['y'] == 0).all() |
                        (prod_fr.prophet_df.ds.max() < stale_before)
                        ):
                        self._write_zero_forecast(prod_fr, forecast_start_date)
                        continue

                    if (len(prod_fr.sale_series) < 4):
                        self._write_zero_forecast(prod_fr, forecast_start_date)
                        continue

                    prod_fr.model_selection()
                    try:
                        prod_fr.predict()
                        prod_fr.redistribute_smoothing()
                        prod_fr.save_csv()
                    except ValueError:
                        self._write_zero_forecast(prod_fr, forecast_start_date)
                        continue    
                else:
                    continue

        forecast_total_df = pd.read_csv(self.forecasts)
        return forecast_total_df

    def append_pipeline(self, pivot, updated_dep_dict):

        file_path = os.path.join(DATA_DIR, 'pipeline', self.curr_qrt, f'{self.curr_qrt}_pipeline.xlsx')
        pipeline_df = pd.read_excel(file_path)
        required_columns = {'product_fa', 'provider', 'dep'}
        if not required_columns.issubset(pipeline_df.columns):
            raise ValueError(f'Excel file must contain the following columns: \n{required_columns}')
        
        pipeline_df['file_name'] = pipeline_df.dep.map(updated_dep_dict)
        if pipeline_df['file_name'].isnull().any():
            raise ValueError('Some departments in the pipeline file are not found in department dictionary.')
        pipeline_records = []
        for _, row in pipeline_df.iterrows():
            record = {
                'product_fa': row['product_fa'],
                'dep': row['dep'],
                'provider': row['provider'],
                'status': 'عدد',
                'file_name': row['file_name'],
            }
            pipeline_records.append(record)
        
        pipeline_pivot = pd.DataFrame(pipeline_records)
        aligned_pipe_piv = pipeline_pivot.reindex(columns=pivot.columns, fill_value=0)
        updated_pivot = pd.concat([pivot, aligned_pipe_piv], ignore_index=True)
        return updated_pivot

    def run(
        self,
        forecast_start_date,
        generate_forecasts=True,
        vintage=False,
        force=False,
    ):
        basket_df = load_basket_products()
        sale_df_total = self.load_sales_data()
        if vintage:
            sale_df_total = self.truncate_sales_before_origin(
                sale_df_total, forecast_start_date
            )
            self.reset_forecast_csv(force=force)
            forecast_df = pd.DataFrame(columns=self.headers)
        elif not generate_forecasts:
            # Reset forecast file to headers only so we build output with zeros only
            os.makedirs(os.path.dirname(self.forecasts), exist_ok=True)
            with open(self.forecasts, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
            forecast_df = pd.DataFrame(columns=self.headers)
        else:
            forecast_df = self.load_forecast_data()

        forecast_total_df = self.process_sales_data(
            sale_df_total,
            forecast_df,
            forecast_start_date,
            skip_forecast=not generate_forecasts,
            basket_df=basket_df,
        )
        updated_dep_dict = update_department_info( self.curr_qrt)

        forecast_total_df['sales'] = forecast_total_df['forecast']
        forecast_total_df['type'] = 'forecast'
        if sale_df_total is not None and not sale_df_total.empty:
            sale_df_total['type'] = 'actual'

        # temp = pd.concat([sale_df_total, forecast_total_df])
        # forecast_total_df_mod = replace_negative_sales(temp)

        pivot = pivot_and_format_data(forecast_total_df, updated_dep_dict, forecast_start_date)
        pivot = drop_unmapped_departments(pivot)
        # updated_pivot = self.append_pipeline(pivot, updated_dep_dict)
        manage_excel(pivot, os.path.join(DATA_DIR, 'results', self.curr_qrt), self.curr_qrt)
