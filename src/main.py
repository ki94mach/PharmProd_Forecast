from pkg.forecast import SalesForecast
import pandas as pd
import os
import csv
if __name__ == "__main__":
    sale = 'data/sales/monthly_sales.csv'
    forecasts = 'data/results/Forecasts test.csv'
    headers=['product', 'date', 'forecast', 'model']
    if not os.path.exists(forecasts):
        with open(forecasts, "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
    sale_df_total = pd.read_csv(sale)
    forecast_df = pd.read_csv(forecasts)

    products_fr = pd.unique(forecast_df['product'])
    products = pd.unique(sale_df_total['product'])
    sale_df_total.date += 62100
    # for product in products:
    #     if product not in products_fr:
    #         sale_df = sale_df_total[sale_df_total['product'] == product]
    #         prod_fr = SalesForecast(product, sale_df, forecasts)
    #         prod_fr.preprocess_data()
    #         prod_fr.model_selection()
    #         prod_fr.predict()
    #         prod_fr.save()
    #     else:
    #         continue
    sale_df = sale_df_total[sale_df_total['product'] == "Altebrel 50"]
    prod_fr = SalesForecast("Altebrel 50", sale_df, forecasts)
    prod_fr.preprocess_data()
    prod_fr.model_selection()
    prod_fr.predict()
    # prod_fr.redistribute_smoothing()
    prod_fr.plot()
