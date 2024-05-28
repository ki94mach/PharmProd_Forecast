import os
import csv
import logging
import pandas as pd
import numpy as np
from pkg.forecast import SalesForecast
from pkg.excelmanager import ExcelManager

logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
if __name__ == "__main__":
    # The monthly sales csv should contain
    # "product", "date", "dep", "sale" column
    sale = 'data/sales/1403Q2/monthly_sales.csv'
    forecasts = 'data/results/1403Q2/1403Q2_total_forecast.csv'
    headers=['product', 'date', 'forecast', 'model','dep']
    if not os.path.exists(forecasts):
        with open(forecasts, "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
    sale_df_total = pd.read_csv(sale)
    forecast_df = pd.read_csv(forecasts)

    products_fr = pd.unique(forecast_df['product'])
    products = pd.unique(sale_df_total['product'])
    sale_df_total.date += 62100
    for product in products:
        if product not in products_fr:
            sale_df = sale_df_total[sale_df_total['product'] == product]
            prod_fr = SalesForecast(product, sale_df, forecasts)
            prod_fr.preprocess_data()
            prod_fr.model_selection()
            prod_fr.predict()
            prod_fr.redistribute_smoothing()
            prod_fr.save_csv()
        else:
            continue
    forecast_total_df = pd.read_csv(forecasts)
    dep_dict = {
        "انکولوژی بیولوژیک": '1403Q2 Bio Onco',
        'انکولوژی شیمیایی': '1403Q2 Chem Onco',
        'رسپیراتوری': '1403Q2 Resp',
        'کانتراست مدیا': '1403Q2 Contrast',
        'زیبایی': '1403Q2 Beaut',
        'نورولوژی بیولوژیک': '1403Q2 Bio Neuro',
        'انکولوژی اطفال': '1403Q2 Ped Onco',
        'نفرولوژی': '1403Q2 Nephro',
        'نورولوژی شیمیایی': '1403Q2 Chem Neuro',
        'کاردیو متابولیک': '1403Q2 Cardio-Metab',
        'غدد': '1403Q2 Endo',
        'مکمل': '1403Q2 Suppl',
        'خود ایمنی و پوکی استخوان': '1403Q2 Autoimm & Osteo',
        'ناباروری': '1403Q2 Infert',
        'بیماری های عفونی و واکسن': '1403Q2 ID & Vacc',
        'چشم': '1403Q2 Ophth',
        'درمو کازمتیک': '1403Q2 Dermo',
    }
    forecast_total_df_mod = pd.DataFrame(columns=headers)
    for product in products:
        sale_df = sale_df_total[sale_df_total['product'] == product]
        sale_mean = sale_df['sales'][-12:].mean()
        forecast = forecast_total_df[forecast_total_df['product']==product]
        forecast = forecast.replace(forecast['forecast']<=0, sale_mean)
        forecast_total_df_mod = pd.concat([forecast_total_df_mod,forecast])

    pivot =forecast_total_df_mod[forecast_total_df_mod['date']>=140304].pivot_table(index=['product', 'dep'], columns='date', values='forecast').reset_index()
    pivot['file_name'] = pivot.dep.map(dep_dict)
    excel_manager = ExcelManager(directory="data/results/1403Q2")
    # Call the function to append rows to Excel
    excel_manager.append_rows_to_excel(pivot)
    # Apply table formatting and adjust column widths to all files in the directory
    excel_manager.apply_formatting_to_all_files()

