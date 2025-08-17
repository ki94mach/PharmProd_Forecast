import csv
import os
import pandas as pd
from pkg.excelmanager import ExcelManager
import numpy as np

def define_path(curr_qrt):
    forecasts = f'data/results/{curr_qrt}/{curr_qrt}_total_forecast.csv'
    return forecasts

def setup_forecast_file(forecasts, headers):
    if not os.path.exists(forecasts):
        with open(forecasts, "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)

def update_department_info(curr_qrt):
    dep_dict = {
        "انکولوژی بیولوژیک": 'Bio Onco',
        'انکولوژی شیمیایی': 'Chem Onco',
        'رسپیراتوری': 'Resp',
        'کانتراست مدیا': 'Contrast',
        'زیبایی': 'Beaut',
        'نورولوژی بیولوژیک': 'Bio Neuro',
        'انکولوژی اطفال': 'Ped Onco',
        'نفرولوژی': 'Nephro',
        'نورولوژی شیمیایی': 'Chem Neuro',
        'کاردیو متابولیک خوراکی': 'Oral Cardio-Metab',
        'کاردیو متابولیک تزریقی': 'Inj Cardio-Metab',
        'غدد': 'Endo',
        'مکمل': 'Suppl',
        'خود ایمنی و پوکی استخوان': 'Autoimm & Osteo',
        'ناباروری': 'Infert',
        'بیماری های عفونی و واکسن': 'ID & Vacc',
        'چشم': 'Ophth',
        'درمو کازمتیک': 'Dermo',
    }
    # Add the qaurter information for each department name for later references
    updated_dep_dict = {key: curr_qrt + '_' + value for key, value in dep_dict.items()}
    return updated_dep_dict

def replace_negative_sales(df):
    df.sort_values(by=['product', 'date'], inplace=True)
    df.date += 62100
    df.loc[:, 'date'] = pd.to_datetime(df['date'], 
                                        format='%Y%m',
                                        errors='coerce')
    for product in df['product'].unique():
        product_df = df[df['product'] == product].copy()
        product_df.sort_values(by=['product', 'date'], inplace=True)
        for i in range(len(product_df)):
            if product_df.iloc[i]['sales'] < 0:
                # Get the past 12 months
                past_12_months = product_df.iloc[max(0, i-12):i]['sales']
                if len(past_12_months) > 0:
                    avg_sales = past_12_months[past_12_months >= 0].mean()
                    if not np.isnan(avg_sales):
                        df.loc[product_df.index[i], 'sales'] = int(avg_sales)
    df['date'] = df['date'].apply(lambda x: int(str(x).replace('-', '')[:6]) - 62100)
    return df[df['type'] == 'forecast']

def pivot_and_format_data(forecast_total_df_mod, updated_dep_dict, forecast_start_date):
    pivot = forecast_total_df_mod[
        forecast_total_df_mod['date'] >= forecast_start_date].pivot_table(
            index=['product_fa', 'dep', 'provider', 'status'],
            columns='date',
            values='sales',
        ).reset_index()
    pivot['file_name'] = pivot.dep.map(updated_dep_dict)
    return pivot

def manage_excel(pivot, directory, curr_qrt):
    excel_manager = ExcelManager(directory=directory)
    excel_manager.append_rows_to_excel(pivot)
    excel_manager.apply_formatting_to_all_files()
    excel_manager.summary_export(pivot, curr_qrt)
