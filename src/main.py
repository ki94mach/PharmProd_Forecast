import logging
from pkg.sales_forecasting import SalesForecasting
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
if __name__ == "__main__":
    # Define the current quarter like: 1403Q1
    curr_qrt = '1403Q2'
    sales_forecasting = SalesForecasting(curr_qrt)
    sales_forecasting.run()

    # # The monthly sales csv should contain
    # # "product", "date", "dep", "sale" column
    # # Name format: 1403Q1_monthly_sales.csv
    # sale = f'data/sales/{curr_qrt}/{curr_qrt}_monthly_sales.csv'
    # forecasts = f'data/results/{curr_qrt}/{curr_qrt}_total_forecast.csv'
    # headers=['product', 'date', 'forecast', 'model','dep']
    # if not os.path.exists(forecasts):
    #     with open(forecasts, "w", newline='') as file:
    #         writer = csv.writer(file)
    #         writer.writerow(headers)
    # sale_df_total = pd.read_csv(sale)
    # forecast_df = pd.read_csv(forecasts)

    # products_fr = pd.unique(forecast_df['product'])
    # products = pd.unique(sale_df_total['product'])
    # sale_df_total.date += 62100
    # for product in products:
    #     if product not in products_fr:
    #         sale_df = sale_df_total[sale_df_total['product'] == product]
    #         prod_fr = SalesForecast(product, sale_df, forecasts)
    #         prod_fr.preprocess_data()
    #         prod_fr.model_selection()
    #         prod_fr.predict()
    #         prod_fr.redistribute_smoothing()
    #         prod_fr.save_csv()
    #     else:
    #         continue
    # forecast_total_df = pd.read_csv(forecasts)
    # dep_dict = {
    #     "انکولوژی بیولوژیک": 'Bio Onco',
    #     'انکولوژی شیمیایی': 'Chem Onco',
    #     'رسپیراتوری': 'Resp',
    #     'کانتراست مدیا': 'Contrast',
    #     'زیبایی': 'Beaut',
    #     'نورولوژی بیولوژیک': 'Bio Neuro',
    #     'انکولوژی اطفال': 'Ped Onco',
    #     'نفرولوژی': 'Nephro',
    #     'نورولوژی شیمیایی': 'Chem Neuro',
    #     'کاردیو متابولیک': 'Cardio-Metab',
    #     'غدد': 'Endo',
    #     'مکمل': 'Suppl',
    #     'خود ایمنی و پوکی استخوان': 'Autoimm & Osteo',
    #     'ناباروری': 'Infert',
    #     'بیماری های عفونی و واکسن': 'ID & Vacc',
    #     'چشم': 'Ophth',
    #     'درمو کازمتیک': 'Dermo',
    # }
    # # Add the qaurter information for each department name for later references
    # updated_dep_dict = {key: curr_qrt + '_' + value for key, value in dep_dict.items()}

    # forecast_total_df_mod = pd.DataFrame()

    # def replace_negative_sales(df):
    #     df.sort_values(by=['product', 'date'], inplace=True)
    #     df.date += 62100
    #     df.loc[:, 'date'] = pd.to_datetime(df['date'], 
    #                                         format='%Y%m',
    #                                         errors='coerce')
    #     for product in df['product'].unique():
    #         product_df = df[df['product'] == product].copy()
    #         product_df.sort_values(by=['product', 'date'], inplace=True)
    #         for i in range(len(product_df)):
    #             if product_df.iloc[i]['sales'] < 0:
    #                 # Get the past 12 months
    #                 past_12_months = product_df.iloc[max(0, i-12):i]['sales']
    #                 if len(past_12_months) > 0:
    #                     avg_sales = past_12_months[past_12_months >= 0].mean()
    #                     if not np.isnan(avg_sales):
    #                         df.loc[product_df.index[i], 'sales'] = int(avg_sales)
    #         date = []
    #     for i in range(len(df)):
    #         try:
    #             date_i = int(str(df.iloc[i]['date']).replace('-', '')[:6])-62100
    #         except ValueError:
    #             date_i = 0
    #         date.append(date_i)
    #     df.loc[: ,'date'] = date
    #     return df[df['type'] == 'forecast']
    
    # forecast_df['sales'] = forecast_df.loc[:, 'forecast']
    # forecast_df['type'] = 'forecast'
    # sale_df_total['type'] = 'actual'
    # temp = pd.concat([sale_df_total, forecast_df])
    # forecast_total_df_mod = replace_negative_sales(temp)

    # # for product in products:
    # #     sale_df = sale_df_total[sale_df_total['product'] == product]
    # #     sale_mean = int(sale_df['sales'][-12:].mean())
    # #     forecast = forecast_total_df[forecast_total_df['product']==product]
    # #     forecast.loc[forecast['forecast'] <= 0, 'forecast'] = sale_mean
    # #     forecast_total_df_mod = pd.concat([forecast_total_df_mod, forecast])

    # pivot =forecast_total_df_mod[
    #     forecast_total_df_mod['date'] >= 140304].pivot_table(
    #         index=['product fa', 'dep', 'provider', 'status'],
    #         columns='date',
    #         values='sales',
    #         ).reset_index()
    
    # pivot['file_name'] = pivot.dep.map(updated_dep_dict)
    # excel_manager = ExcelManager(directory="data/results/1403Q1")
    # # Call the function to append rows to Excel
    # excel_manager.append_rows_to_excel(pivot)
    # # Apply table formatting and adjust column widths to all files in the directory
    # excel_manager.apply_formatting_to_all_files()

