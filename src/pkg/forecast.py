import logging
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import pmdarima as pm
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error
from prophet import Prophet
import matplotlib.dates as mdates
from matplotlib.ticker import AutoMinorLocator
import numpy as np
from gspread_formatting import *

class SalesForecast:
    """
    A class for performing sales forecasting on a specified product using a
    variety of models, including ARIMA, ETS, and Prophet. This class takes a
    DataFrame containing sales data, preprocesses it for modeling, selects
    the best-fitting model based on historical data, and then uses that model
    to forecast future sales.

    Attributes:
        product (str): The product name for which sales forecasting is
        performed. sale_df (pd.DataFrame): A DataFrame containing the sales
        data, expected to have at least 'sales' and 'date' columns.
        model (various): A placeholder for the fitted forecasting model,
        its type depends on the best model selected.
        forecast (np.array or pd.Series): The forecasted sales values.
        best_model_type (str): The type of the best-performing model based on 
        RMSE.
    """
    def __init__(self, product: str, sale_df: pd.DataFrame, output: str):
        """
        Initializes the SalesForecast class of product of interest.

        Parameters:

        - product (str): The product name to filter and forecast sales for.
        - sale_df (pd.DataFrame): Sales data of the product with "sales" and
        "date" columns
        - output (str): csv results path file
        """
        self.product = product
        self.sale_df = sale_df
        self.model = None
        self.forecast = None
        self.best_model_type = None 
        self.output = output
        self.dep = self.sale_df['dep'].unique()[-1]
        self.product_fa = self.sale_df['product_fa'].unique()[-1]
        self.provider = self.sale_df['provider'].unique()[-1]
        boxq_ser = pd.Series(self.sale_df['boxq'])
        boxq_ser.dropna(inplace=True) 
        self.status = "بسته" if boxq_ser.iat[-1] == 1 else "عدد"
        logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

    def preprocess_data(self):
        """
        Preprocesses the input DataFrame to prepare it for forecasting. This
        includes converting
        Jalali dates to Gregorian, handling missing values, converting sales
        data to a numeric format,
        and filtering the data to include only the sales after the first
        non-zero sale. It also
        prepares separate DataFrames for use with the Prophet model.

        No parameters. Modifies the instance's sale_df and prophet_df
        attributes in place.
        """
        self.sale_df.loc[:, 'date'] = pd.to_datetime(self.sale_df['date'], 
                                                     format='%Y%m')
        self.sale_df.loc[:, 'sales'] = pd.to_numeric(self.sale_df['sales'], 
                                                     errors='coerce')
        self.sale_df.loc[:, 'sales'] = self.sale_df['sales'].fillna(value=0)
        self.sale_df = self.sale_df.drop_duplicates(subset='date')
        first_sale = self.sale_df[self.sale_df['sales'] > 5].index.min()
        if first_sale is not None:
            self.sale_df = self.sale_df.loc[first_sale:].reset_index(drop=True)
        self.sale_df.set_index('date', inplace=True)
        self.sale_df = self.sale_df.asfreq('MS').fillna(value=0)
        self.sale_df['product'] = self.sale_df['product'].replace(0, self.product)
        self.sale_df['dep'] = self.sale_df['dep'].replace(0, self.dep)
        self.sale_series = self.sale_df['sales'][:-1]
        self.sale_df.reset_index(inplace=True)
        self.prophet_df = self.sale_df[['date', 'sales']][:-1].rename(
            columns={'date': 'ds', 'sales': 'y'})
   
    def _rolling_forecast(self, model_type: str) -> tuple:
        """
        Performs a rolling forecast to evaluate the predictive accuracy of
        specified forecasting models.

        Parameters:
            model_type (str): The type of forecasting model to use. Options
            are "ARIMA", "ETS", or "Prophet".

        Returns:
            tuple: A tuple containing two lists, the first with the predicted
            values and the second with the actual values used for testing.
        """

        h = 1
        size = int(len(self.sale_series)*0.8)
        self.predictions = []
        self.actuals = []
        model = None
        if (len(self.sale_series) - size) > 2:
            for t in range(size, len(self.sale_series) - h + 1):
                train, test = self.sale_series[:t].copy(), self.sale_series[t:t+h].copy()
                prophet_train, prophet_test = self.prophet_df.iloc[:t].copy(), self.prophet_df.iloc[t:t+h].copy()
                if prophet_train.empty or prophet_test.empty or train.empty:
                    continue
                max_y = prophet_train['y'].max() * 1.2
                min_y = prophet_train['y'].min() * 0.8
                prophet_train = prophet_train.reset_index(drop=True)
                prophet_train['cap'] = max_y
                prophet_train['floor'] = min_y

                # Fit the specified model
                if model_type == 'ARIMA':
                    if len(train) > 24:
                        try:
                            model = pm.auto_arima(train, seasonal=True, m=12,
                                                max_order=None, max_p=6, max_q=6, max_d=2,
                                                max_P=4, max_Q=4, max_D=1,
                                                stepwise=True, suppress_warnings=True, error_action="ignore",
                                                )
                        except ValueError:
                            model = pm.auto_arima(train, seasonal=True, m=12,
                                                max_order=None, max_p=6, max_q=6, max_d=2,
                                                max_P=4, max_Q=4, max_D=1,
                                                stepwise=True, suppress_warnings=True, error_action="ignore",
                                                seasonal_test='ch',
                                                )
                    else:
                        model = pm.auto_arima(train, 
                                            max_order=None, max_p=6, max_q=6, max_d=2,
                                            max_P=4, max_Q=4, max_D=2,
                                            stepwise=True, suppress_warnings=True, error_action="ignore"
                                            )
                    prediction = model.predict(n_periods=h)

                elif model_type == 'ETS':
                    if len(train) > 24:
                        model = ETSModel(train, seasonal_periods=12, error="add", trend="add").fit()
                    else:
                        model = ETSModel(
                            train, error="add", trend="add").fit()
                    prediction = model.forecast(steps=h)                  
                elif model_type == 'Prophet':

                    if len(prophet_train) > 24:
                        model = Prophet(growth='logistic',
                                    yearly_seasonality=True,
                                    changepoint_prior_scale=0.05,
                                    ).fit(prophet_train)
                    else:
                        model = Prophet(changepoint_prior_scale=0.05).fit(prophet_train)
                    future = model.make_future_dataframe(
                        periods=h, freq='M')
                    future['cap'] = max_y
                    future ['floor']= min_y
                    forecast = model.predict(future)
                    prediction = forecast['yhat'].values[-h:]
                
                self.predictions.extend(prediction)
                
                if model_type in ['ARIMA', 'ETS']:
                    self.actuals.extend(test.values)
                elif model_type == 'Prophet':
                    self.actuals.extend(prophet_test['y'].values)
                
        else:
            train, test = self.sale_series[:size].copy(), self.sale_series[size:].copy()
            prophet_train, prophet_test = self.prophet_df.iloc[:size].copy(), self.prophet_df.iloc[size:].copy()
            max_y = prophet_train['y'].max() * 1.2
            min_y = prophet_train['y'].min() * 0.8

            prophet_train.loc[:, 'cap'] = max_y
            prophet_train.loc[:, 'floor'] = min_y 
            if model_type == 'ARIMA':
                model = pm.auto_arima(train, 
                    max_order=None, max_p=6, max_q=6, max_d=2,
                    max_P=4, max_Q=4, max_D=2,
                    stepwise=True, suppress_warnings=True, error_action="ignore"
                    )
                prediction = model.predict(n_periods=len(test))

            elif model_type == 'ETS':
                model = ETSModel(train, error="add", trend="add").fit()
                prediction = model.forecast(steps=len(test))
            
            elif model_type == 'Prophet':
                model = Prophet(changepoint_prior_scale=0.1,).fit(prophet_train)
                future = model.make_future_dataframe(
                    periods=len(test), freq='M')
                future['cap'] = max_y
                future ['floor']= min_y
                forecast = model.predict(future)
                prediction = forecast['yhat'].values[-len(test):]

            self.predictions.extend(prediction)
            if model_type in ['ARIMA', 'ETS']:
                self.actuals.extend(test.values)
            elif model_type == 'Prophet':
                self.actuals.extend(prophet_test['y'].values)
        # print(self.predictions, self.actuals)
        return self.predictions, self.actuals

    def model_selection(self):
        """"
        Evaluates different forecasting models (ARIMA, ETS, Prophet) by
        performing a rolling forecast and computing the Root Mean Squared
        Error (RMSE) for each. The model with the lowest RMSE is
        selected as the best model for the forecasting task.

        No parameters. Updates the instance's best_model_type attribute with
        the name of the best-performing model.
        """
        arima_predictions, arima_actuals = self._rolling_forecast(model_type='ARIMA')
        ets_predictions, ets_actuals = self._rolling_forecast(model_type='ETS')
        prophet_predictions, prophet_actuals = self._rolling_forecast(model_type='Prophet')

        # print(ets_actuals, ets_predictions)
        # print(arima_actuals, arima_predictions)
        # print(prophet_actuals, prophet_predictions)

        # Calculate RMSE  
        if not arima_predictions or not arima_actuals:
            arima_rmse = None
        else:
            arima_rmse = sqrt(mean_squared_error(arima_actuals, arima_predictions))
        ets_rmse = sqrt(mean_squared_error(ets_actuals, ets_predictions))
        prophet_rmse = sqrt(mean_squared_error(prophet_actuals, prophet_predictions))

        # print(f"ARIMA RMSE: {arima_rmse}")
        # print(f"ETS RMSE: {ets_rmse}")
        # print(f"Prophet RMSE: {prophet_rmse}")
        rmse_values = {
            'ARIMA': arima_rmse,
            'ETS': ets_rmse,
            'Prophet': prophet_rmse,
            }
        valid_rmse_values = {
            k: v for k, v in rmse_values.items() if v is not None}
        self.best_model_type = min(
            valid_rmse_values, key=valid_rmse_values.get)
        
    def predict(self):
        """
        Generates a sales forecast using the best-performing model determined
        by the model_selection method.
        This involves fitting the selected model to the entire dataset and
        predicting future sales.

        No parameters. Updates the instance's forecast, lower_pi, and upper_pi
        attributes with the forecasted sales,
        lower prediction interval, and upper prediction interval, respectively.
        """
        if self.best_model_type == 'ARIMA':
            if len(self.sale_series) > 24:
                try:
                    final_model = pm.auto_arima(
                        self.sale_series, seasonal=True, m=12, stepwise=True,
                        max_order=None, max_p=6, max_q=6, max_d=2,
                        max_P=4, max_Q=4, max_D=2, error_action="ignore")
                except ValueError:
                    final_model = pm.auto_arima(
                        self.sale_series, seasonal=True, m=12, stepwise=True,
                        max_order=None, max_p=6, max_q=6, max_d=2,
                        max_P=4, max_Q=4, max_D=2, error_action="ignore",
                        seasonal_test='ch')
            else:
                final_model = pm.auto_arima(
                    self.sale_series, stepwise=True,
                    max_order=None, max_p=6, max_q=6, max_d=2,
                    max_P=4, max_Q=4, max_D=2, error_action="ignore")
            forecast_all, pred_intv = final_model.predict(
                n_periods=16, return_conf_int=True)
            self.forecast = forecast_all[-15:]
            self.lower_pi = pred_intv[-15:, 0]
            self.upper_pi = pred_intv[-15:, 1]
            
        elif self.best_model_type == 'ETS':
            if len(self.sale_series)>24:
                final_model = ETSModel(
                    self.sale_series,
                    error="add",
                    trend="add",
                    seasonal="add",
                    seasonal_periods=12).fit()
            else:
                final_model = ETSModel(
                    self.sale_series,
                    error="add",
                    trend="add",).fit()
            start_period = len(self.sale_series)+1
            forecast_obj = final_model.get_prediction(
                start=start_period, end=start_period+14)
            forecast_summary = forecast_obj.summary_frame(alpha=0.05)
            self.forecast = forecast_summary['mean']
            self.lower_pi = forecast_summary['pi_lower']
            self.upper_pi = forecast_summary['pi_upper']

        elif self.best_model_type == 'Prophet':
            max_y = self.prophet_df['y'].max() * 1.2
            min_y = self.prophet_df['y'].min() * 0.8
            self.prophet_df.loc[:, 'cap'] = max_y
            self.prophet_df.loc[:, 'floor'] = min_y
            if len(self.prophet_df)>24:
                prophet_model = Prophet(growth='logistic',
                                        yearly_seasonality=True,
                                        changepoint_prior_scale=0.05).fit(self.prophet_df)
            else:
                prophet_model = Prophet(growth='logistic',
                                        changepoint_prior_scale=0.05,).fit(self.prophet_df)
            future = prophet_model.make_future_dataframe(
                periods=len(self.prophet_df)+16, freq='M')
            future['cap'] = max_y
            future ['floor']= min_y
            forecast_df = prophet_model.predict(future)[-15:]
            # print(future)
            self.forecast = forecast_df['yhat'] * 0.8
            self.lower_pi = forecast_df['yhat_lower'] * 0.8
            self.upper_pi = forecast_df['yhat_upper'] * 0.8

    def redistribute_smoothing(self):
        # Copy the column to avoid modifying the original data
        data = np.array(self.forecast.copy())
        
        n = len(data)
        
        # Calculate adjustments needed
        adjustments = np.zeros(n)
        smoothed = np.zeros(n)
        # smoothed[0] = data[0]
        for i in range(1, n+1):
            if i % 3 == 0:
                # Calculate quarter's mean
                q_mean = data[(i-3): i].mean()
                # Calculate difference from quarter mean and add halve of it
                # to smooth the data
                for j in range(i - 3, i):
                    adjustments[j] = (data[j] - q_mean) * 0.7
                    smoothed[j] = data[j] - adjustments[j]
                q_adjustments = adjustments[(i-3): i].sum()
                smoothed[(i-3): i] += q_adjustments
            
            # if i > 0 and i < len(data) - 1:
            #     if data[i] > data[i-1] and data[i] > data[i+1]:  # Peak
            #         window = base_window + 2
            #     elif data[i] < data[i-1] and data[i] < data[i+1]:  # Trough
            #         window = base_window + 2
            #     else:
            #         window = base_window
            # else:
            #     window = base_window
            # # Apply smoothing
            # window_data = data[max(i-window//2, 0):min(i+window//2+1,
            # len(data))]
            # smoothed[i] = window_data.mean()
        # # Ensure total adjustment is zero
        # adjustments = smoothed - data
        # total_adjustment = adjustments.sum()
        # smoothed += total_adjustment / n
        # # applying difference adjustments in each quarter
        # for i in range(n):
        #     if i % 3 == 0:
        #         total_adjustment = adjustments[(i-2):(i+1)].sum()
        #         if total_adjustment != 0:
        #             smoothed[(i-2):(i+1)] += total_adjustment / 3

        # Apply adjustments
        self.forecast = smoothed

    def replace_negative_sales(self, series):
        series.sort_index(inplace=True)
        for i in range(len(series)):
            if series.iloc[i] < 0:
                # Get the past 12 months
                past_12_months = series.iloc[max(0, i-12):i]
                if len(past_12_months) > 0:
                    avg_sales = past_12_months[past_12_months >= 0].mean()
                    if not np.isnan(avg_sales):
                        series.iloc[i] = int(avg_sales)
        return series[-15:].values
    
    def save_csv(self):
        """
        Converts back date to Jalali and saves the forecast mean in a csv file

        """
        # Converting back to Jalali YearMonth format
        forecast_steps = 15
        self.forecast_index = pd.date_range(
            start=self.sale_series.index.max()+pd.DateOffset(months=2
                                                             ),periods=forecast_steps, freq='MS')
        forecast_date = []
        for i in self.forecast_index:
            forecast_date.append(int(str(i).replace("-", "")[:6])-62100)
        
        forecast_series = pd.Series(self.forecast, index=self.forecast_index)
        
        self.forecast_df = pd.DataFrame(columns=[
            'product', 'product_fa', 'date',
            'provider', 'model', 'dep', 'status', 'forecast'
            ])
        self.forecast_df.date = forecast_date
        self.forecast_df['product'] = self.product
        self.forecast_df['product_fa'] = self.product_fa
        self.forecast_df['provider'] = self.provider
        self.forecast_df['model'] = self.best_model_type
        self.forecast_df['dep'] = self.dep
        self.forecast_df['status'] = self.status
        self.forecast_df.forecast = self.replace_negative_sales(pd.concat([forecast_series, self.sale_series]))
        self.forecast_df['forecast'] = self.forecast_df['forecast'].round()
        self.forecast_df.to_csv(self.output, index=False, mode="a", header=False, encoding='utf-8-sig')
        print(f"{self.product} forecasting is done!")
        # return self.forecast_df

    def plot(self):
        """
        Plots the historical sales data along with the forecasted sales and prediction intervals for the next 15 periods.

        Uses matplotlib to generate a line plot of the historical and forecasted sales, including a shaded area representing
        the prediction interval. The plot is labeled with the dates and sales values, and includes a legend.

        No parameters. This method produces a plot and does not return any values.
        """
        self.forecast_index = pd.date_range(start=self.sale_series.index.max()+pd.DateOffset(months=2), periods=15, freq='MS')
        plt.figure(figsize=(14, 7))
        # Historical Sales
        plt.plot(self.sale_series.index, self.sale_series, label='Historical Sales', color='black')
        # Forecast
        plt.plot(self.forecast_index, self.forecast, color='blue', label='Forecast')
        plt.fill_between(self.forecast_index, self.lower_pi, self.upper_pi, color='blue', alpha=0.1, label='Prediction Interval')

        # Major ticks every year, minor ticks every month
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        # Set minor tick formatter (optional, depending on your preference for display)
        # plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter('%m'))
        
        # AutoMinorLocator for y-axis
        plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
    
        plt.grid(which='both', linestyle='--', linewidth=0.5)
        plt.title(f'{self.best_model_type} Sales Forecast :{self.product}')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.show()
