import logging
from pkg.sales_forecasting import SalesForecasting
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
if __name__ == "__main__":
    # Define the current quarter like: 1403Q1
    curr_qrt = '1404Q2'
    # Define the starting forecast YearMonth
    forecast_start_date = 140404
    sales_forecasting = SalesForecasting(curr_qrt)
    sales_forecasting.run(forecast_start_date)
