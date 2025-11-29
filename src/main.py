import logging
from pkg.sales_forecasting import SalesForecasting
from dotenv import load_dotenv
import os

load_dotenv()
FORECAST_START_DATE = int(os.getenv('FORECAST_START_DATE'))
CURR_QRT = os.getenv('CURR_QRT')
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

if __name__ == "__main__":
    sales_forecasting = SalesForecasting(CURR_QRT)
    sales_forecasting.run(FORECAST_START_DATE)
