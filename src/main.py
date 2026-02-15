"""CLI for pharmaceutical sales time series forecasting."""
import argparse
import logging
import sys

from dotenv import load_dotenv
from pkg.sales_forecasting import SalesForecasting

load_dotenv()
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pharmaceutical sales time series forecast.",
        epilog="""
Examples:
  Full forecast:   python main.py --qrt 1405Q1 --start-date 140501
  Template only:   python main.py --qrt 1405Q1 --start-date 140501 --template
        """,
    )
    parser.add_argument("--qrt", required=True, help="Quarter (e.g. 1405Q1)")
    parser.add_argument("--start-date", required=True, help="Forecast start date Shamsi YYYYMM (e.g. 140501)")
    parser.add_argument("--template", action="store_true", help="Only generate output files with zero forecasts")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        start_date = int(args.start_date)
    except (TypeError, ValueError):
        print("Error: --start-date must be an integer (e.g. 140501).", file=sys.stderr)
        sys.exit(1)

    generate_forecasts = not args.template
    if generate_forecasts:
        print(f"Running full forecast (qrt={args.qrt}, start-date={start_date}).")
    else:
        print(f"Generating template only (qrt={args.qrt}, start-date={start_date}).")

    sales_forecasting = SalesForecasting(args.qrt)
    sales_forecasting.run(start_date, generate_forecasts=generate_forecasts)
    print("Done.")


if __name__ == "__main__":
    main()
