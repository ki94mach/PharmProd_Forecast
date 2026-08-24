"""CLI for pharmaceutical sales time series forecasting."""
import argparse
import logging
import sys

from dotenv import load_dotenv

load_dotenv()

from pkg.sales_forecasting import SalesForecasting

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pharmaceutical sales time series forecast.",
        epilog="""
Examples:
  Full forecast:   python main.py --qrt 1405Q1 --start-date 140501
  Template only:   python main.py --qrt 1405Q1 --start-date 140501 --template
  Basket vintage:  python main.py --qrt 1404Q1 --start-date 140312 --vintage --force
        """,
    )
    parser.add_argument("--qrt", required=True, help="Quarter (e.g. 1405Q1)")
    parser.add_argument("--start-date", required=True, help="Forecast start date Shamsi YYYYMM (e.g. 140501)")
    parser.add_argument("--template", action="store_true", help="Only generate output files with zero forecasts")
    parser.add_argument(
        "--vintage",
        action="store_true",
        help=(
            "As-of vintage: keep sales with date < --start-date, reset the CSV, "
            "and forecast the Dim.Product basket (not TARGET_GENERIC_EN)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="With --vintage: backup and overwrite an existing forecast CSV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        start_date = int(args.start_date)
    except (TypeError, ValueError):
        print("Error: --start-date must be an integer (e.g. 140501).", file=sys.stderr)
        sys.exit(1)

    if args.template and args.vintage:
        print("Error: --template and --vintage cannot be used together.", file=sys.stderr)
        sys.exit(1)
    if args.force and not args.vintage:
        print("Error: --force is only valid with --vintage.", file=sys.stderr)
        sys.exit(1)

    generate_forecasts = not args.template
    if args.vintage:
        print(
            f"Running basket vintage (qrt={args.qrt}, start-date={start_date}, "
            "sales date < start-date)."
        )
    elif generate_forecasts:
        print(f"Running full forecast (qrt={args.qrt}, start-date={start_date}).")
    else:
        print(f"Generating template only (qrt={args.qrt}, start-date={start_date}).")

    sales_forecasting = SalesForecasting(args.qrt)
    sales_forecasting.run(
        start_date,
        generate_forecasts=generate_forecasts,
        vintage=args.vintage,
        force=args.force,
    )
    print("Done.")


if __name__ == "__main__":
    main()
