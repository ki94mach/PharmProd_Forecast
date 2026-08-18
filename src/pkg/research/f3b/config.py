"""F3B Step 1 paths. Does not mutate F0 / F1 / F2 / F3A artifacts or the v1 freeze."""
from __future__ import annotations

from pathlib import Path

PRICE_SHEET_NAME = "جدول تغییر قیمت ها"
MAP_SHEET_NAME = "map"

SOURCE_PRODUCT_COL = "نام کالا"
PROVIDER_COL = "نام شرکت"
DISTRIBUTOR_PRICE_COL = "بهای فروش به پخش"
PHARMACY_PRICE_COL = "بهای فروش به داروخانه"
CONSUMER_PRICE_COL = "بهای مصرف کننده"
PACK_QTY_COL = "تعداد در بسته"
DATE_COL = "تاریخ"

MAP_SOURCE_COL = "نام محصول در تحویل به پخش"
MAP_TARGET_COL = "Dim Product"

PRICE_HISTORY_COLS = (
    "product_id",
    "product",
    "generic",
    "provider",
    "source_product_fa",
    "mapped_product_fa",
    "effective_date_raw",
    "effective_date",
    "effective_month",
    "distributor_price",
    "pharmacy_price",
    "consumer_price",
    "pack_quantity",
    "mapping_applied",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def src_dir() -> Path:
    return Path(__file__).resolve().parents[3]


def external_price_dir() -> Path:
    return src_dir() / "data" / "external" / "f3b_price"


def triple_price_xlsx() -> Path:
    return external_price_dir() / "فرم قیمت سه گانهsc-fr-008 (2).xlsx"


def product_map_xlsx() -> Path:
    return external_price_dir() / "Map Product-Delivery dis.xlsx"


def f3b_output_dir() -> Path:
    out = src_dir() / "data" / "results" / "f3b"
    out.mkdir(parents=True, exist_ok=True)
    return out


def f3b_source_dir() -> Path:
    out = f3b_output_dir() / "source"
    out.mkdir(parents=True, exist_ok=True)
    return out


def docs_dir() -> Path:
    return repo_root() / "docs"
