"""Shamsi (Persian calendar) month arithmetic helpers."""
from __future__ import annotations

from typing import NamedTuple, Optional


class ShamsiYmd(NamedTuple):
    year: int
    month: int
    day: int

    @property
    def yyyymm(self) -> int:
        return self.year * 100 + self.month

    @property
    def yyyymmdd(self) -> int:
        return self.year * 10000 + self.month * 100 + self.day


def parse_shamsi_ymd(raw) -> Optional[ShamsiYmd]:
    """Parse a Shamsi ``YYYY/MM/DD`` (or ``YYYY-MM-DD``) string.

    Returns ``None`` for missing, placeholder (``0000/00/00``), or invalid
    month/day. Does not use YYYYMM integer subtraction or Gregorian parsers.
    """
    if raw is None:
        return None
    if isinstance(raw, float) and raw != raw:  # NaN
        return None
    text = str(raw).strip()
    if not text:
        return None
    for sep in ("/", "-", "."):
        if sep in text:
            parts = text.split(sep)
            break
    else:
        return None
    if len(parts) != 3:
        return None
    try:
        year = int(parts[0])
        month = int(parts[1])
        day = int(parts[2])
    except ValueError:
        return None
    if year < 1300 or year > 1599:
        return None
    if month < 1 or month > 12:
        return None
    if day < 1 or day > 31:
        return None
    return ShamsiYmd(year, month, day)


def shamsi_add_months(ym: int, n: int) -> int:
    y, m = divmod(int(ym), 100)
    idx = y * 12 + (m - 1) + n
    y2, m0 = divmod(idx, 12)
    return y2 * 100 + (m0 + 1)


def shamsi_month_diff(ym_a: int, ym_b: int) -> int:
    """Months from ym_b to ym_a (ym_a - ym_b)."""
    ya, ma = divmod(int(ym_a), 100)
    yb, mb = divmod(int(ym_b), 100)
    return (ya * 12 + ma) - (yb * 12 + mb)


def last_complete_6m(origin_ym: int) -> tuple[int, int]:
    """Inclusive Shamsi window: origin-6 .. origin-1."""
    end = shamsi_add_months(origin_ym, -1)
    start = shamsi_add_months(origin_ym, -6)
    return start, end
