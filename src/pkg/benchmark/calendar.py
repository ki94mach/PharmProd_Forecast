"""Shamsi (Persian calendar) month arithmetic helpers."""
from __future__ import annotations

from datetime import date, timedelta
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


# ---------------------------------------------------------------------------
# Jalali ↔ Gregorian (Khayam / jalaali algorithm, no external package)
# ---------------------------------------------------------------------------

def _jalali_to_gregorian(jy: int, jm: int, jd: int) -> date:
    """Convert a Jalali (Shamsi) date to Gregorian. Pure arithmetic, no packages."""
    jy2 = jy + 1595
    days = -355668 + (365 * jy2) + ((jy2 // 33) * 8 + (jy2 % 33 + 3) // 4) + jd
    if jm < 7:
        days += (jm - 1) * 31
    else:
        days += ((jm - 1) * 30) + 6
    gy = 400 * (days // 146097)
    days %= 146097
    if days > 36524:
        gy += 100 * ((days - 1) // 36524)
        days = (days - 1) % 36524
        if days >= 365:
            days += 1
    gy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        gy += (days - 1) // 365
        days = (days - 1) % 365
    gm = 0
    leap = (gy % 4 == 0 and gy % 100 != 0) or gy % 400 == 0
    g_d_m = [0, 31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    for i in range(1, 13):
        if days < g_d_m[i]:
            gm = i
            break
        days -= g_d_m[i]
    return date(gy, gm, days + 1)


def shamsi_month_start_gregorian(yyyymm: int) -> date:
    """Gregorian date of Shamsi ``YYYYMM`` month-start (day 1).

    Uses a pure-Python Khayam/jalaali algorithm; does not require any
    calendar conda package.
    """
    ym = int(yyyymm)
    year, month = divmod(ym, 100)
    if year < 1300 or year > 1599:
        raise ValueError(f"Shamsi year out of range: {yyyymm}")
    if month < 1 or month > 12:
        raise ValueError(f"Shamsi month out of range: {yyyymm}")
    return _jalali_to_gregorian(year, month, 1)
