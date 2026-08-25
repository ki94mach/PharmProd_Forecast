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


def parse_shamsi_quarter(quarter: str) -> tuple[int, int]:
    """Parse ``YYYYQn`` (e.g. ``1405Q1``) into ``(year, quarter)``."""
    text = str(quarter).strip().upper()
    if len(text) < 6 or "Q" not in text:
        raise ValueError(f"invalid Shamsi quarter label: {quarter!r}")
    year_s, q_s = text.split("Q", 1)
    try:
        year = int(year_s)
        q = int(q_s)
    except ValueError as exc:
        raise ValueError(f"invalid Shamsi quarter label: {quarter!r}") from exc
    if year < 1300 or year > 1599:
        raise ValueError(f"Shamsi year out of range in quarter: {quarter!r}")
    if q < 1 or q > 4:
        raise ValueError(f"quarter must be 1..4, got {quarter!r}")
    return year, q


def format_shamsi_quarter(year: int, quarter: int) -> str:
    """Format ``(year, quarter)`` as ``YYYYQn``."""
    if quarter < 1 or quarter > 4:
        raise ValueError(f"quarter must be 1..4, got {quarter}")
    return f"{int(year)}Q{int(quarter)}"


def origin_from_quarter(quarter: str) -> int:
    """Forecast origin for a Shamsi quarter: **first month of that quarter**.

    This is the inverse of :func:`quarter_from_origin`:

    - Q1 → month 01
    - Q2 → month 04
    - Q3 → month 07
    - Q4 → month 10

    Example: ``1405Q1`` → ``140501``, ``1404Q2`` → ``140404``.
    """
    year, q = parse_shamsi_quarter(quarter)
    month = (q - 1) * 3 + 1
    return year * 100 + month


def quarter_from_origin(forecast_origin: int) -> str:
    """Shamsi quarter label for a YYYYMM origin, e.g. ``140501`` → ``1405Q1``."""
    ym = int(forecast_origin)
    year, month = divmod(ym, 100)
    if month < 1 or month > 12:
        raise ValueError(f"invalid Shamsi YYYYMM for quarter: {forecast_origin!r}")
    quarter = (month - 1) // 3 + 1
    return format_shamsi_quarter(year, quarter)


def iter_shamsi_quarters(start_quarter: str, end_quarter: str) -> list[str]:
    """Inclusive contiguous Shamsi quarter sequence from ``start`` through ``end``."""
    y0, q0 = parse_shamsi_quarter(start_quarter)
    y1, q1 = parse_shamsi_quarter(end_quarter)
    start_idx = y0 * 4 + (q0 - 1)
    end_idx = y1 * 4 + (q1 - 1)
    if end_idx < start_idx:
        raise ValueError(
            f"end quarter {end_quarter!r} is before start {start_quarter!r}"
        )
    out: list[str] = []
    for idx in range(start_idx, end_idx + 1):
        year, q0_idx = divmod(idx, 4)
        out.append(format_shamsi_quarter(year, q0_idx + 1))
    return out


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
