"""Shamsi (Persian calendar) month arithmetic helpers."""


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
