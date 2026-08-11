"""Shared constants for SQL query modules."""

TARGET_GENERIC_EN = (
    "Aryotrust",
    "Alvoxal",
    "Alvotere",
    "Beraksurf",
    "Leudribine",
    "Aryoseven",
    "Cinnal-f Pen",
    "Zytux",
    "Xybrone",
    "Cinnopar",
    "Dactoma",
    "Tysuna",
    "Paglino",
    "Rolima",
    "Cinnora",
    "Cinnatropin",
    "Lyratan",
    "Cinnomer",
    "FolicoGen",
    "Melitide",
    "Pectuna",
    "Ricanza",
    "Xacrel",
    "Alvopem",
    "Zakaria",
    "Recigen",
    "Cinnal-f",
    "Altebrel",
    "Dalfyra",
    "Clastoz",
    "Alvopax",
)


def sql_in_list(values: tuple[str, ...] | list[str]) -> str:
    """Format values as a SQL IN-list literal, e.g. "'A', 'B'."""
    return ", ".join(f"'{name}'" for name in values)


GENERIC_EN_IN = sql_in_list(TARGET_GENERIC_EN)
