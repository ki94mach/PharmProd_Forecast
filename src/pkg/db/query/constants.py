"""Shared constants for SQL query modules."""

TARGET_GENERIC_EN = (
    "Aryotrust",
    "Tyalia",
    "Alvocade",
    "Beraksurf",
    "Excilia Hair",
    "Canvert",
    "Cinnal-f Pen",
    "Suprotac",
    "Xybrone",
    "Cinnopar",
    "Dactoma",
    "Tysuna",
    "Paglino",
    "Lunaphil",
    "Fiorage",
    "Rolima",
    "Xabano",
    "Cinnora",
    "Cinnatropin",
    "RenalFact",
    "Lyratan",
    "Maciza",
    "Cinnomer",
    "FolicoGen",
    "Nanojade",
    "Melitide",
    "Pectuna",
    "Ricanza",
    "Xacrel",
    "Solavis",
    "Xetarem",
    "Zakaria",
)


def sql_in_list(values: tuple[str, ...] | list[str]) -> str:
    """Format values as a SQL IN-list literal, e.g. "'A', 'B'."""
    return ", ".join(f"'{name}'" for name in values)


GENERIC_EN_IN = sql_in_list(TARGET_GENERIC_EN)
