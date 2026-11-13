"""Excel output with clean formatting."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

DARK = PatternFill("solid", fgColor="0B1016")
STRONG = PatternFill("solid", fgColor="D9F99D")
CONF = PatternFill("solid", fgColor="ECFCCB")
VALUE = PatternFill("solid", fgColor="FEF3C7")
SLIP = PatternFill("solid", fgColor="E0F2FE")
WHITE = Font(color="FFFFFF", bold=True)
BOLD = Font(bold=True)
THIN = Border(bottom=Side(style="thin", color="E5E7EB"))

PCT_HINTS = ("%", "chance", "prob", "edge", "implied", "rate", "roi", "accuracy", "share")


def _style(ws, df: pd.DataFrame, widths: dict | None = None):
    for c in ws[1]:
        c.fill, c.font = DARK, WHITE
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[1].height = 30
    ws.freeze_panes = "A2"
    if len(df):
        ws.auto_filter.ref = ws.dimensions
    for j, col in enumerate(df.columns, 1):
        letter = get_column_letter(j)
        w = (widths or {}).get(col) or min(max(len(str(col)) + 2, *(len(str(v)) + 2 for v in df[col].head(200)), 10), 44)
        ws.column_dimensions[letter].width = w
        low = str(col).lower()
        if any(h in low for h in PCT_HINTS):
            for cell in ws[letter][1:]:
                cell.number_format = "0.0%"
        elif "odds" in low or low in ("stake",):
            for cell in ws[letter][1:]:
                cell.number_format = "0.00"
        for cell in ws[letter][1:]:
            cell.border = THIN


def _fill_rows(ws, df: pd.DataFrame, col: str, fills: dict):
    if col not in df.columns:
        return
    j = list(df.columns).index(col) + 1
    for i in range(2, len(df) + 2):
        f = fills.get(ws.cell(i, j).value)
        if f:
            for c in ws[i]:
                c.fill = f


def write_predictions(path: Path, sheets: dict[str, pd.DataFrame], notes: list[str]):
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            _style(ws, df)
            if name in ("All", "All predictions"):
                _fill_rows(ws, df, "Strength", {"BANGER": STRONG, "STRONG": STRONG, "GOOD": CONF})
            if "acca" in name or name == "My picks":
                _fill_rows(ws, df, "Row", {"SLIP": SLIP})
            if name == "Value bets":
                for row in ws.iter_rows(min_row=2):
                    for c in row:
                        c.fill = VALUE
        ws = xw.book.create_sheet("How to read")
        ws.column_dimensions["A"].width = 120
        for i, line in enumerate(notes, 1):
            c = ws.cell(i, 1, line)
            c.alignment = Alignment(wrap_text=True, vertical="top")
            if i == 1:
                c.font = Font(bold=True, size=14)
            elif line.isupper():
                c.font = BOLD


def write_table(path: Path, sheets: dict[str, pd.DataFrame]):
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name, index=False)
            _style(xw.sheets[name], df)
