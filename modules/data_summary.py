"""
data_summary.py
Computes dataset-level and column-level summary statistics.
"""

from __future__ import annotations

import pandas as pd

from modules.utils import format_bytes, get_numeric_columns, get_categorical_columns, get_datetime_columns


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """Return a dict of high-level dataset statistics."""
    n_rows, n_cols = df.shape
    n_missing = int(df.isna().sum().sum())
    n_duplicates = int(df.duplicated().sum())

    try:
        mem_bytes = df.memory_usage(deep=True).sum()
        mem_display = format_bytes(mem_bytes)
    except Exception:
        mem_display = "N/A"

    n_numeric = len(get_numeric_columns(df))
    n_categorical = len(get_categorical_columns(df))
    n_datetime = len(get_datetime_columns(df))

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_missing": n_missing,
        "n_duplicates": n_duplicates,
        "memory_usage": mem_display,
        "n_numeric": n_numeric,
        "n_object": n_categorical,
        "n_datetime": n_datetime,
    }


def get_column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-column summary table."""
    rows = []
    n_rows = len(df)
    for col in df.columns:
        series = df[col]
        non_null = int(series.notna().sum())
        missing = int(series.isna().sum())
        missing_pct = round((missing / n_rows) * 100, 2) if n_rows else 0.0
        unique = int(series.nunique(dropna=True))
        rows.append({
            "Column": col,
            "Data Type": str(series.dtype),
            "Non-Null Count": non_null,
            "Missing Count": missing,
            "Missing %": missing_pct,
            "Unique Values": unique,
        })
    return pd.DataFrame(rows)
