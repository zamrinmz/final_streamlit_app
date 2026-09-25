"""
data_cleaner.py
All dataset transformation operations. Every function here is pure: it takes
a DataFrame (and parameters) and returns a NEW DataFrame plus a small stats
dict describing what changed. The caller (app.py) is responsible for writing
the result back into st.session_state, so nothing here mutates global state.
"""

from __future__ import annotations

from typing import List, Optional, Literal

import pandas as pd

from modules.utils import safe_to_numeric


# ---------------------------------------------------------------------------
# Missing rows
# ---------------------------------------------------------------------------
def preview_remove_missing_rows(df: pd.DataFrame, how: str = "any",
                                 subset: Optional[List[str]] = None) -> dict:
    """Compute how many rows WOULD be removed, without modifying df."""
    subset = subset if subset else None
    if how == "any":
        rows_kept = df.dropna(how="any", subset=subset)
    else:  # "all"
        rows_kept = df.dropna(how="all", subset=subset)
    rows_before = len(df)
    rows_after = len(rows_kept)
    return {
        "rows_before": rows_before,
        "rows_removed": rows_before - rows_after,
        "rows_after": rows_after,
    }


def remove_missing_rows(df: pd.DataFrame, how: str = "any",
                         subset: Optional[List[str]] = None) -> pd.DataFrame:
    subset = subset if subset else None
    if how == "any":
        return df.dropna(how="any", subset=subset).reset_index(drop=True)
    return df.dropna(how="all", subset=subset).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------
def get_duplicate_stats(df: pd.DataFrame, subset: Optional[List[str]] = None) -> dict:
    subset = subset if subset else None
    n_dupes = int(df.duplicated(subset=subset, keep=False).sum())
    n_unique = len(df) - int(df.duplicated(subset=subset, keep="first").sum())
    return {"n_duplicate_rows": n_dupes, "n_unique_rows": n_unique}


def preview_remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None,
                               keep: str = "first") -> dict:
    subset = subset if subset else None
    n_to_remove = int(df.duplicated(subset=subset, keep=keep).sum())
    return {
        "rows_before": len(df),
        "rows_removed": n_to_remove,
        "rows_after": len(df) - n_to_remove,
    }


def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None,
                       keep: str = "first") -> pd.DataFrame:
    subset = subset if subset else None
    return df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Object / string missing values
# ---------------------------------------------------------------------------
def fill_object_missing(df: pd.DataFrame, columns: List[str],
                         method: str, custom_value: Optional[str] = None) -> dict:
    """
    method: one of "custom", "unknown", "not_available", "missing", "mode"
    Returns (new_df, stats) via dict for consistency with rest of module.
    """
    df = df.copy()
    total_filled = 0
    per_column = {}

    for col in columns:
        if col not in df.columns:
            continue
        before = int(df[col].isna().sum())
        if before == 0:
            per_column[col] = 0
            continue

        if method == "custom":
            fill_val = custom_value if custom_value is not None else ""
            df[col] = df[col].fillna(fill_val)
        elif method == "unknown":
            df[col] = df[col].fillna("Unknown")
        elif method == "not_available":
            df[col] = df[col].fillna("Not Available")
        elif method == "missing":
            df[col] = df[col].fillna("Missing")
        elif method == "mode":
            mode_series = df[col].mode(dropna=True)
            fill_val = mode_series.iloc[0] if not mode_series.empty else "Unknown"
            df[col] = df[col].fillna(fill_val)
        else:
            raise ValueError(f"Unsupported fill method: {method}")

        after = int(df[col].isna().sum())
        filled = before - after
        per_column[col] = filled
        total_filled += filled

    stats = {"total_filled": total_filled, "per_column": per_column}
    return df, stats


# ---------------------------------------------------------------------------
# Numeric missing values
# ---------------------------------------------------------------------------
def fill_numeric_missing(df: pd.DataFrame, columns: List[str],
                          method: Literal["ffill", "bfill", "interpolate"]) -> dict:
    df = df.copy()
    total_filled = 0
    per_column = {}

    for col in columns:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            # skip non-numeric columns silently-safe: report 0 filled
            per_column[col] = 0
            continue

        before = int(df[col].isna().sum())
        if before == 0:
            per_column[col] = 0
            continue

        if method == "ffill":
            df[col] = df[col].ffill()
        elif method == "bfill":
            df[col] = df[col].bfill()
        elif method == "interpolate":
            df[col] = df[col].interpolate()
        else:
            raise ValueError(f"Unsupported numeric fill method: {method}")

        after = int(df[col].isna().sum())
        filled = before - after
        per_column[col] = filled
        total_filled += filled

    stats = {"total_filled": total_filled, "per_column": per_column}
    return df, stats


# ---------------------------------------------------------------------------
# Numeric type conversion
# ---------------------------------------------------------------------------
def convert_columns_to_numeric(df: pd.DataFrame, columns: List[str]) -> dict:
    df = df.copy()
    report = {}
    for col in columns:
        if col not in df.columns:
            continue
        converted, n_invalid = safe_to_numeric(df[col])
        df[col] = converted
        report[col] = {"n_invalid_coerced_to_nan": n_invalid}
    return df, report


# ---------------------------------------------------------------------------
# Datetime conversion
# ---------------------------------------------------------------------------
def convert_column_to_datetime(df: pd.DataFrame, column: str) -> dict:
    df = df.copy()
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")

    before_non_null = int(df[column].notna().sum())
    converted = pd.to_datetime(df[column], errors="coerce", format="mixed")
    n_failed = int(converted.isna().sum() - df[column].isna().sum())
    df[column] = converted
    after_non_null = int(df[column].notna().sum())

    report = {
        "before_non_null": before_non_null,
        "after_non_null": after_non_null,
        "n_failed_conversions": max(n_failed, 0),
    }
    return df, report
