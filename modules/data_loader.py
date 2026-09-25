"""
data_loader.py
Handles reading uploaded CSV / Excel files into pandas DataFrames, including
multi-sheet Excel handling and graceful error reporting.
"""

from __future__ import annotations

import io
from typing import Optional, Tuple, List

import pandas as pd


class DataLoadError(Exception):
    """Raised when an uploaded file cannot be parsed into a usable DataFrame."""


def get_file_extension(filename: str) -> str:
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1].lower()


def list_excel_sheets(file_bytes: bytes) -> List[str]:
    """Return the list of sheet names in an Excel file."""
    try:
        xls = pd.ExcelFile(io.BytesIO(file_bytes))
        return xls.sheet_names
    except Exception as e:
        raise DataLoadError(f"Could not read Excel file structure: {e}")


def load_csv(file_bytes: bytes) -> pd.DataFrame:
    if not file_bytes or len(file_bytes) == 0:
        raise DataLoadError("The uploaded CSV file is empty.")
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except pd.errors.EmptyDataError:
        raise DataLoadError("The uploaded CSV file has no columns/data to parse.")
    except pd.errors.ParserError as e:
        raise DataLoadError(f"The CSV file could not be parsed: {e}")
    except UnicodeDecodeError:
        # retry with a more permissive encoding
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding="latin1")
        except Exception as e:
            raise DataLoadError(f"Could not decode CSV file: {e}")
    except Exception as e:
        raise DataLoadError(f"Unexpected error reading CSV file: {e}")

    if df.shape[1] == 0:
        raise DataLoadError("The CSV file does not contain any columns.")
    if df.shape[0] == 0:
        raise DataLoadError("The CSV file does not contain any data rows.")
    return df


def load_excel(file_bytes: bytes, sheet_name: Optional[str] = None) -> pd.DataFrame:
    if not file_bytes or len(file_bytes) == 0:
        raise DataLoadError("The uploaded Excel file is empty.")
    try:
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)
    except ValueError as e:
        raise DataLoadError(f"Could not read the requested sheet: {e}")
    except Exception as e:
        raise DataLoadError(f"Unexpected error reading Excel file: {e}")

    if isinstance(df, dict):
        # sheet_name was None and multiple sheets returned; caller should have
        # picked a sheet already, but guard just in case.
        first_key = next(iter(df))
        df = df[first_key]

    if df.shape[1] == 0:
        raise DataLoadError("The selected sheet does not contain any columns.")
    if df.shape[0] == 0:
        raise DataLoadError("The selected sheet does not contain any data rows.")
    return df


def load_dataset(filename: str, file_bytes: bytes, sheet_name: Optional[str] = None
                  ) -> Tuple[pd.DataFrame, str]:
    """
    Load a dataset from raw bytes based on the file extension.
    Returns (dataframe, detected_file_type).
    Raises DataLoadError on failure.
    """
    ext = get_file_extension(filename)

    if ext == "csv":
        return load_csv(file_bytes), "csv"
    elif ext in ("xlsx", "xls"):
        df = load_excel(file_bytes, sheet_name=sheet_name)
        return df, "excel"
    else:
        raise DataLoadError(
            f"Unsupported file type '.{ext}'. Please upload a .csv, .xlsx, or .xls file."
        )
