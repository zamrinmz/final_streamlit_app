"""
utils.py
General-purpose helper functions shared across modules:
 - text normalization
 - synonym-based / fuzzy column matching
 - safe numeric & datetime conversion helpers
 - small formatting helpers
"""

from __future__ import annotations

import re
import difflib
from typing import Optional, List, Dict, Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Synonym dictionary used by the natural-language query engine and by
# column-detection helpers elsewhere in the app.
# Keys are "canonical concepts"; values are lists of words/phrases that
# should be treated as referring to that concept.
# ---------------------------------------------------------------------------
COLUMN_SYNONYMS: Dict[str, List[str]] = {
    "sales": ["sales", "sale", "selling", "sold", "revenue", "amount", "turnover",
              "total_sales", "price", "value", "income", "earnings"],
    "quantity": ["quantity", "qty", "units", "volume", "unit_sold", "units_sold"],
    "region": ["region", "area", "territory", "zone", "location", "state"],
    "product": ["product", "item", "product_name", "item_name", "products", "items"],
    "customer": ["customer", "client", "buyer", "customers", "clients"],
    "category": ["category", "cat", "type", "segment", "categories"],
    "country": ["country", "nation", "countries"],
    "date": ["date", "time", "datetime", "order_date", "timestamp", "order date"],
    "profit": ["profit", "margin", "net_income"],
    "employee": ["employee", "staff", "worker", "rep", "salesperson"],
}


def normalize_text(text: str) -> str:
    """Lowercase, strip, and collapse internal whitespace / underscores."""
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_columns_map(columns: Iterable[str]) -> Dict[str, str]:
    """Return {normalized_name: original_column_name} for a set of columns."""
    return {normalize_text(c): c for c in columns}


def find_synonym_concept(token: str) -> Optional[str]:
    """Given a normalized token/phrase, return the canonical concept it maps to, if any."""
    token = normalize_text(token)
    for concept, words in COLUMN_SYNONYMS.items():
        for w in words:
            if normalize_text(w) == token:
                return concept
    return None


def match_column(df: pd.DataFrame, phrase: str, prefer_types: Optional[List[str]] = None
                  ) -> Optional[str]:
    """
    Attempt to match a natural-language phrase to an actual DataFrame column.

    Matching is done at both the whole-phrase and individual-word level so that
    phrases like "products where sales" still surface the "sales"-like column.
    When prefer_types is given, ONLY columns of those types are ever returned —
    the function will not fall back to a wrongly-typed column just because it
    text-matched, since that would silently misinterpret the query.

    Matching order (each stage filtered to prefer_types when given):
      1. Exact normalized match of the whole phrase against a column name.
      2. Synonym concept match, checked against the whole phrase AND each word.
      3. Substring match between column names and the phrase (whole phrase, then words).
      4. Fuzzy match (difflib) against normalized column names (whole phrase, then words).
    """
    if not phrase:
        return None

    norm_map = normalize_columns_map(df.columns)
    norm_phrase = normalize_text(phrase)
    if not norm_phrase:
        return None

    words = [w for w in norm_phrase.split(" ") if w]
    search_terms = [norm_phrase] + words  # whole phrase first, then individual words

    def type_ok(orig_col: str) -> bool:
        if not prefer_types:
            return True
        return _column_type(df[orig_col]) in prefer_types

    # 1. exact whole-phrase match
    if norm_phrase in norm_map and type_ok(norm_map[norm_phrase]):
        return norm_map[norm_phrase]

    # 2. synonym concept match (whole phrase, then each word)
    for term in search_terms:
        concept = find_synonym_concept(term)
        if not concept:
            continue
        syn_words = [normalize_text(w) for w in COLUMN_SYNONYMS[concept]]
        candidates = [orig for norm_col, orig in norm_map.items()
                      if norm_col in syn_words and type_ok(orig)]
        if candidates:
            return candidates[0]
        # relaxed: column name contains one of the synonym words
        candidates = [orig for norm_col, orig in norm_map.items()
                      if any(sw in norm_col for sw in syn_words) and type_ok(orig)]
        if candidates:
            return candidates[0]

    # 3. substring match between phrase and column names
    for term in search_terms:
        if len(term) < 3:
            continue
        candidates = [orig for norm_col, orig in norm_map.items()
                      if norm_col and (norm_col in term or term in norm_col) and type_ok(orig)]
        if candidates:
            return candidates[0]

    # 4. fuzzy match, restricted to allowed types
    allowed_norm_cols = [norm_col for norm_col, orig in norm_map.items() if type_ok(orig)]
    for term in search_terms:
        if len(term) < 3:
            continue
        close = difflib.get_close_matches(term, allowed_norm_cols, n=1, cutoff=0.72)
        if close:
            return norm_map[close[0]]

    return None


def _column_type(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns
            if not pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_datetime64_any_dtype(df[c])]


def get_datetime_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]


def try_parse_datetime_columns(df: pd.DataFrame, max_check_rows: int = 200) -> List[str]:
    """
    Heuristically detect object columns that LOOK like dates without mutating
    the DataFrame. Returns a list of column names that are reasonable datetime
    conversion candidates.
    """
    candidates = []
    for col in get_categorical_columns(df):
        sample = df[col].dropna().astype(str).head(max_check_rows)
        if sample.empty:
            continue
        try:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        except (ValueError, TypeError):
            try:
                parsed = pd.to_datetime(sample, errors="coerce")
            except Exception:
                continue
        success_ratio = parsed.notna().mean()
        if success_ratio >= 0.8:
            candidates.append(col)
    return candidates


def format_bytes(num_bytes: float) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} TB"


def safe_to_numeric(series: pd.Series):
    """Convert a series to numeric, coercing invalid values to NaN. Returns (series, n_invalid)."""
    original_non_null = series.notna().sum()
    converted = pd.to_numeric(series, errors="coerce")
    new_non_null = converted.notna().sum()
    n_invalid = original_non_null - new_non_null
    return converted, int(n_invalid)
