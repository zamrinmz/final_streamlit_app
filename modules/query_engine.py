"""
query_engine.py

A fully deterministic, local natural-language query engine for tabular data.
NO external API calls and NO AI/LLM model are used anywhere in this module.

Supported intents:
  - Aggregation (sum/total, average/mean, min, max, count) grouped by a column
  - Ranking ("best selling X", "top N ... by ...", "bottom N ...")
  - Simple filtering ("X in <value>", "X where <col> > <value>")
  - Time-based grouping ("... by month/year/day") when a datetime column exists

Design: the query is matched against a small, explicit grammar of recognized
phrases/keywords. If the engine cannot confidently identify the columns and
operation required, it returns a structured failure result asking the user
to pick columns manually, rather than guessing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import pandas as pd

from modules.utils import (
    normalize_text, match_column, get_numeric_columns,
    get_categorical_columns, get_datetime_columns,
)

AGG_KEYWORDS = {
    "sum": ["sum", "total"],
    "mean": ["average", "avg", "mean"],
    "min": ["minimum", "min", "lowest", "smallest"],
    "max": ["maximum", "max", "highest", "largest", "biggest"],
    "count": ["count", "number of records", "number of", "how many"],
}

RANK_DESC_WORDS = ["best", "top", "highest", "most", "largest", "biggest"]
RANK_ASC_WORDS = ["worst", "bottom", "lowest", "least", "smallest"]

TIME_WORDS = {"month": "M", "year": "Y", "day": "D", "week": "W", "quarter": "Q"}

FILTER_COMPARATORS = [
    (">=", ">="), ("<=", "<="), ("==", "=="), ("!=", "!="), (">", ">"), ("<", "<"),
]


@dataclass
class QueryResult:
    success: bool
    interpretation: Dict[str, Any] = field(default_factory=dict)
    result_df: Optional[pd.DataFrame] = None
    message: str = ""
    suggested_chart: Optional[Dict[str, Any]] = None
    needs_column_selection: bool = False
    candidate_group_cols: List[str] = field(default_factory=list)
    candidate_metric_cols: List[str] = field(default_factory=list)


def _detect_aggregation(norm_query: str) -> Optional[str]:
    for agg, words in AGG_KEYWORDS.items():
        for w in words:
            if w in norm_query:
                return agg
    return None


def _detect_top_n(norm_query: str) -> Optional[int]:
    m = re.search(r"\btop\s+(\d+)\b", norm_query)
    if m:
        return int(m.group(1))
    m = re.search(r"\bbottom\s+(\d+)\b", norm_query)
    if m:
        return int(m.group(1))
    return None


def _detect_rank_direction(norm_query: str) -> Optional[str]:
    for w in RANK_ASC_WORDS:
        if w in norm_query:
            return "asc"
    for w in RANK_DESC_WORDS:
        if w in norm_query:
            return "desc"
    return None


def _detect_time_grouping(norm_query: str) -> Optional[str]:
    for word, freq in TIME_WORDS.items():
        if re.search(rf"\bby\s+{word}\b", norm_query) or re.search(rf"\bper\s+{word}\b", norm_query):
            return freq
        if re.search(rf"\b{word}ly\b", norm_query):
            return freq
    return None


def _extract_filter(norm_query: str, df: pd.DataFrame):
    """
    Detect simple filters:
      "<col> where <col2> > 100"
      "sales in india"  -> categorical equality filter
    Returns (filter_col, operator, value, remaining_query) or None.
    """
    # comparator-based filter: "where X > 100" or "X > 100"
    for op_word, op_symbol in FILTER_COMPARATORS:
        m = re.search(r"([a-z0-9 _]+?)\s*" + re.escape(op_word) + r"\s*([0-9.]+)", norm_query)
        if m:
            col_phrase = m.group(1).replace("where", "").strip()
            value = float(m.group(2))
            col = match_column(df, col_phrase, prefer_types=["numeric"])
            if col:
                remaining = norm_query[:m.start()] + norm_query[m.end():]
                return col, op_symbol, value, remaining

    # equality-style filter: "<metric/group> in <value>"  e.g. "sales in india"
    m = re.search(r"\bin\s+([a-z0-9 _]+)$", norm_query)
    if m:
        raw_value = m.group(1).strip()
        # try to find a categorical column containing this value
        for col in get_categorical_columns(df):
            try:
                values_norm = df[col].dropna().astype(str).str.lower().unique()
            except Exception:
                continue
            if raw_value in values_norm:
                remaining = norm_query[:m.start()]
                return col, "==", raw_value, remaining

    return None


def _split_by_per(norm_query: str) -> Optional[List[str]]:
    """Split a query on 'by' or 'per' to get [left, right] phrases."""
    for sep in [" by ", " per "]:
        if sep in norm_query:
            parts = norm_query.split(sep, 1)
            if len(parts) == 2:
                return [p.strip() for p in parts]
    return None


def _strip_keywords(text: str, extra: Optional[List[str]] = None) -> str:
    words_to_strip = [
        "show", "me", "the", "what", "which", "is", "are", "of", "please",
        "count", "number", "records", "sum", "total", "average", "avg", "mean",
        "minimum", "min", "maximum", "max", "highest", "lowest", "best", "worst",
        "selling", "top", "bottom", "a", "an", "has",
    ]
    if extra:
        words_to_strip += extra
    tokens = text.split()
    tokens = [t for t in tokens if t not in words_to_strip and not t.isdigit()]
    return " ".join(tokens).strip()


def parse_and_run_query(df: pd.DataFrame, query: str) -> QueryResult:
    if df is None or df.empty:
        return QueryResult(success=False, message="No dataset is loaded, or the dataset is empty.")
    if not query or not query.strip():
        return QueryResult(success=False, message="Please enter a question about your data.")

    norm_query = normalize_text(query)
    working_df = df

    # ---- optional filter extraction -------------------------------------
    filter_info = None
    extracted = _extract_filter(norm_query, df)
    if extracted:
        filter_col, op, value, remaining_query = extracted
        try:
            if op == "==":
                mask = working_df[filter_col].astype(str).str.lower() == str(value).lower()
            else:
                series = pd.to_numeric(working_df[filter_col], errors="coerce")
                if op == ">":
                    mask = series > value
                elif op == "<":
                    mask = series < value
                elif op == ">=":
                    mask = series >= value
                elif op == "<=":
                    mask = series <= value
                elif op == "!=":
                    mask = series != value
                else:
                    mask = pd.Series([True] * len(working_df))
            working_df = working_df[mask]
            filter_info = {"column": filter_col, "operator": op, "value": value}
            norm_query = remaining_query.strip()
        except Exception:
            filter_info = None  # if filtering fails, fall back to unfiltered query

    if working_df.empty:
        return QueryResult(
            success=False,
            message="The filter condition matched no rows in the dataset.",
            interpretation={"filter": filter_info} if filter_info else {},
        )

    # ---- aggregation / ranking / time keywords ---------------------------
    agg = _detect_aggregation(norm_query)
    top_n = _detect_top_n(norm_query)
    rank_dir = _detect_rank_direction(norm_query)
    time_freq = _detect_time_grouping(norm_query)

    # ---- time-based grouping ----------------------------------------------
    if time_freq:
        datetime_cols = get_datetime_columns(working_df)
        numeric_cols = get_numeric_columns(working_df)
        if not datetime_cols:
            return QueryResult(
                success=False,
                message=("This looks like a time-based question, but no datetime column was "
                          "found. Convert a column to datetime first in the Data Cleaning "
                          "section, then try again."),
            )
        date_col = datetime_cols[0]
        # metric: best numeric column mentioned, else first numeric column
        metric_col = None
        parts = _split_by_per(norm_query)
        if parts:
            metric_col = match_column(working_df, parts[0], prefer_types=["numeric"])
        if not metric_col and numeric_cols:
            metric_col = numeric_cols[0]
        if not metric_col:
            return QueryResult(success=False, message="Could not identify a numeric column to summarize.")

        agg_fn = agg or "sum"
        period = working_df[date_col].dt.to_period(time_freq).astype(str)
        grouped = working_df.groupby(period)[metric_col].agg(agg_fn).reset_index()
        grouped.columns = [date_col, f"{agg_fn.title()} of {metric_col}"]
        grouped = grouped.sort_values(by=date_col)

        interpretation = {
            "group_by": date_col,
            "metric": metric_col,
            "operation": agg_fn,
            "time_frequency": time_freq,
            "filter": filter_info,
        }
        return QueryResult(
            success=True,
            interpretation=interpretation,
            result_df=grouped,
            message="Query understood.",
            suggested_chart={"type": "line", "x": date_col, "y": grouped.columns[1]},
        )

    # ---- identify group column & metric column ----------------------------
    group_col = None
    metric_col = None

    parts = _split_by_per(norm_query)
    if parts:
        left, right = parts
        left_col = match_column(working_df, left)
        right_col = match_column(working_df, right)

        left_numeric = bool(left_col) and pd.api.types.is_numeric_dtype(working_df[left_col])
        right_numeric = bool(right_col) and pd.api.types.is_numeric_dtype(working_df[right_col])

        if left_numeric and not right_numeric:
            # "sales by region" -> metric=left, group=right
            metric_col, group_col = left_col, right_col
        elif right_numeric and not left_numeric:
            # "region per sales" -> metric=right, group=left
            metric_col, group_col = right_col, left_col
        elif right_numeric and left_numeric:
            # both numeric: treat left as metric, right as group if possible
            metric_col = left_col
            group_col = right_col if right_col != metric_col else None
        else:
            # neither side numeric (e.g. "count of customers by region"):
            # default English convention "<what> by <dimension>" -> group=right
            group_col = right_col if right_col else left_col
            metric_col = left_col if left_col != group_col else None

    # "best selling products" / "top 10 products by sales" style queries
    # without explicit "by"/"per": try to find a categorical mention anywhere.
    if group_col is None:
        residual = _strip_keywords(norm_query)
        group_col = match_column(working_df, residual, prefer_types=["categorical", "datetime"])
        if group_col is None:
            for cand in get_categorical_columns(working_df):
                if normalize_text(cand) in norm_query or normalize_text(cand).rstrip("s") in norm_query:
                    group_col = cand
                    break

    if metric_col is None or metric_col == group_col:
        # "selling" / "sales" implies a sales-like numeric column even if not spelled out
        metric_col = match_column(working_df, norm_query, prefer_types=["numeric"])
        if metric_col == group_col:
            metric_col = None
        if metric_col is None:
            numeric_cols = get_numeric_columns(working_df)
            numeric_cols = [c for c in numeric_cols if c != group_col]
            if len(numeric_cols) == 1:
                metric_col = numeric_cols[0]
            elif len(numeric_cols) > 1 and agg is not None:
                # multiple numeric columns but an aggregation keyword was given;
                # prefer a column whose name suggests it's the primary metric
                metric_col = numeric_cols[0]

    # if a numeric filter was applied, default the metric to the filtered column
    if metric_col is None and filter_info and filter_info.get("operator") != "==":
        fcol = filter_info.get("column")
        if fcol and fcol != group_col and pd.api.types.is_numeric_dtype(working_df[fcol]):
            metric_col = fcol

    # count-of-records queries don't require a metric column
    if agg == "count" and metric_col is None:
        metric_col = group_col  # size-based count

    if group_col is None:
        # No grouping dimension identified. If we at least have a metric
        # (optionally narrowed by a filter), fall back to a scalar summary
        # instead of failing outright.
        if metric_col is None:
            metric_col = match_column(working_df, norm_query, prefer_types=["numeric"])
        if metric_col:
            agg_fn = agg or "sum"
            try:
                value = working_df[metric_col].agg(agg_fn)
            except Exception as e:
                return QueryResult(success=False, message=f"Could not execute the query: {e}")
            result_df = pd.DataFrame({metric_col: [value]})
            interpretation = {
                "metric": metric_col,
                "operation": agg_fn,
                "filter": filter_info,
                "note": "No grouping column was identified; showing an overall summary.",
            }
            return QueryResult(success=True, interpretation=interpretation, result_df=result_df,
                                message="Query understood (summary result).")

        candidates = get_categorical_columns(working_df) + get_datetime_columns(working_df)
        return QueryResult(
            success=False,
            needs_column_selection=True,
            candidate_group_cols=candidates,
            candidate_metric_cols=get_numeric_columns(working_df),
            message=("I couldn't confidently identify which column to group by. "
                     "Please select the columns to use below."),
        )

    if metric_col is None:
        candidates = get_numeric_columns(working_df)
        return QueryResult(
            success=False,
            needs_column_selection=True,
            candidate_group_cols=[group_col],
            candidate_metric_cols=candidates,
            message=("I identified a grouping column but couldn't confidently identify "
                     "a numeric column to summarize. Please select one below."),
        )

    agg_fn = agg or "sum"

    try:
        if agg_fn == "count":
            grouped = working_df.groupby(group_col).size().reset_index(name="Count")
            metric_label = "Count"
        else:
            grouped = working_df.groupby(group_col)[metric_col].agg(agg_fn).reset_index()
            metric_label = f"{agg_fn.title()} of {metric_col}"
            grouped.columns = [group_col, metric_label]
    except Exception as e:
        return QueryResult(success=False, message=f"Could not execute the query: {e}")

    # ranking / sorting
    ascending = False
    if rank_dir == "asc":
        ascending = True
    grouped = grouped.sort_values(by=grouped.columns[1], ascending=ascending)

    limit = top_n
    if limit is None and (rank_dir is not None or any(w in norm_query for w in ["best selling", "top selling"])):
        limit = 10
    if limit:
        grouped = grouped.head(limit)

    grouped = grouped.reset_index(drop=True)

    interpretation = {
        "group_by": group_col,
        "metric": metric_col if agg_fn != "count" else "Count",
        "operation": agg_fn,
        "sort": "descending" if not ascending else "ascending",
        "limit": limit,
        "filter": filter_info,
    }

    return QueryResult(
        success=True,
        interpretation=interpretation,
        result_df=grouped,
        message="Query understood.",
        suggested_chart={"type": "bar", "x": group_col, "y": grouped.columns[1]},
    )
