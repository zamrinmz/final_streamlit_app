"""
visualizations.py
Chart-generation helpers for Matplotlib, Seaborn, and Plotly.

Each build_* function validates its inputs and raises ChartValidationError
with a clear, user-facing message rather than letting a raw exception
propagate to the Streamlit UI.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from modules.utils import get_numeric_columns


class ChartValidationError(Exception):
    """Raised when a requested chart configuration is invalid."""


AGG_FUNCS = {"sum": "sum", "mean": "mean", "count": "count", "min": "min", "max": "max"}


def _apply_aggregation(df: pd.DataFrame, x: str, y: Optional[str], agg: Optional[str],
                        sort: Optional[str] = None, top_n: Optional[int] = None) -> pd.DataFrame:
    """Optionally group df by x and aggregate y, then sort/limit."""
    work = df
    if agg and agg != "none" and y:
        if agg == "count":
            work = work.groupby(x, dropna=False).size().reset_index(name=y)
        else:
            work = work.groupby(x, dropna=False)[y].agg(AGG_FUNCS[agg]).reset_index()
    if sort in ("ascending", "descending") and y in work.columns:
        work = work.sort_values(by=y, ascending=(sort == "ascending"))
    if top_n:
        work = work.head(int(top_n))
    return work


def _require_column(df: pd.DataFrame, col: Optional[str], label: str):
    if not col:
        raise ChartValidationError(f"{label} is required for this chart type.")
    if col not in df.columns:
        raise ChartValidationError(f"Column '{col}' was not found in the dataset.")


def _require_numeric(df: pd.DataFrame, col: str, label: str):
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise ChartValidationError(f"{label} ('{col}') must be a numeric column for this chart type.")


# ---------------------------------------------------------------------------
# Matplotlib
# ---------------------------------------------------------------------------
def build_matplotlib_chart(df: pd.DataFrame, chart_type: str, x: Optional[str], y: Optional[str],
                            agg: Optional[str] = None, title: str = "", sort: Optional[str] = None,
                            top_n: Optional[int] = None, width: float = 8, height: float = 5):
    fig, ax = plt.subplots(figsize=(width, height))

    if chart_type == "Bar Chart":
        _require_column(df, x, "X-axis")
        _require_column(df, y, "Y-axis")
        data = _apply_aggregation(df, x, y, agg or "sum", sort, top_n)
        ax.bar(data[x].astype(str), data[y])
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        plt.xticks(rotation=45, ha="right")

    elif chart_type == "Line Chart":
        _require_column(df, x, "X-axis")
        _require_column(df, y, "Y-axis")
        data = _apply_aggregation(df, x, y, agg, sort=None, top_n=None)
        try:
            data = data.sort_values(by=x)
        except Exception:
            pass
        ax.plot(data[x].astype(str), data[y], marker="o")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        plt.xticks(rotation=45, ha="right")

    elif chart_type == "Histogram":
        _require_column(df, x, "Column")
        _require_numeric(df, x, "Column")
        ax.hist(df[x].dropna(), bins=30, color="#4C72B0", edgecolor="white")
        ax.set_xlabel(x)
        ax.set_ylabel("Frequency")

    elif chart_type == "Scatter Plot":
        _require_column(df, x, "X-axis")
        _require_column(df, y, "Y-axis")
        _require_numeric(df, x, "X-axis")
        _require_numeric(df, y, "Y-axis")
        ax.scatter(df[x], df[y], alpha=0.7)
        ax.set_xlabel(x)
        ax.set_ylabel(y)

    elif chart_type == "Pie Chart":
        _require_column(df, x, "Category column")
        _require_column(df, y, "Value column")
        data = _apply_aggregation(df, x, y, agg or "sum", "descending", top_n or 10)
        ax.pie(data[y], labels=data[x].astype(str), autopct="%1.1f%%")
        ax.axis("equal")

    else:
        plt.close(fig)
        raise ChartValidationError(f"Unsupported Matplotlib chart type: {chart_type}")

    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Seaborn
# ---------------------------------------------------------------------------
def build_seaborn_chart(df: pd.DataFrame, chart_type: str, x: Optional[str], y: Optional[str],
                         agg: Optional[str] = None, title: str = "", sort: Optional[str] = None,
                         top_n: Optional[int] = None, width: float = 8, height: float = 5):
    fig, ax = plt.subplots(figsize=(width, height))
    sns.set_style("whitegrid")

    if chart_type == "Bar Plot":
        _require_column(df, x, "X-axis")
        _require_column(df, y, "Y-axis")
        data = _apply_aggregation(df, x, y, agg or "sum", sort, top_n)
        sns.barplot(data=data, x=x, y=y, ax=ax)
        plt.xticks(rotation=45, ha="right")

    elif chart_type == "Count Plot":
        _require_column(df, x, "Column")
        data = df
        if top_n:
            top_categories = df[x].value_counts().head(int(top_n)).index
            data = df[df[x].isin(top_categories)]
        sns.countplot(data=data, x=x, ax=ax,
                       order=data[x].value_counts().index if top_n else None)
        plt.xticks(rotation=45, ha="right")

    elif chart_type == "Histogram / Distribution":
        _require_column(df, x, "Column")
        _require_numeric(df, x, "Column")
        sns.histplot(df[x].dropna(), kde=True, ax=ax)

    elif chart_type == "Box Plot":
        _require_column(df, y, "Value (Y-axis)")
        _require_numeric(df, y, "Value (Y-axis)")
        if x:
            sns.boxplot(data=df, x=x, y=y, ax=ax)
            plt.xticks(rotation=45, ha="right")
        else:
            sns.boxplot(data=df, y=y, ax=ax)

    elif chart_type == "Scatter Plot":
        _require_column(df, x, "X-axis")
        _require_column(df, y, "Y-axis")
        _require_numeric(df, x, "X-axis")
        _require_numeric(df, y, "Y-axis")
        sns.scatterplot(data=df, x=x, y=y, ax=ax)

    elif chart_type == "Heatmap":
        numeric_cols = get_numeric_columns(df)
        if len(numeric_cols) < 2:
            raise ChartValidationError("A heatmap requires at least two numeric columns in the dataset.")
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, fmt=".2f")

    else:
        plt.close(fig)
        raise ChartValidationError(f"Unsupported Seaborn chart type: {chart_type}")

    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plotly
# ---------------------------------------------------------------------------
def build_plotly_chart(df: pd.DataFrame, chart_type: str, x: Optional[str], y: Optional[str],
                        agg: Optional[str] = None, title: str = "", sort: Optional[str] = None,
                        top_n: Optional[int] = None, width: Optional[float] = None,
                        height: Optional[float] = None):
    height = height or 500
    width = width if width else None

    if chart_type == "Interactive Bar Chart":
        _require_column(df, x, "X-axis")
        _require_column(df, y, "Y-axis")
        data = _apply_aggregation(df, x, y, agg or "sum", sort, top_n)
        fig = px.bar(data, x=x, y=y, title=title or None)

    elif chart_type == "Line Chart":
        _require_column(df, x, "X-axis")
        _require_column(df, y, "Y-axis")
        data = _apply_aggregation(df, x, y, agg, sort=None, top_n=None)
        try:
            data = data.sort_values(by=x)
        except Exception:
            pass
        fig = px.line(data, x=x, y=y, markers=True, title=title or None)

    elif chart_type == "Scatter Plot":
        _require_column(df, x, "X-axis")
        _require_column(df, y, "Y-axis")
        _require_numeric(df, x, "X-axis")
        _require_numeric(df, y, "Y-axis")
        fig = px.scatter(df, x=x, y=y, title=title or None)

    elif chart_type == "Pie / Donut Chart":
        _require_column(df, x, "Category column")
        _require_column(df, y, "Value column")
        data = _apply_aggregation(df, x, y, agg or "sum", "descending", top_n or 10)
        fig = px.pie(data, names=x, values=y, hole=0.35, title=title or None)

    elif chart_type == "Histogram":
        _require_column(df, x, "Column")
        _require_numeric(df, x, "Column")
        fig = px.histogram(df, x=x, title=title or None)

    elif chart_type == "Box Plot":
        _require_column(df, y, "Value (Y-axis)")
        _require_numeric(df, y, "Value (Y-axis)")
        fig = px.box(df, x=x if x else None, y=y, title=title or None)

    else:
        raise ChartValidationError(f"Unsupported Plotly chart type: {chart_type}")

    fig.update_layout(height=height)
    if width:
        fig.update_layout(width=width)
    return fig


CHART_TYPES_BY_LIBRARY = {
    "Matplotlib": ["Bar Chart", "Line Chart", "Histogram", "Scatter Plot", "Pie Chart"],
    "Seaborn": ["Bar Plot", "Count Plot", "Histogram / Distribution", "Box Plot",
                "Scatter Plot", "Heatmap"],
    "Plotly": ["Interactive Bar Chart", "Line Chart", "Scatter Plot", "Pie / Donut Chart",
               "Histogram", "Box Plot"],
}

# which chart types need an x, a y, both, or neither (used by the UI to show/hide controls)
CHART_FIELD_REQUIREMENTS = {
    "Bar Chart": {"x": True, "y": True},
    "Line Chart": {"x": True, "y": True},
    "Histogram": {"x": True, "y": False},
    "Scatter Plot": {"x": True, "y": True},
    "Pie Chart": {"x": True, "y": True},
    "Bar Plot": {"x": True, "y": True},
    "Count Plot": {"x": True, "y": False},
    "Histogram / Distribution": {"x": True, "y": False},
    "Box Plot": {"x": False, "y": True},
    "Heatmap": {"x": False, "y": False},
    "Interactive Bar Chart": {"x": True, "y": True},
    "Pie / Donut Chart": {"x": True, "y": True},
}
