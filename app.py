"""
app.py
Streamlit application: CSV / Excel data cleaning, analysis, natural-language
querying (fully local/deterministic - no AI/LLM/API calls), and visualization.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import traceback

import pandas as pd
import streamlit as st

from modules import data_loader as dl
from modules import data_summary as dsm
from modules import data_cleaner as dc
from modules import query_engine as qe
from modules import visualizations as viz
from modules.utils import (
    get_numeric_columns, get_categorical_columns, get_datetime_columns,
    try_parse_datetime_columns,
)


# =============================================================================
# Page config & session state initialization
# =============================================================================
st.set_page_config(page_title="Data Cleaning & Analysis Studio", layout="wide",
                    page_icon="📊")

DEFAULT_STATE = {
    "original_df": None,
    "working_df": None,
    "filename": None,
    "file_type": None,
    "excel_sheet": None,
    "excel_sheets_available": [],
    "cleaning_log": [],
    "last_query_result": None,
    "last_query_text": "",
    "pending_excel_bytes": None,
    "pending_excel_filename": None,
}
for key, default in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default


def has_data() -> bool:
    return st.session_state.working_df is not None and not st.session_state.working_df.empty


def set_working_df(new_df: pd.DataFrame, log_entry: str | None = None):
    st.session_state.working_df = new_df
    if log_entry:
        st.session_state.cleaning_log.append(log_entry)


def show_change_summary(rows_before, rows_after, missing_before=None, missing_after=None,
                         extra_lines=None):
    lines = [
        f"Rows before: {rows_before:,}",
        f"Rows after: {rows_after:,}",
        f"Rows removed: {rows_before - rows_after:,}",
    ]
    if missing_before is not None and missing_after is not None:
        lines.append(f"Missing values before: {missing_before:,}")
        lines.append(f"Missing values after: {missing_after:,}")
    if extra_lines:
        lines.extend(extra_lines)
    st.success("Operation completed\n\n" + "\n".join(lines))


# =============================================================================
# Sidebar navigation
# =============================================================================
st.sidebar.title("📊 Data Studio")
page = st.sidebar.radio(
    "Navigate",
    ["Upload Dataset", "Dataset Overview", "Data Cleaning", "Ask Questions",
     "Visualization", "Download"],
)

st.sidebar.markdown("---")
if has_data():
    df = st.session_state.working_df
    st.sidebar.markdown("### Current Dataset")
    st.sidebar.write(f"**File:** {st.session_state.filename}")
    st.sidebar.write(f"**Rows:** {df.shape[0]:,}")
    st.sidebar.write(f"**Columns:** {df.shape[1]:,}")
    st.sidebar.write(f"**Missing values:** {int(df.isna().sum().sum()):,}")
    st.sidebar.write(f"**Duplicate rows:** {int(df.duplicated().sum()):,}")
    if st.sidebar.button("🔄 Reset Dataset to Original", use_container_width=True):
        st.session_state["_confirm_reset"] = True

    if st.session_state.get("_confirm_reset"):
        st.sidebar.warning("This will discard all cleaning changes.")
        c1, c2 = st.sidebar.columns(2)
        if c1.button("Confirm Reset", use_container_width=True):
            st.session_state.working_df = st.session_state.original_df.copy()
            st.session_state.cleaning_log = []
            st.session_state.last_query_result = None
            st.session_state["_confirm_reset"] = False
            st.rerun()
        if c2.button("Cancel", use_container_width=True):
            st.session_state["_confirm_reset"] = False
            st.rerun()
else:
    st.sidebar.info("No dataset loaded yet.")


# =============================================================================
# PAGE: Upload Dataset
# =============================================================================
if page == "Upload Dataset":
    st.title("📁 Upload Dataset")
    st.write("Upload a CSV or Excel file to begin cleaning and analyzing your data.")

    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        ext = dl.get_file_extension(uploaded_file.name)

        try:
            if ext in ("xlsx", "xls"):
                sheets = dl.list_excel_sheets(file_bytes)
                if len(sheets) > 1:
                    st.info(f"This Excel file contains {len(sheets)} sheets.")
                    chosen_sheet = st.selectbox("Select the sheet to analyze", sheets)
                else:
                    chosen_sheet = sheets[0]

                if st.button("Load Dataset", type="primary"):
                    new_df = dl.load_excel(file_bytes, sheet_name=chosen_sheet)
                    st.session_state.original_df = new_df.copy()
                    st.session_state.working_df = new_df.copy()
                    st.session_state.filename = uploaded_file.name
                    st.session_state.file_type = "excel"
                    st.session_state.excel_sheet = chosen_sheet
                    st.session_state.cleaning_log = []
                    st.session_state.last_query_result = None
                    st.success(f"Loaded '{uploaded_file.name}' — sheet '{chosen_sheet}' "
                               f"({new_df.shape[0]:,} rows × {new_df.shape[1]:,} columns)")
            else:
                if st.button("Load Dataset", type="primary"):
                    new_df = dl.load_csv(file_bytes)
                    st.session_state.original_df = new_df.copy()
                    st.session_state.working_df = new_df.copy()
                    st.session_state.filename = uploaded_file.name
                    st.session_state.file_type = "csv"
                    st.session_state.excel_sheet = None
                    st.session_state.cleaning_log = []
                    st.session_state.last_query_result = None
                    st.success(f"Loaded '{uploaded_file.name}' "
                               f"({new_df.shape[0]:,} rows × {new_df.shape[1]:,} columns)")
        except dl.DataLoadError as e:
            st.error(f"Could not load file: {e}")
        except Exception as e:
            st.error("An unexpected error occurred while loading the file. "
                      "Please check that it is a valid CSV or Excel file.")
            with st.expander("Technical details"):
                st.code(str(e))

    if has_data():
        st.markdown("---")
        st.subheader(f"Preview: {st.session_state.filename}")
        if st.session_state.excel_sheet:
            st.caption(f"Sheet: {st.session_state.excel_sheet}")
        st.dataframe(st.session_state.working_df.head(20), use_container_width=True)


# =============================================================================
# PAGE: Dataset Overview
# =============================================================================
elif page == "Dataset Overview":
    st.title("📈 Dataset Overview")

    if not has_data():
        st.warning("No dataset loaded. Please upload a dataset first.")
    else:
        df = st.session_state.working_df
        summary = dsm.get_dataset_summary(df)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{summary['n_rows']:,}")
        c2.metric("Columns", f"{summary['n_cols']:,}")
        c3.metric("Missing Values", f"{summary['n_missing']:,}")
        c4.metric("Duplicate Rows", f"{summary['n_duplicates']:,}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Numeric Columns", summary["n_numeric"])
        c6.metric("Object/String Columns", summary["n_object"])
        c7.metric("Datetime Columns", summary["n_datetime"])
        c8.metric("Memory Usage", summary["memory_usage"])

        st.markdown("### Column-Level Summary")
        st.dataframe(dsm.get_column_summary(df), use_container_width=True)

        st.markdown("### Data Preview")
        tab1, tab2 = st.tabs(["First rows", "Last rows"])
        with tab1:
            n_head = st.slider("Number of rows", 5, min(100, len(df)) if len(df) >= 5 else 5,
                                min(10, len(df)), key="head_slider")
            st.dataframe(df.head(n_head), use_container_width=True)
        with tab2:
            n_tail = st.slider("Number of rows", 5, min(100, len(df)) if len(df) >= 5 else 5,
                                min(10, len(df)), key="tail_slider")
            st.dataframe(df.tail(n_tail), use_container_width=True)

        st.caption(f"Current shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")

        if st.session_state.cleaning_log:
            with st.expander("Cleaning history"):
                for i, entry in enumerate(st.session_state.cleaning_log, 1):
                    st.write(f"{i}. {entry}")


# =============================================================================
# PAGE: Data Cleaning
# =============================================================================
elif page == "Data Cleaning":
    st.title("🧹 Data Cleaning")

    if not has_data():
        st.warning("No dataset loaded. Please upload a dataset first.")
    else:
        df = st.session_state.working_df

        tabs = st.tabs([
            "Missing Rows", "Duplicates", "Object/String Fill",
            "Numeric Fill", "Type Conversion",
        ])

        # ---------------------------------------------------------------
        # Missing rows
        # ---------------------------------------------------------------
        with tabs[0]:
            st.subheader("Remove Rows Containing Missing Values")
            how = st.radio("Removal strategy", ["Remove rows with ANY missing value",
                                                  "Remove rows only when ALL values are missing"])
            how_key = "any" if "ANY" in how else "all"

            use_subset = st.checkbox("Only consider specific columns")
            subset = None
            if use_subset:
                subset = st.multiselect("Columns to check", df.columns.tolist())

            preview = dc.preview_remove_missing_rows(df, how=how_key, subset=subset or None)
            st.info(f"Rows before: {preview['rows_before']:,} | "
                    f"Rows that will be removed: {preview['rows_removed']:,} | "
                    f"Rows remaining: {preview['rows_after']:,}")

            if st.button("Apply: Remove Missing Rows", type="primary"):
                if use_subset and not subset:
                    st.error("Please select at least one column, or uncheck 'Only consider specific columns'.")
                else:
                    missing_before = int(df.isna().sum().sum())
                    new_df = dc.remove_missing_rows(df, how=how_key, subset=subset or None)
                    missing_after = int(new_df.isna().sum().sum())
                    set_working_df(new_df, f"Removed missing rows ({how_key}); "
                                             f"{preview['rows_removed']} rows removed")
                    show_change_summary(preview["rows_before"], len(new_df),
                                         missing_before, missing_after)
                    st.rerun()

        # ---------------------------------------------------------------
        # Duplicates
        # ---------------------------------------------------------------
        with tabs[1]:
            st.subheader("Duplicate Records")
            use_subset_dup = st.checkbox("Detect duplicates using specific columns only",
                                          key="dup_subset_chk")
            dup_subset = None
            if use_subset_dup:
                dup_subset = st.multiselect("Columns to check for duplicates",
                                             df.columns.tolist(), key="dup_subset_cols")

            dstats = dc.get_duplicate_stats(df, subset=dup_subset or None)
            c1, c2 = st.columns(2)
            c1.metric("Duplicate Rows", dstats["n_duplicate_rows"])
            c2.metric("Unique Rows", dstats["n_unique_rows"])

            keep = st.radio("Which occurrence to keep?", ["First", "Last"])
            keep_key = keep.lower()

            preview_dup = dc.preview_remove_duplicates(df, subset=dup_subset or None, keep=keep_key)
            st.info(f"Rows before: {preview_dup['rows_before']:,} | "
                    f"Rows that will be removed: {preview_dup['rows_removed']:,} | "
                    f"Rows remaining: {preview_dup['rows_after']:,}")

            if st.button("Apply: Remove Duplicates", type="primary"):
                if use_subset_dup and not dup_subset:
                    st.error("Please select at least one column, or uncheck the subset option.")
                else:
                    new_df = dc.remove_duplicates(df, subset=dup_subset or None, keep=keep_key)
                    set_working_df(new_df, f"Removed {preview_dup['rows_removed']} duplicate rows "
                                             f"(keep={keep_key})")
                    show_change_summary(preview_dup["rows_before"], len(new_df))
                    st.rerun()

        # ---------------------------------------------------------------
        # Object / string fill
        # ---------------------------------------------------------------
        with tabs[2]:
            st.subheader("Missing Value Treatment — Object/String Columns")
            obj_cols = get_categorical_columns(df)
            if not obj_cols:
                st.info("No object/string columns found in the current dataset.")
            else:
                selection_mode = st.radio("Apply to", ["Single column", "Multiple columns", "All eligible columns"],
                                           key="obj_selmode")
                if selection_mode == "Single column":
                    chosen = st.selectbox("Column", obj_cols)
                    columns_to_fill = [chosen]
                elif selection_mode == "Multiple columns":
                    columns_to_fill = st.multiselect("Columns", obj_cols)
                else:
                    columns_to_fill = obj_cols
                    st.caption(f"Will apply to: {', '.join(obj_cols)}")

                method_label = st.selectbox("Fill method", [
                    "Custom value", '"Unknown"', '"Not Available"', '"Missing"', "Mode",
                ])
                method_map = {
                    "Custom value": "custom", '"Unknown"': "unknown",
                    '"Not Available"': "not_available", '"Missing"': "missing", "Mode": "mode",
                }
                method_key = method_map[method_label]

                custom_value = None
                if method_key == "custom":
                    custom_value = st.text_input("Custom fill value", value="")

                missing_before = sum(int(df[c].isna().sum()) for c in columns_to_fill) if columns_to_fill else 0
                st.caption(f"Missing values in selected column(s): {missing_before:,}")

                if st.button("Apply: Fill Missing Values", type="primary", key="obj_fill_btn"):
                    if not columns_to_fill:
                        st.error("Please select at least one column.")
                    else:
                        new_df, stats = dc.fill_object_missing(
                            df, columns_to_fill, method_key, custom_value=custom_value)
                        set_working_df(
                            new_df,
                            f"Filled {stats['total_filled']} missing value(s) in "
                            f"{', '.join(columns_to_fill)} using '{method_label}'"
                        )
                        st.success(f"Operation completed — {stats['total_filled']} value(s) filled.\n\n"
                                   + "\n".join(f"- {c}: {n} filled" for c, n in stats["per_column"].items()))
                        st.rerun()

        # ---------------------------------------------------------------
        # Numeric fill
        # ---------------------------------------------------------------
        with tabs[3]:
            st.subheader("Missing Value Treatment — Numeric Columns")
            num_cols = get_numeric_columns(df)
            if not num_cols:
                st.info("No numeric columns found in the current dataset.")
            else:
                num_selection_mode = st.radio("Apply to", ["Single column", "Multiple columns"],
                                               key="num_selmode")
                if num_selection_mode == "Single column":
                    chosen_num = st.selectbox("Column", num_cols, key="num_single")
                    num_columns_to_fill = [chosen_num]
                else:
                    num_columns_to_fill = st.multiselect("Columns", num_cols, key="num_multi")

                strategy_label = st.selectbox("Strategy", [
                    "Forward Fill (ffill)", "Backward Fill (bfill)", "Interpolation",
                ])
                strategy_map = {
                    "Forward Fill (ffill)": "ffill",
                    "Backward Fill (bfill)": "bfill",
                    "Interpolation": "interpolate",
                }
                strategy_key = strategy_map[strategy_label]

                missing_before_num = sum(int(df[c].isna().sum()) for c in num_columns_to_fill) \
                    if num_columns_to_fill else 0
                st.caption(f"Missing values before: {missing_before_num:,}")

                if st.button("Apply: Fill Numeric Missing Values", type="primary", key="num_fill_btn"):
                    if not num_columns_to_fill:
                        st.error("Please select at least one numeric column.")
                    else:
                        new_df, stats = dc.fill_numeric_missing(df, num_columns_to_fill, strategy_key)
                        missing_after_num = sum(int(new_df[c].isna().sum()) for c in num_columns_to_fill)
                        set_working_df(
                            new_df,
                            f"Filled {stats['total_filled']} numeric missing value(s) in "
                            f"{', '.join(num_columns_to_fill)} using '{strategy_label}'"
                        )
                        show_change_summary(len(df), len(new_df), missing_before_num, missing_after_num)
                        st.rerun()

        # ---------------------------------------------------------------
        # Type conversion (numeric + datetime)
        # ---------------------------------------------------------------
        with tabs[4]:
            st.subheader("Convert Column Types")

            st.markdown("#### Convert to Numeric")
            candidate_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
            if candidate_cols:
                numeric_targets = st.multiselect("Columns to convert to numeric", candidate_cols)
                if numeric_targets:
                    st.dataframe(df[numeric_targets].head(5), use_container_width=True)
                if st.button("Convert Selected Columns to Numeric"):
                    if not numeric_targets:
                        st.error("Please select at least one column.")
                    else:
                        new_df, report = dc.convert_columns_to_numeric(df, numeric_targets)
                        set_working_df(new_df, f"Converted to numeric: {', '.join(numeric_targets)}")
                        lines = [f"- {c}: {r['n_invalid_coerced_to_nan']} value(s) could not be "
                                 f"converted and were set to missing" for c, r in report.items()]
                        st.success("Operation completed\n\n" + "\n".join(lines))
                        st.rerun()
            else:
                st.info("All columns are already numeric.")

            st.markdown("---")
            st.markdown("#### Convert to Datetime")
            dt_candidates = try_parse_datetime_columns(df)
            existing_dt = get_datetime_columns(df)
            if existing_dt:
                st.caption(f"Already datetime: {', '.join(existing_dt)}")

            eligible = [c for c in df.columns if c not in existing_dt]
            if dt_candidates:
                st.info(f"These column(s) look like they could be dates: {', '.join(dt_candidates)}")

            date_col = st.selectbox("Select a column to convert to datetime",
                                     ["-- select --"] + eligible)
            if date_col != "-- select --":
                if st.button("Convert to Datetime"):
                    try:
                        new_df, report = dc.convert_column_to_datetime(df, date_col)
                        set_working_df(new_df, f"Converted '{date_col}' to datetime")
                        st.success(
                            "Operation completed\n\n"
                            f"Values successfully parsed: {report['after_non_null']:,}\n"
                            f"Values that failed to parse (set to missing): "
                            f"{report['n_failed_conversions']:,}"
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not convert column to datetime: {e}")


# =============================================================================
# PAGE: Ask Questions (Natural Language Query Engine)
# =============================================================================
elif page == "Ask Questions":
    st.title("💬 Ask Questions About Your Data")
    st.caption("This uses a local, deterministic parser — no external AI/LLM or API calls are made.")

    if not has_data():
        st.warning("No dataset loaded. Please upload a dataset first.")
    else:
        df = st.session_state.working_df

        with st.expander("Example questions you can ask"):
            st.markdown(
                "- sales by region\n- region per sales\n- best selling products\n"
                "- top 10 products by sales\n- average sales by region\n"
                "- total sales by category\n- count of customers by region\n"
                "- maximum sales by product\n- minimum sales by product\n"
                "- show sales by month\n- which region has the highest sales?"
            )

        query = st.text_input("Type your question", value=st.session_state.last_query_text,
                               placeholder="e.g. total sales by region")

        run = st.button("Run Query", type="primary")

        if run and query.strip():
            st.session_state.last_query_text = query
            result = qe.parse_and_run_query(df, query)
            st.session_state.last_query_result = result

        result: qe.QueryResult | None = st.session_state.last_query_result

        if result is not None:
            if result.success:
                st.markdown("#### Interpretation")
                interp_lines = []
                if "group_by" in result.interpretation:
                    interp_lines.append(f"- **Group by:** {result.interpretation['group_by']}")
                if "metric" in result.interpretation:
                    interp_lines.append(f"- **Metric:** {result.interpretation['metric']}")
                if "operation" in result.interpretation:
                    interp_lines.append(f"- **Operation:** {result.interpretation['operation']}")
                if result.interpretation.get("time_frequency"):
                    interp_lines.append(f"- **Time grouping:** {result.interpretation['time_frequency']}")
                if result.interpretation.get("sort"):
                    interp_lines.append(f"- **Sort:** {result.interpretation['sort']}")
                if result.interpretation.get("limit"):
                    interp_lines.append(f"- **Limit:** {result.interpretation['limit']}")
                if result.interpretation.get("filter"):
                    f_ = result.interpretation["filter"]
                    interp_lines.append(f"- **Filter:** {f_['column']} {f_['operator']} {f_['value']}")
                if result.interpretation.get("note"):
                    interp_lines.append(f"- **Note:** {result.interpretation['note']}")
                st.markdown("\n".join(interp_lines))

                st.markdown("#### Result")
                st.dataframe(result.result_df, use_container_width=True)

                if len(result.result_df) > 0 and len(result.result_df.columns) >= 2:
                    if st.checkbox("📊 Visualize Result", value=False, key="viz_query_result"):
                        chart_suggestion = result.suggested_chart or {"type": "bar",
                                                                        "x": result.result_df.columns[0],
                                                                        "y": result.result_df.columns[1]}
                        try:
                            if chart_suggestion["type"] == "line":
                                fig = viz.build_plotly_chart(
                                    result.result_df, "Line Chart",
                                    x=chart_suggestion["x"], y=chart_suggestion["y"])
                            else:
                                fig = viz.build_plotly_chart(
                                    result.result_df, "Interactive Bar Chart",
                                    x=chart_suggestion["x"], y=chart_suggestion["y"])
                            st.plotly_chart(fig, use_container_width=True)
                        except viz.ChartValidationError as e:
                            st.warning(f"Could not auto-generate a chart: {e}")
            else:
                st.warning(result.message)
                if result.needs_column_selection:
                    st.markdown("**Select columns manually:**")
                    c1, c2 = st.columns(2)
                    manual_group = c1.selectbox(
                        "Group by column",
                        ["-- none --"] + result.candidate_group_cols, key="manual_group")
                    manual_metric = c2.selectbox(
                        "Metric column",
                        ["-- none --"] + result.candidate_metric_cols, key="manual_metric")
                    manual_agg = st.selectbox("Aggregation", ["sum", "mean", "count", "min", "max"],
                                               key="manual_agg")
                    if st.button("Run Manual Query"):
                        if manual_group == "-- none --" or manual_metric == "-- none --":
                            st.error("Please select both a group-by column and a metric column.")
                        else:
                            try:
                                if manual_agg == "count":
                                    grouped = df.groupby(manual_group).size().reset_index(name="Count")
                                else:
                                    grouped = df.groupby(manual_group)[manual_metric].agg(manual_agg).reset_index()
                                    grouped.columns = [manual_group, f"{manual_agg.title()} of {manual_metric}"]
                                grouped = grouped.sort_values(by=grouped.columns[1], ascending=False).reset_index(drop=True)
                                st.dataframe(grouped, use_container_width=True)
                            except Exception as e:
                                st.error(f"Could not execute manual query: {e}")


# =============================================================================
# PAGE: Visualization
# =============================================================================
elif page == "Visualization":
    st.title("📊 Visualization")

    if not has_data():
        st.warning("No dataset loaded. Please upload a dataset first.")
    else:
        df = st.session_state.working_df

        col_lib, col_type = st.columns(2)
        library = col_lib.selectbox("Chart library", ["Matplotlib", "Seaborn", "Plotly"])
        chart_type = col_type.selectbox("Chart type", viz.CHART_TYPES_BY_LIBRARY[library])

        requirements = viz.CHART_FIELD_REQUIREMENTS.get(chart_type, {"x": True, "y": True})

        all_cols = df.columns.tolist()
        numeric_cols = get_numeric_columns(df)

        col_x, col_y = st.columns(2)
        x_col = None
        y_col = None
        if requirements.get("x"):
            x_col = col_x.selectbox("X-axis", ["-- select --"] + all_cols)
            x_col = None if x_col == "-- select --" else x_col
        if requirements.get("y"):
            y_col = col_y.selectbox("Y-axis", ["-- select --"] + all_cols)
            y_col = None if y_col == "-- select --" else y_col

        needs_agg = chart_type in ("Bar Chart", "Bar Plot", "Interactive Bar Chart",
                                     "Pie Chart", "Pie / Donut Chart", "Line Chart")
        agg = None
        if needs_agg:
            agg_label = st.selectbox("Aggregation", ["Sum", "Mean", "Count", "Min", "Max"])
            agg = agg_label.lower()

        with st.expander("Optional settings"):
            title = st.text_input("Chart title", value="")
            sort_label = st.selectbox("Sort", ["None", "Descending", "Ascending"])
            sort = None if sort_label == "None" else sort_label.lower()
            use_top_n = st.checkbox("Limit to Top N categories")
            top_n = st.number_input("Top N", min_value=1, max_value=1000, value=10) if use_top_n else None
            c_w, c_h = st.columns(2)
            fig_width = c_w.slider("Figure width", 4, 16, 8)
            fig_height = c_h.slider("Figure height", 3, 12, 5)

        if st.button("Generate Chart", type="primary"):
            try:
                if library == "Matplotlib":
                    fig = viz.build_matplotlib_chart(df, chart_type, x_col, y_col, agg, title,
                                                       sort, top_n, fig_width, fig_height)
                    st.pyplot(fig)
                elif library == "Seaborn":
                    fig = viz.build_seaborn_chart(df, chart_type, x_col, y_col, agg, title,
                                                    sort, top_n, fig_width, fig_height)
                    st.pyplot(fig)
                else:
                    fig = viz.build_plotly_chart(df, chart_type, x_col, y_col, agg, title,
                                                   sort, top_n, height=fig_height * 80)
                    st.plotly_chart(fig, use_container_width=True)
            except viz.ChartValidationError as e:
                st.error(f"Chart configuration issue: {e}")
            except Exception as e:
                st.error("An unexpected error occurred while generating the chart.")
                with st.expander("Technical details"):
                    st.code(traceback.format_exc())


# =============================================================================
# PAGE: Download
# =============================================================================
elif page == "Download":
    st.title("⬇️ Download Cleaned Dataset")

    if not has_data():
        st.warning("No dataset loaded. Please upload a dataset first.")
    else:
        df = st.session_state.working_df
        st.write(f"Current dataset: **{df.shape[0]:,} rows × {df.shape[1]:,} columns**")
        st.dataframe(df.head(10), use_container_width=True)

        c1, c2 = st.columns(2)

        with c1:
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download as CSV",
                data=csv_bytes,
                file_name="cleaned_dataset.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with c2:
            excel_buffer = io.BytesIO()
            try:
                with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                    df.to_excel(writer, index=False, sheet_name="Cleaned Data")
                excel_bytes = excel_buffer.getvalue()
                st.download_button(
                    label="⬇️ Download as Excel",
                    data=excel_bytes,
                    file_name="cleaned_dataset.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Could not generate Excel file: {e}")
