"""Waterbend Case Study — Streamlit dashboard."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Waterbend Case Study", layout="wide")


@st.cache_data
def load_data():
    base = Path(__file__).parent / "data"
    return (
        pd.read_pickle(base / "profitability.pkl"),
        pd.read_pickle(base / "merged.pkl"),
    )


profitability, merged = load_data()

METRIC_LABELS = {
    "revenue": "Limited-service revenue ($)",
    "menu_price_index": "Limited-Service CPI",
    "implied_volume_index": "Implied traffic index",
    "food_input_index": "Food input price index",
    "food_cost_pct_revenue": "Food cost % of revenue",
    "labor_cost_pct_revenue": "Labor cost % of revenue",
    "restaurant_profit": "Restaurant profit ($)",
    "profit_margin": "Profit margin",
}

PCT_METRICS = {"profit_margin", "food_cost_pct_revenue", "labor_cost_pct_revenue"}
DOLLAR_METRICS = {"revenue", "restaurant_profit"}


def _fmt_raw(metric: str) -> str:
    if metric in PCT_METRICS:
        return "{:.1%}"
    if metric in DOLLAR_METRICS:
        return "${:,.0f}"
    return "{:.1f}"


MACRO_SOURCES_MD = """\
| Metric | Source | Definition / methodology |
|---|---|---|
| Limited-Service Revenue | BLS series **SM72251XUSN** | Monthly retail sales for limited-service eating places. Category covers fast-food, QSR, and sandwich-style concepts (NAICS 722513). Likely excludes "Snack and Nonalcoholic Beverage Bars" (NAICS 722515) where coffee shops sit — so BROS-style concepts are benchmarked against a partial-fit category. |
| Limited-Service CPI | BLS CPI series **CUUR0000SEFV02** | "Limited service meals and snacks", US city avg, not seasonally adjusted. Tracks the price consumers pay at the register; excludes mix-shift effects (such as LTOs) that companies report inside their own ticket calc. |
| Implied Traffic Index | Derived | Limited-Service Revenue (indexed) ÷ Limited-Service CPI × 100. Real volume of transactions implied by dividing the limited service revenue by the limited service CPI. |
| Food Input Price Index | FRED / BLS PPI series **PCU311311** | Producer Price Index by Industry: Food Manufacturing — wholesale prices manufacturers charge for processed foods. An industry wide index. |
| Food Cost % Revenue | Derived | Anchored at a 30% starting food-cost ratio that drifts with a food-pressure factor built from Implied Traffic × Food Input Price relative to Revenue (all indexed). Not a measured industry quantity. |
| Labor Cost % Revenue | Derived | Labor cost ÷ Revenue, where labor cost is BLS **QCEW** total quarterly wages for NAICS 722513, monthly-split by QCEW headcount. Captures industry-wide wage and headcount pressure but NOT company-specific scheduling, tip-credit, or per-store productivity. |
| Restaurant Profit | Derived | Revenue × 0.80 − Food Cost − Labor Cost. Industry-wide four-wall economics implied by the macros above. Analogous to public-co "restaurant-level profit" but constructed top-down; the 0.80 reflects a flat 20% allowance for occupancy / marketing / utilities / supplies that the app deliberately does not chart on its own. |
| Profit Margin | Derived | Restaurant Profit ÷ Revenue. |
"""


ESTIMATE_SOURCES_MD = """\
| Estimate | Source | Blindspots |
|---|---|---|
| Restaurant sales estimate | BLS **SM72251XUSN** — limited-service eating places retail sales | Data only captures macro dynamics across limited-service restaurants, with no further visibility into company-specific dynamics (such as LTOs, regional biases, etc.) |
| Food estimate | FRED PPI per ticker: **BROS = Coffee, whole bean / ground / instant** · **SHAK = Beef and Veal, Fresh or Frozen** · **CAVA = Processed Foods and Feeds: Meats, Poultry, and Fish** | The index only reflects the dominant commodity for each ticker, not the full mix of inputs. We only include the producer price, which does not reflect the quantity of food produced. Changes in quantity purchased would thus lead to an under/overcapture of actuals. The producer price may reflect a transaction made further upstream from the company (eg: from the original farmer and a wholesaler). Does not reflect company specific practices to avoid cost inflation, including but not limited to: hedging, negotiated deals with suppliers, etc. |
| Labor estimate | BLS QCEW (limited service restaurants only) average weekly wage per state, shop-weighted by the ticker's state footprint (from 10-K store-count maps in the pipeline) | Only captures wage and does not account for hours worked. Does not consider different locations having a different number of employees, which could impact the mix differently from a store-weighted mix. Data comes in on a longer lag, which means more quarters need to be estimated. |
| Profit estimate | The difference between restaurant sales estimate, food estimate, labor estimate, and other operating expenses estimate. (Other operating expense estimate assumes that the previous 4Q y/y growth rate remains constant in the next quarter) | See above. |
"""


def render_macro_sources() -> None:
    with st.expander("Data sources & methodology", expanded=True):
        st.markdown(MACRO_SOURCES_MD)


def render_estimate_sources() -> None:
    with st.expander("Estimate sources & methodology", expanded=True):
        st.markdown(ESTIMATE_SOURCES_MD)


def _annual_aggregates(prof: pd.DataFrame, n_months: int | None = None) -> pd.DataFrame:
    """Aggregate monthly profitability to annual values.

    If n_months is None, only years with a full 12 months of data are kept.
    If n_months is set, aggregate the first n_months of each year and keep
    only years that have exactly that many rows (used for partial-year YoY
    comparables: e.g. Jan-Mar 2026 vs Jan-Mar 2025).

    Aggregation rules: SUM nominal ($), AVG raw indexes, then RECOMPUTE the
    derived metrics (implied traffic, food/labor cost %, restaurant profit,
    profit margin) from the aggregates.
    """
    expected = 12 if n_months is None else n_months
    p = prof.copy()
    p["date"] = pd.to_datetime(p["date"])
    p["year"] = p["date"].dt.year.astype(int)
    p["month"] = p["date"].dt.month.astype(int)
    if n_months is not None:
        p = p[p["month"] <= n_months]

    grouped = p.groupby("year")
    agg = pd.DataFrame({
        "revenue": grouped["revenue"].sum(min_count=expected),
        "menu_price_index": grouped["menu_price_index"].mean(),
        "food_input_index": grouped["food_input_index"].mean(),
        "labor_cost": grouped["labor_cost"].sum(min_count=expected),
        "_count": grouped.size(),
    })
    agg = agg[agg["_count"] == expected].drop(columns=["_count"])
    if agg.empty:
        return agg

    first = agg.iloc[0]
    agg["revenue_index"] = agg["revenue"] / first["revenue"] * 100
    rebased_menu = agg["menu_price_index"] / first["menu_price_index"] * 100
    rebased_food = agg["food_input_index"] / first["food_input_index"] * 100
    agg["implied_volume_index"] = agg["revenue_index"] / rebased_menu * 100
    food_pressure = (agg["implied_volume_index"] / 100) * (rebased_food / 100) / (agg["revenue_index"] / 100)
    agg["food_cost_pct_revenue"] = 0.30 * food_pressure
    agg["food_cost"] = agg["revenue"] * agg["food_cost_pct_revenue"]
    agg["labor_cost_pct_revenue"] = agg["labor_cost"] / agg["revenue"]
    agg["other_opex"] = agg["revenue"] * 0.20
    agg["restaurant_profit"] = agg["revenue"] - agg["food_cost"] - agg["labor_cost"] - agg["other_opex"]
    agg["profit_margin"] = agg["restaurant_profit"] / agg["revenue"]
    return agg


def _render_monthly_table(prof: pd.DataFrame, view: str) -> None:
    st.subheader("Macro metrics — last 36 months")
    suffix = {"Raw values": "", "YoY growth": "_yoy", "2Y CAGR": "_2y_cagr"}[view]
    metrics = list(METRIC_LABELS.keys())

    # *_pct_revenue series aren't in the pre-computed growth_cols, so derive
    # their YoY and 2Y CAGR on the fly.
    prof = prof.sort_values("date").copy()
    for k in ("food_cost_pct_revenue", "labor_cost_pct_revenue"):
        prof[k + "_yoy"] = prof[k] / prof[k].shift(12) - 1
        prof[k + "_2y_cagr"] = (prof[k] / prof[k].shift(24)) ** 0.5 - 1

    cols = [m + suffix for m in metrics]
    base = (
        prof.set_index("date")[cols]
        .sort_index(ascending=True)
        .tail(36)
    )
    base.columns = [METRIC_LABELS[m] for m in metrics]
    table = base.T
    table.columns = [d.strftime("%b %Y") for d in table.columns]

    if suffix:
        styler = table.style.format("{:.1%}", na_rep="—")
        max_abs = float(table.abs().to_numpy().max())
        if max_abs > 0 and not pd.isna(max_abs):
            styler = styler.background_gradient(
                cmap="RdYlGn", vmin=-max_abs, vmax=max_abs, axis=None,
            )
        st.dataframe(styler, width="stretch")
    else:
        formatted = pd.DataFrame(
            index=table.index, columns=table.columns, dtype=object,
        )
        for m in metrics:
            label = METRIC_LABELS[m]
            fmt = _fmt_raw(m)
            formatted.loc[label] = [
                fmt.format(v) if pd.notna(v) else "—" for v in table.loc[label]
            ]
        st.dataframe(formatted, width="stretch")


def _render_annual_table(prof: pd.DataFrame, view: str) -> None:
    st.subheader("Macro metrics — annual")
    metrics = list(METRIC_LABELS.keys())

    full = _annual_aggregates(prof)
    p = prof.copy()
    p["date"] = pd.to_datetime(p["date"])
    p["year"] = p["date"].dt.year.astype(int)
    latest_year = int(p["year"].max())
    n_partial = int((p["year"] == latest_year).sum())
    has_partial = n_partial < 12

    if view == "Raw values":
        if full.empty:
            st.info("No complete years available for annual aggregation.")
            return
        base = full[metrics].copy()
        base.index = [str(y) for y in base.index]
        table = base.T
        table.index = [METRIC_LABELS[m] for m in metrics]
        formatted = pd.DataFrame(
            index=table.index, columns=table.columns, dtype=object,
        )
        for m in metrics:
            label = METRIC_LABELS[m]
            fmt = _fmt_raw(m)
            formatted.loc[label] = [
                fmt.format(v) if pd.notna(v) else "—" for v in table.loc[label]
            ]
        st.dataframe(formatted, width="stretch")
        return

    # Growth view (YoY or 2Y CAGR)
    shift_n, power = (1, 1.0) if view == "YoY growth" else (2, 0.5)

    if full.empty:
        st.info("No complete years available for annual aggregation.")
        return

    growth = pd.DataFrame(index=full.index.astype(int), columns=metrics, dtype=float)
    for m in metrics:
        growth[m] = (full[m] / full[m].shift(shift_n)) ** power - 1

    if has_partial:
        partial = _annual_aggregates(prof, n_months=n_partial)
        prior_year = latest_year - shift_n
        if (
            not partial.empty
            and latest_year in partial.index
            and prior_year in partial.index
        ):
            partial_growth = {}
            for m in metrics:
                partial_growth[m] = (partial.loc[latest_year, m] / partial.loc[prior_year, m]) ** power - 1
            partial_row = pd.DataFrame([partial_growth], index=[latest_year])
            growth = pd.concat([growth, partial_row])

    growth = growth.dropna(how="all")
    if growth.empty:
        st.info("Not enough years of history to compute growth.")
        return

    year_labels = [
        f"{y} ({n_partial}mo)" if (has_partial and y == latest_year) else str(y)
        for y in growth.index
    ]
    table = growth[metrics].T
    table.columns = year_labels
    table.index = [METRIC_LABELS[m] for m in metrics]

    styler = table.style.format("{:.1%}", na_rep="—")
    max_abs = float(table.abs().to_numpy().max())
    if max_abs > 0 and not pd.isna(max_abs):
        styler = styler.background_gradient(
            cmap="RdYlGn", vmin=-max_abs, vmax=max_abs, axis=None,
        )
    st.dataframe(styler, width="stretch")


def render_macro_table(prof: pd.DataFrame, frequency: str) -> None:
    view = st.radio(
        "View", ["Raw values", "YoY growth", "2Y CAGR"],
        horizontal=True, key="macro_view",
    )
    if frequency == "Monthly":
        _render_monthly_table(prof, view)
    else:
        _render_annual_table(prof, view)


def _base_layout(fig: go.Figure, title: str, y_pct: bool = False) -> go.Figure:
    fig.update_layout(
        title=title,
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", y=-0.2),
        hovermode="x unified",
    )
    if y_pct:
        fig.update_yaxes(tickformat=".1%")
    return fig


def _multi_line(df: pd.DataFrame, x: str, y_cols: dict[str, str],
                title: str, y_pct: bool, zero_ref: bool = False) -> go.Figure:
    fig = go.Figure()
    for col, label in y_cols.items():
        fig.add_trace(go.Scatter(
            x=df[x], y=df[col], mode="lines", name=label,
        ))
    if zero_ref:
        fig.add_hline(y=0, line_dash="dot", line_color="#888")
    return _base_layout(fig, title, y_pct=y_pct)


def _year_overlay(df: pd.DataFrame, value_col: str, title: str,
                  y_pct: bool = False) -> go.Figure:
    d = df[["date", value_col]].dropna().copy()
    d["year"] = d["date"].dt.year.astype(int)
    d["month"] = d["date"].dt.month.astype(int)
    fig = px.line(
        d, x="month", y=value_col, color="year",
        markers=True,
        color_discrete_sequence=px.colors.sequential.Viridis,
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(1, 13)),
        ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        title_text="",
    )
    fig.update_yaxes(title_text="")
    fig.update_layout(legend_title_text="Year")
    return _base_layout(fig, title, y_pct=y_pct)


def render_macro_charts(prof: pd.DataFrame, frequency: str) -> None:
    st.subheader("Macro charts")
    growth_view = st.radio(
        "Growth view", ["YoY", "2Y CAGR"],
        horizontal=True, key="grow_view",
    )
    growth_suffix = "_yoy" if growth_view == "YoY" else "_2y_cagr"
    growth_power = 1.0 if growth_view == "YoY" else 0.5
    growth_label = growth_view  # "YoY" or "2Y CAGR"
    is_annual = frequency == "Annual"

    # Build the chart dataframe + the x-axis column for the chosen frequency.
    if is_annual:
        annual = _annual_aggregates(prof)
        for col in ("revenue", "menu_price_index", "implied_volume_index",
                    "food_input_index", "food_cost_pct_revenue",
                    "labor_cost_pct_revenue", "profit_margin"):
            annual[col + "_yoy"] = annual[col] / annual[col].shift(1) - 1
            annual[col + "_2y_cagr"] = (annual[col] / annual[col].shift(2)) ** 0.5 - 1
        chart_df = annual.reset_index()  # year becomes a column
        x_col = "year"
        growth_lag = 1
    else:
        chart_df = prof.sort_values("date").copy()
        x_col = "date"
        growth_lag = 12 if growth_view == "YoY" else 24

    # Row 1 — Industry revenue / Implied traffic / Limited-Service CPI
    st.markdown("**Industry revenue, implied traffic, and menu price**")
    c1, c2 = st.columns(2)
    with c1:
        fig = _multi_line(
            chart_df, x_col,
            {
                f"revenue{growth_suffix}": f"Limited-service revenue {growth_label}",
                f"implied_volume_index{growth_suffix}": f"Implied traffic index {growth_label}",
                f"menu_price_index{growth_suffix}": f"Limited-Service CPI {growth_label}",
            },
            title=f"{growth_label} growth — revenue, implied traffic, menu price",
            y_pct=True, zero_ref=True,
        )
        st.plotly_chart(fig, width="stretch")
    with c2:
        # Year-overlay is a monthly seasonality view — always render against the
        # raw monthly profitability dataframe even when Frequency is Annual.
        choice = st.radio(
            "Series",
            ["Industry revenue", "Implied traffic index", "Limited-Service CPI"],
            horizontal=True, key="grp1_series",
        )
        series_col, year_overlay_title = {
            "Industry revenue": ("revenue", "Limited-service revenue by year"),
            "Implied traffic index": ("implied_volume_index", "Implied traffic index by year"),
            "Limited-Service CPI": ("menu_price_index", "Limited-Service CPI by year"),
        }[choice]
        st.plotly_chart(_year_overlay(prof, series_col, year_overlay_title),
                        width="stretch")

    # Row 2 — Food + Labor as % of revenue
    st.markdown("**Food + Labor as % of revenue**")
    c1, c2 = st.columns(2)
    cost_series = {
        "food_cost_pct_revenue": "Food cost %",
        "labor_cost_pct_revenue": "Labor cost %",
    }
    with c1:
        st.plotly_chart(
            _multi_line(chart_df, x_col, cost_series,
                        title="Food + Labor as % of revenue",
                        y_pct=True),
            width="stretch",
        )
    with c2:
        d = chart_df[[x_col]].copy()
        for k in cost_series:
            d[k + "_grow"] = (chart_df[k] / chart_df[k].shift(growth_lag)) ** growth_power - 1
        grow_series = {k + "_grow": v + f" {growth_label}" for k, v in cost_series.items()}
        st.plotly_chart(
            _multi_line(d, x_col, grow_series,
                        title=f"Food + Labor as % of revenue ({growth_label})",
                        y_pct=True, zero_ref=True),
            width="stretch",
        )

    # Row 3 — Profit margin
    st.markdown("**Profit margin**")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            _multi_line(chart_df, x_col, {"profit_margin": "Profit margin"},
                        title="Profit margin",
                        y_pct=True),
            width="stretch",
        )
    with c2:
        st.plotly_chart(
            _multi_line(chart_df, x_col,
                        {f"profit_margin{growth_suffix}": f"Profit margin {growth_label}"},
                        title=f"Profit margin {growth_label} change",
                        y_pct=True, zero_ref=True),
            width="stretch",
        )


ESTIMATE_MARKER = dict(size=11, symbol="circle", color="#888")


def _actual_vs_estimate_fig(df: pd.DataFrame, actual_col: str, estimate_col: str,
                            title: str, y_title: str,
                            hover_value_fmt: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Period"], y=df[actual_col],
        mode="lines+markers", name="Actual",
        line=dict(width=2),
        hovertemplate=f"%{{x}}<br>{hover_value_fmt}<extra>Actual</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["Period"], y=df[estimate_col],
        mode="markers", name="Estimate",
        marker=ESTIMATE_MARKER,
        hovertemplate=f"%{{x}}<br>{hover_value_fmt}<extra>Estimate</extra>",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Period",
        yaxis_title=y_title,
        height=360,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", y=1.1),
        hovermode="x unified",
    )
    return fig


def actual_vs_estimate(df: pd.DataFrame, actual_col: str, estimate_col: str,
                       title: str, y_title: str) -> go.Figure:
    return _actual_vs_estimate_fig(
        df.sort_values("Period Start"),
        actual_col, estimate_col, title, y_title,
        hover_value_fmt="$%{y:,.0f}K",
    )


def actual_vs_estimate_pct(df: pd.DataFrame, actual_col: str, estimate_col: str,
                           title: str) -> go.Figure:
    fig = _actual_vs_estimate_fig(
        df.sort_values("Period Start"),
        actual_col, estimate_col, title, "YoY change",
        hover_value_fmt="%{y:.1%}",
    )
    fig.update_yaxes(tickformat=".1%")
    fig.add_hline(y=0, line_dash="dot", line_color="#888")
    return fig


def render_estimate_charts(df: pd.DataFrame, ticker: str) -> None:
    if df.empty:
        st.info(f"No data for {ticker}.")
        return

    df = df.sort_values("Period Start").copy()

    charts = [
        ("Restaurant-Level Profit ($K)", "Profit Estimate",
         f"{ticker} — Restaurant-Level Profit ($K)"),
        ("Restaurant Sales ($K)", "estimate_rev",
         f"{ticker} — Revenue ($K)"),
        ("Food & Distribution ($K)", "estimate_food",
         f"{ticker} — Food & Distribution ($K)"),
        ("Labor ($K)", "estimate_labor",
         f"{ticker} — Labor ($K)"),
    ]

    # Pre-compute YoY columns: actual_yoy = actual / actual.shift(4) - 1,
    # estimate_yoy = estimate / actual.shift(4) - 1 (TY estimate / LY actual - 1).
    for actual_col, est_col, _ in charts:
        lag4 = df[actual_col].shift(4)
        df[f"_yoy_actual_{actual_col}"] = df[actual_col] / lag4 - 1
        df[f"_yoy_est_{est_col}"] = df[est_col] / lag4 - 1

    for i, (actual_col, est_col, title) in enumerate(charts):
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                actual_vs_estimate(df, actual_col, est_col, title, "$ thousands"),
                width="stretch",
            )
        with c2:
            st.plotly_chart(
                actual_vs_estimate_pct(
                    df,
                    f"_yoy_actual_{actual_col}",
                    f"_yoy_est_{est_col}",
                    f"{title} — YoY",
                ),
                width="stretch",
            )
        if i == 0 and "mape_4q" in df.columns:
            mape_vals = df["mape_4q"].dropna()
            if not mape_vals.empty:
                st.caption(f"Trailing 4Q MAPE: {mape_vals.iloc[-1]:.1%}")

    with st.expander("Estimate values & Y/Y growth", expanded=False):
        table_rows = [
            ("Restaurant Level Profit ($K)", "Profit Estimate", False),
            ("Restaurant Level Profit ($K) - Y/Y", "_yoy_est_Profit Estimate", True),
            ("Revenue ($K)", "estimate_rev", False),
            ("Revenue ($K) - Y/Y", "_yoy_est_estimate_rev", True),
            ("Food & Distribution ($K)", "estimate_food", False),
            ("Food & Distribution ($K) - Y/Y", "_yoy_est_estimate_food", True),
            ("Labor ($K)", "estimate_labor", False),
            ("Labor ($K) - Y/Y", "_yoy_est_estimate_labor", True),
        ]
        periods = df["Period"].tolist()
        table = pd.DataFrame(
            index=[r[0] for r in table_rows], columns=periods, dtype=object,
        )
        for label, col, is_pct in table_rows:
            vals = df[col].tolist()
            if is_pct:
                table.loc[label] = [f"{v:.1%}" if pd.notna(v) else "—" for v in vals]
            else:
                table.loc[label] = [f"${v:,.0f}" if pd.notna(v) else "—" for v in vals]
        st.dataframe(table, width="stretch")


tab_macro, tab_est = st.tabs(["Macro Analysis", "Modeled Estimates"])

with tab_macro:
    render_macro_sources()
    frequency = st.radio(
        "Frequency", ["Monthly", "Annual"],
        horizontal=True, key="macro_freq",
    )
    render_macro_table(profitability, frequency)
    st.divider()
    render_macro_charts(profitability, frequency)

with tab_est:
    render_estimate_sources()
    ticker = st.selectbox("Ticker", ["SHAK", "CAVA", "BROS"], key="ticker_select")
    render_estimate_charts(merged[merged["Ticker"] == ticker], ticker)
