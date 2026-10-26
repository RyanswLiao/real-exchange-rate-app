"""Monthly bilateral real exchange rates using the IMF SDMX 3.0 API."""
from __future__ import annotations

from io import StringIO
from datetime import date
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pycountry
import requests
import streamlit as st
from plotly.subplots import make_subplots

BASE = "https://api.imf.org/external/sdmx/3.0"
FIRST_MONTH = pd.Period("1976-01", freq="M")
CURRENT_MONTH = pd.Period(date.today(), freq="M")

st.set_page_config(page_title="IMF Real Exchange Rate Explorer", page_icon="🌍", layout="wide")


class IMFError(Exception):
    pass


def country_options():
    """ISO-3 candidates; the API verifies actual IMF coverage when queried."""
    result = {f"{c.name} ({c.alpha_3})": c.alpha_3 for c in pycountry.countries}
    result["Euro area (EA) [IMF regional aggregate]"] = "EA"
    return dict(sorted(result.items()))


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def get_imf_series(flow: str, key: str) -> pd.Series:
    """Get one IMF monthly series in SDMX-CSV; raise instead of fabricating data."""
    # IMF's current SDMX 3.0 dataflow resource accepts an ISO3-keyed series.
    paths = [
        f"{BASE}/data/dataflow/IMF.STA/{flow}/+/{key}",
        f"{BASE}/data/IMF.STA,{flow},+/{key}",
    ]
    last_error = None
    for url in paths:
        for accept in (
            "csvfile",
            "application/vnd.sdmx.data+csv;version=2.0.0",
        ):
            try:
                resp = requests.get(
                    url,
                    params={"startPeriod": str(FIRST_MONTH), "endPeriod": str(CURRENT_MONTH)},
                    headers={"Accept": accept},
                    timeout=35,
                )
                if resp.status_code == 204:
                    raise IMFError("IMF returned no observations for this series.")
                resp.raise_for_status()
                if not resp.text.lstrip().startswith(("STRUCTURE", "DATAFLOW", "KEY", "TIME_PERIOD", "FREQ", "COUNTRY")):
                    # Some servers return an error document with HTTP 200.
                    raise IMFError("IMF did not return a recognizable SDMX-CSV table.")
                data = pd.read_csv(StringIO(resp.text), low_memory=False)
                data.columns = [str(x).strip().upper() for x in data.columns]
                if not {"TIME_PERIOD", "OBS_VALUE"}.issubset(data.columns):
                    raise IMFError("The response does not include TIME_PERIOD and OBS_VALUE.")
                time_text = data["TIME_PERIOD"].astype(str)
                time_text = time_text.str.replace(r"^(\d{4})-M(\d{1,2})$", lambda m: f"{m.group(1)}-{int(m.group(2)):02d}", regex=True)
                periods = pd.to_datetime(time_text, errors="coerce").dt.to_period("M")
                values = pd.to_numeric(data["OBS_VALUE"], errors="coerce")
                result = pd.Series(values.to_numpy(), index=pd.PeriodIndex(periods, freq="M"), dtype=float)
                result = result.loc[~result.index.isna()].dropna()
                result = result.loc[(result.index >= FIRST_MONTH) & (result.index <= CURRENT_MONTH)]
                if result.empty:
                    raise IMFError("The IMF series has no numeric monthly observations in the requested interval.")
                if result.index.has_duplicates:
                    result = result.groupby(level=0).last()
                return result.sort_index()
            except (requests.RequestException, ValueError, IMFError, pd.errors.ParserError) as exc:
                last_error = exc
    raise IMFError(f"Unable to retrieve {flow} / {key}. Last response: {last_error}")


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def cpi(country: str):
    # CPI level, all items, monthly (NOT a growth rate).
    return get_imf_series("CPI", f"{country}.CPI._T.IX.M")


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def usd_per_local(country: str):
    """USD per unit of local currency, using monthly-period-average FX."""
    if country == "USA":
        return pd.Series(1.0, index=pd.period_range(FIRST_MONTH, CURRENT_MONTH, freq="M"))
    # Direct IMF USD_XDC quote; inverse IMF XDC_USD used only if needed.
    try:
        x = get_imf_series("ER", f"{country}.USD_XDC.PA_RT.M")
        return x.where(x > 0).dropna()
    except IMFError as first_error:
        try:
            y = get_imf_series("ER", f"{country}.XDC_USD.PA_RT.M")
            return (1 / y.where(y > 0)).dropna()
        except IMFError as second_error:
            raise IMFError(f"No usable USD FX quote for {country}: {first_error}; {second_error}")


def retrieve(country: str, field: str) -> pd.Series:
    # Try known aggregate labels rather than silently mapping to an individual state.
    candidates = ("EA", "EA20", "EA19", "U2") if country == "EA" else (country,)
    errors = []
    for code in candidates:
        try:
            out = cpi(code) if field == "CPI" else usd_per_local(code)
            return out
        except IMFError as exc:
            errors.append(f"{code}: {exc}")
    raise IMFError(" | ".join(errors))


def build_table(c1: str, c2: str) -> pd.DataFrame:
    p1 = retrieve(c1, "CPI")
    p2 = retrieve(c2, "CPI")
    usd1 = retrieve(c1, "FX")
    usd2 = retrieve(c2, "FX")
    months = pd.period_range(FIRST_MONTH, CURRENT_MONTH, freq="M")
    table = pd.DataFrame(index=months)
    table.index.name = "Month"
    table["P1_CPI"] = p1.reindex(months)
    table["P2_CPI"] = p2.reindex(months)
    table["USD_per_Country1_currency"] = usd1.reindex(months)
    table["USD_per_Country2_currency"] = usd2.reindex(months)
    # (USD / currency1) / (USD / currency2) = currency2 / currency1.
    table["S_country2_per_country1"] = (
        table["USD_per_Country1_currency"] / table["USD_per_Country2_currency"]
    )
    table["RER"] = table["S_country2_per_country1"] * table["P1_CPI"] / table["P2_CPI"]
    for column in ["P1_CPI", "P2_CPI", "USD_per_Country1_currency", "USD_per_Country2_currency", "S_country2_per_country1", "RER"]:
        table.loc[~np.isfinite(table[column]) | (table[column] <= 0), column] = np.nan
    return table


st.title("🌍 IMF Real Exchange Rate Explorer")
st.caption("Monthly IMF CPI and period-average exchange rates · January 1976 onward, where available")
st.markdown(
    r"**Definitions:** $S_t =$ Country 2 currency units per 1 Country 1 currency unit; "
    r"$RER_t=S_t(P_{1,t}/P_{2,t})$; "
    r"$RER^*=\operatorname{mean}(RER_t)$ across valid selected months; "
    r"$Real\ Index_t=100(RER_t/RER^*)$."
)

options = country_options()
labels = list(options)
def label_for(code):
    return next((k for k, v in options.items() if v == code), labels[0])

with st.sidebar:
    st.header("Settings")
    label1 = st.selectbox("Country 1", labels, index=labels.index(label_for("USA")))
    label2 = st.selectbox("Country 2", labels, index=labels.index(label_for("JPN")))
    c1, c2 = options[label1], options[label2]
    start = st.selectbox("Start month", pd.period_range(FIRST_MONTH, CURRENT_MONTH, freq="M").astype(str).tolist(), index=0)
    end_options = pd.period_range(FIRST_MONTH, CURRENT_MONTH, freq="M").astype(str).tolist()
    end = st.selectbox("End month", end_options, index=len(end_options)-1)
    mode = st.radio("Chart", ["Both", "Real", "Nominal"], horizontal=True)
    st.caption("Data are downloaded directly from IMF SDMX 3.0. Data availability depends on the selected countries.")

if c1 == c2:
    st.warning("Select two different economies to compare exchange rates.")
    st.stop()
if pd.Period(start, freq="M") > pd.Period(end, freq="M"):
    st.error("Start month must be on or before end month.")
    st.stop()

try:
    with st.spinner("Downloading IMF CPI and exchange-rate series..."):
        full = build_table(c1, c2)
except IMFError as exc:
    st.error("The selected IMF series could not be retrieved. Try another country pair or retry later.")
    st.code(str(exc))
    st.info("Some IMF countries and euro-area aggregates do not have both required monthly datasets. No estimated or synthetic observations are used.")
    st.stop()

sample = full.loc[start:end].copy()
valid = sample["RER"].dropna()
if valid.empty:
    st.warning("No months in this interval contain all three required values: CPI 1, CPI 2, and nominal S.")
    st.dataframe(sample.reset_index().astype({"Month": str}), use_container_width=True)
    st.stop()

rer_star = valid.mean()  # Recomputed on every change to selected start/end dates.
sample["RER_star_sample_mean"] = rer_star
sample["Real_Index_100"] = 100 * sample["RER"] / rer_star
sample["Valid_for_RER"] = sample["RER"].notna()

m1, m2, m3 = st.columns(3)
m1.metric("RER* (sample mean)", f"{rer_star:,.5f}")
m2.metric("Valid RER observations", f"{len(valid):,} / {len(sample):,} months")
m3.metric("Latest nominal S in sample", f"{sample['S_country2_per_country1'].dropna().iloc[-1]:,.5f}" if sample["S_country2_per_country1"].notna().any() else "N/A")

x = sample.index.to_timestamp()
fig = make_subplots(specs=[[{"secondary_y": True}]])
if mode in ("Real", "Both"):
    fig.add_trace(go.Scatter(x=x, y=sample["Real_Index_100"], mode="lines", name="Real index (mean = 100)", connectgaps=False, line=dict(width=2.5)), secondary_y=False)
    fig.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.55)
if mode in ("Nominal", "Both"):
    fig.add_trace(go.Scatter(x=x, y=sample["S_country2_per_country1"], mode="lines", name="Nominal S (actual currency units)", connectgaps=False, line=dict(width=2)), secondary_y=(mode == "Both"))
fig.update_layout(title=f"{label1} → {label2} | {start} to {end}", hovermode="x unified", height=530, legend=dict(orientation="h", y=1.12), margin=dict(t=100))
fig.update_xaxes(title_text="Month")
fig.update_yaxes(title_text="Real exchange rate index (sample mean = 100)" if mode != "Nominal" else f"{c2} per 1 {c1} (actual units)", secondary_y=False)
if mode == "Both":
    fig.update_yaxes(title_text=f"Nominal: {c2} per 1 {c1}", secondary_y=True, showgrid=False)
st.plotly_chart(fig, use_container_width=True)

st.info("A real-index reading above 100 means Country 1's currency has a higher RER than its selected-period average under the specified S × P1/P2 convention. CPI base years may differ across countries; the index uses the sample mean to normalize the combined series.")

export = sample.reset_index()
export["Month"] = export["Month"].astype(str)
export.insert(1, "Country_1", c1)
export.insert(2, "Country_2", c2)
st.download_button("⬇️ Download selected monthly data (CSV)", data=export.to_csv(index=False).encode("utf-8-sig"), file_name=f"imf_rer_{c1}_{c2}_{start}_{end}.csv", mime="text/csv")
with st.expander("View monthly observations and calculations"):
    st.dataframe(export, use_container_width=True, hide_index=True)
with st.expander("Methodology and IMF sources"):
    st.markdown("""
- **CPI:** IMF CPI dataset, monthly headline all-items index: `COUNTRY.CPI._T.IX.M`.
- **FX:** IMF ER dataset, **monthly period average**: `COUNTRY.USD_XDC.PA_RT.M` (USD per local currency). If missing, the app attempts the reciprocal of `COUNTRY.XDC_USD.PA_RT.M`.
- **Bilateral nominal FX:** `S = (USD per Country 1 currency) / (USD per Country 2 currency)`, with **Country 2 currency per Country 1 currency** as units. This is never rebased to 100.
- **Real exchange rate:** `RER = S × CPI1 / CPI2`.
- **Sample normalization:** arithmetic mean of valid RER months **within the selected date interval**, recalculated whenever dates change. Invalid/missing observations are not interpolated.
- **Euro area:** tries several possible IMF area identifiers, but availability is not guaranteed. The app reports missing series instead of filling gaps.
- Sources: [IMF CPI](https://data.imf.org/) and [IMF Exchange Rates](https://data.imf.org/en/datasets/IMF.STA%3AER).
""")
