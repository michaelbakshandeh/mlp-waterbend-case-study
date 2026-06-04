# -*- coding: utf-8 -*-
"""MLP Waterbend Case Study

    https://colab.research.google.com/drive/17RqtzGC4mY0yZ_I2OtOLwoSOTL5Sp4qh
"""

import time
import itertools
import requests
import pandas as pd

BLS_API_KEY = "ea4b4d774dab4c1e9dca55a153ad4b7f"
BLS_V2 = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
QCEW_SLICE = "https://data.bls.gov/cew/data/api/{year}/{qtr}/{kind}/{code}.csv"
HEADERS = {"User-Agent": "research-pipeline (mnbakshandeh@gmail.com)"}


def sae_series(state_fips, area="00000", datatype="01", sa="U",
               supersector_industry="70722000"):
    """Build a State & Area Employment series ID. area='00000' = statewide."""
    return f"SM{sa}{state_fips}{area}{supersector_industry}{datatype}"

def _chunks(seq, n):
    it = iter(seq)
    while batch := list(itertools.islice(it, n)):
        yield batch

def _to_float(v):
    """BLS uses '-' (and occasionally '') for unavailable/suppressed values."""
    try:
        return float(v.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def _period_to_month(period):
    """Map a BLS period code to a representative month (1-12), or None to drop."""
    if period.startswith("M"):
        m = int(period[1:])
        return m if 1 <= m <= 12 else None          # M13 = annual avg -> drop
    if period.startswith("Q"):
        q = int(period[1:])
        return {1: 3, 2: 6, 3: 9, 4: 12}.get(q)      # Q05 annual avg -> drop
    if period.startswith("S"):
        return {1: 6, 2: 12}.get(int(period[1:]))    # S03 annual -> drop
    if period == "A01":
        return 12
    return None

def rebase_index_series_to_100(df, index_series_ids, value_col="value"):
    """Rebase index-style series to 100 at each series' first available date.

    Non-index series are left unchanged.
    Adds value_raw, is_index_series, base_value, and base_date.
    """
    out = df.copy()

    # Clean up in case this function is rerun on an already-modified df
    cols_to_drop = [
        "value_raw",
        "is_index_series",
        "base_value",
        "base_date",
        "base_value_idxbase",
        "base_date_idxbase",
    ]
    out = out.drop(columns=[c for c in cols_to_drop if c in out.columns])

    out["value"] = pd.to_numeric(out[value_col], errors="coerce")
    out["value_raw"] = out["value"]
    out["is_index_series"] = out["series_id"].isin(index_series_ids)

    idx_mask = out["is_index_series"]

    if idx_mask.any():
        base = (
            out.loc[idx_mask]
            .sort_values(["series_id", "date"])
            .groupby("series_id", as_index=False)
            .first()[["series_id", "date", "value"]]
            .rename(columns={
                "date": "base_date",
                "value": "base_value",
            })
        )

        out = out.merge(base, on="series_id", how="left")

        idx_mask = out["is_index_series"]

        out.loc[idx_mask, "value"] = (
            out.loc[idx_mask, "value_raw"]
            / out.loc[idx_mask, "base_value"]
            * 100
        )

    else:
        out["base_date"] = pd.NaT
        out["base_value"] = pd.NA

    return out

def fetch_bls(
    catalog,
    start_year,
    end_year,
    key=BLS_API_KEY,
    rebase_indexes=True,
    index_series_ids=None,
):
    """Pull many series from the v2 API; returns a tidy DataFrame.

    Handles the 50-series and 20-year per-call caps by chunking both,
    converts period codes to a date column, includes catalog values as
    series names, and optionally rebases CPI/PPI index series to 100 at
    the first available observation in the pulled window.
    """
    series_ids = list(catalog.keys())
    series_name_map = dict(catalog)

    rows = []

    year_windows = [
        (y, min(y + 19, end_year))
        for y in range(start_year, end_year + 1, 20)
    ]

    for batch in _chunks(series_ids, 50):
        for w_start, w_end in year_windows:
            payload = {
                "seriesid": batch,
                "startyear": str(w_start),
                "endyear": str(w_end),
                "registrationkey": key,
            }

            r = requests.post(BLS_V2, json=payload, headers=HEADERS, timeout=30)
            r.raise_for_status()

            body = r.json()

            if body.get("status") != "REQUEST_SUCCEEDED":
                print("WARN:", body.get("message"))

            for s in body.get("Results", {}).get("series", []):
                sid = s["seriesID"]
                series_name = series_name_map.get(sid)

                for d in s["data"]:
                    month = _period_to_month(d["period"])
                    val = _to_float(d["value"])

                    if month is None or val is None:
                        continue

                    rows.append({
                        "series_id": sid,
                        "series_name": series_name,
                        "date": pd.Timestamp(int(d["year"]), month, 1),
                        "value": val,
                    })

            time.sleep(0.5)

    df = (
        pd.DataFrame(rows)
        .drop_duplicates(["series_id", "date"])
        .sort_values(["series_id", "date"])
        .reset_index(drop=True)
    )

    if rebase_indexes:
        if index_series_ids is None:
            index_series_ids = INDEX_SERIES_IDS

        df = rebase_index_series_to_100(
            df,
            index_series_ids=index_series_ids,
            value_col="value",
        )

    return df

# ----------------------------------------------------------------------
# QCEW CSV-slice client (county / MSA / state employment + wages)
# ----------------------------------------------------------------------
def fetch_qcew(year, qtr, code, kind="industry", private_only=True):
    """Fetch one QCEW slice.

    kind='industry' -> one NAICS across ALL areas (county/MSA/state/national).
    kind='area'     -> one area across ALL industries.
    Returns establishments, monthly employment, total wages, avg weekly wage,
    and location quotients per (area, industry, ownership) record.
    """
    safe_code = str(code).replace("-", "_")          # e.g. 31-33 -> 31_33
    url = QCEW_SLICE.format(year=year, qtr=qtr, kind=kind, code=safe_code)
    df = pd.read_csv(url, dtype={"area_fips": str, "industry_code": str,
                                 "own_code": str}, storage_options=HEADERS)
    if private_only:
        df = df[df["own_code"] == "5"]               # private sector
    # Classify geography for convenience.
    def geo(fips):
        if fips == "US000":
            return "national"
        if fips.startswith("C"):
            return "msa"
        if fips.endswith("000"):
            return "state"
        return "county"
    df["geo_level"] = df["area_fips"].map(geo)
    return df

import pandas as pd
import numpy as np

LS_CODE = "722513"        # limited-service restaurants: matches the SM72251XUSN revenue universe
START_YEAR = 2015

def available_quarters(start_year=START_YEAR, lag_quarters=2):
    """QCEW posts ~2 quarters late; stop before the unpublished edge."""
    last = (pd.Timestamp.today().to_period("Q") - lag_quarters)
    rng = pd.period_range(f"{start_year}Q1", last, freq="Q")
    return [(p.year, p.quarter) for p in rng]

def ls_wage_index(out, base_to=100.0):
    """Monthly labor-price index (NSA), rebased to base_to at the first month."""
    s = out.set_index("date")["avg_wkly_wage_qtr"].sort_index()
    return (s / s.iloc[0] * base_to).values

def qcew_ls_monthly(area_code="US000", code=LS_CODE):
    rows = []
    for year, qtr in available_quarters():
        try:
            df = fetch_qcew(year, qtr, code, kind="industry", private_only=True)
        except Exception as e:
            print(f"skip {year}Q{qtr}: {e}")
            continue
        nat = df[(df["area_fips"] == "US000") & (df["industry_code"] == str(code))]
        if nat.empty:
            continue
        r = nat.iloc[0]
        emp = np.array([float(r["month1_emplvl"]),
                        float(r["month2_emplvl"]),
                        float(r["month3_emplvl"])])
        wages_q, awe_q = float(r["total_qtrly_wages"]), float(r["avg_wkly_wage"])
        tot_e = emp.sum()
        w = emp / tot_e if tot_e > 0 else np.array([1/3, 1/3, 1/3])   # equal-split fallback
        base_m = (qtr - 1) * 3 + 1
        for k in range(3):
            rows.append({
                "date": pd.Timestamp(year=year, month=base_m + k, day=1),
                "employees": emp[k],
                "wage_bill_month": wages_q * w[k],
                "avg_wkly_wage_qtr": awe_q,
            })
    return (pd.DataFrame(rows)
              .drop_duplicates("date", keep="last")
              .sort_values("date").reset_index(drop=True))

# Lower-48 state FIPS (QCEW area_fips = SS + "000"); excludes AK/HI and DC.
STATE_FIPS = {
    "01000": "AL", "04000": "AZ", "05000": "AR", "06000": "CA", "08000": "CO",
    "09000": "CT", "10000": "DE", "12000": "FL", "13000": "GA", "16000": "ID",
    "17000": "IL", "18000": "IN", "19000": "IA", "20000": "KS", "21000": "KY",
    "22000": "LA", "23000": "ME", "24000": "MD", "25000": "MA", "26000": "MI",
    "27000": "MN", "28000": "MS", "29000": "MO", "30000": "MT", "31000": "NE",
    "32000": "NV", "33000": "NH", "34000": "NJ", "35000": "NM", "36000": "NY",
    "37000": "NC", "38000": "ND", "39000": "OH", "40000": "OK", "41000": "OR",
    "42000": "PA", "44000": "RI", "45000": "SC", "46000": "SD", "47000": "TN",
    "48000": "TX", "49000": "UT", "50000": "VT", "51000": "VA", "53000": "WA",
    "54000": "WV", "55000": "WI", "56000": "WY",
}

def _expand_area_quarter(r, year, qtr):
    """One QCEW industry row (a single area-quarter) -> 3 monthly dicts."""
    emp = np.array([float(r["month1_emplvl"]),
                    float(r["month2_emplvl"]),
                    float(r["month3_emplvl"])])
    wages_q, awe_q = float(r["total_qtrly_wages"]), float(r["avg_wkly_wage"])
    tot_e = emp.sum()
    w = emp / tot_e if tot_e > 0 else np.array([1/3, 1/3, 1/3])   # equal-split fallback
    base_m = (qtr - 1) * 3 + 1
    return [{
        "date": pd.Timestamp(year=year, month=base_m + k, day=1),
        "employees": emp[k],
        "wage_bill_month": wages_q * w[k],
        "avg_wkly_wage_qtr": awe_q,
    } for k in range(3)]

def qcew_ls_state_monthly(code=LS_CODE, states=STATE_FIPS):
    """Lower-48 monthly LS-restaurant panel, long (row per state-month).
    Reuses the single industry slice per quarter (it already holds every area),
    so this costs the same number of fetches as the national-only version.
    """
    rows = []
    for year, qtr in available_quarters():
        try:
            df = fetch_qcew(year, qtr, code, kind="industry", private_only=True)
        except Exception as e:
            print(f"skip {year}Q{qtr}: {e}")
            continue
        sub = df[(df["area_fips"].isin(states)) & (df["industry_code"] == str(code))]
        for _, r in sub.iterrows():
            fips = r["area_fips"]
            for rec in _expand_area_quarter(r, year, qtr):
                rec["state_fips"], rec["state"] = fips, states[fips]
                rows.append(rec)
    out = (pd.DataFrame(rows)
             .drop_duplicates(["state_fips", "date"], keep="last")
             .sort_values(["state", "date"]).reset_index(drop=True))
    return out[["state", "state_fips", "date", "employees",
                "wage_bill_month", "avg_wkly_wage_qtr"]]

monthly = qcew_ls_monthly()

# 1) National + regional menu prices and wages via the time-series API.
CES_NATIONAL = {
    "CES7072200001": "Food svcs & drinking places: all employees (000s)",
    "CES7072200011": "Food svcs & drinking places: avg weekly earnings, all emp"
}

# --- CPI (monthly; NATIONAL + 4 CENSUS REGIONS). CU=CPI-U, CW=CPI-W.
# Item codes: SEFV food away from home, SEFV01 full-service, SEFV02 limited-service,
# SEFJ dairy, SAF113 fruits & veg, SAF112 meats/poultry/fish/eggs, SEFP nonalc bev.
CPI_NATIONAL = {
    "CUUR0000SEFV":   "Food away from home, US city avg (NSA)",
    "CUSR0000SEFV":   "Food away from home, US city avg (SA)",
    "CUUR0000SEFV02": "Limited service meals & snacks, US city avg (NSA)",
    "CUUR0000SEFV01": "Full service meals & snacks, US city avg (NSA)",
    "CWUR0000SEFV":   "Food away from home, CPI-W (NSA)",
}
# Regional food-away-from-home (NSA). Area: 0100 NE, 0200 MW, 0300 S, 0400 W.
# Pull down for the SSS estimate
CPI_REGION_NAMES = {
    "1": "Northeast",
    "2": "Midwest",
    "3": "South",
    "4": "West",
}

CPI_REGIONS = {
    f"CUUR0{r}00SEFV": f"{region}"
    for r, region in CPI_REGION_NAMES.items()
}

INDEX_SERIES_IDS = set(list(CPI_NATIONAL.keys())
                       + list(CPI_REGIONS.keys()))

wages = fetch_bls(CES_NATIONAL, start_year=2015, end_year=2026)
cpi = fetch_bls(CPI_NATIONAL, start_year=2015, end_year=2026)
regional_price = fetch_bls(CPI_REGIONS, start_year=2015, end_year=2026)

INDEXES = {
    # "PCOFFOTMUSDM": ("coffee_arabica", "cents_lb"),
    # "PCOFFROBUSDM": ("coffee_robusta", "cents_lb"),
    # "PBEEFUSDM": ("beef",           "cents_lb"),
    # "PPOULTUSDM": ("poultry",        "cents_lb"),
    # "PWHEAMTUSDM": ("wheat",          "usd_tonne"),
    # "PFOODINDEXM": ("food_index",     "index"),
    "SM72251XUSN": ("retail sales: limited service eating places", "thousands_dollars"),
    "PCU311311": ("PPI by Industry: Food Manufacturing", 'index'),
    "WPU026301": ("PPI: Coffee, whole bean, ground, and instant", 'index'), # use for BROS
    "WPU022101": ("PPI: Beef and Veal Products, Fresh or Frozen", 'index'), # use for SHAK
    "WPU022": ("PPI: Processed Foods and Feeds: Meats, Poultry, and Fish", 'index'), # use for CAVA
    "WPU011302": ("PPI: Fresh Vegetables ex Potatoes", "index") # use for CAVA
}

FRED_KEY = "bf3ab50f282445393be43b89cb254b47"
FRED = "https://api.stlouisfed.org/fred/series/observations"

import time

def fetch_fred(series_id, start="2015-01-01", units="lin", max_retries=5):
    # units: lin level | pch m/m % | pc1 y/y % | chg change | ch1 y/y change
    p = {"series_id": series_id, "api_key": FRED_KEY, "file_type": "json",
         "observation_start": start, "units": units}
    for attempt in range(max_retries):
        r = requests.get(FRED, params=p, timeout=30)
        if r.status_code == 429:
            time.sleep(2 ** attempt)   # exponential backoff: 1,2,4,8,16s
            continue
        r.raise_for_status()
        data = r.json()
        if "observations" not in data:
            raise RuntimeError(f"FRED returned no observations for {series_id}: {data}")
        df = pd.DataFrame(data["observations"])[["date", "value"]]
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")  # '.' -> NaN
        return df.assign(series_id=series_id).dropna(subset=["value"])
    raise RuntimeError(f"FRED rate-limited after {max_retries} retries: {series_id}")

def normalize_series(df, unit):
    """Per-series: rebase index types to 100 at the first observation, leave levels alone."""
    df = df.sort_values("date").copy()
    df["value_raw"] = df["value"]                       # keep the original
    if unit == "index":
        base = df["value"].iloc[0]                      # first (start-period) observation
        df["value"] = df["value"] / base * 100.0
        df["base_value"] = base
        df["base_date"]  = df["date"].iloc[0]
    return df

costs = pd.concat(
    normalize_series(fetch_fred(sid, units="lin"), unit)
        .assign(commodity=name, raw_units=unit)
    for sid, (name, unit) in INDEXES.items()
)

import pandas as pd
import numpy as np

# -----------------------------
# 1. CONFIG   (labor series + BASE_LABOR_COST_PCT removed; labor is now observed)
# -----------------------------
MENU_CPI_SERIES = "CUUR0000SEFV02"   # Limited-service meals & snacks, NSA

SALES_IN_MILLIONS = True
FOOD_PRICE_ALREADY_INDEXED = False

BASE_FOOD_COST_PCT = 0.30
BASE_OTHER_OPEX_PCT = 0.20
LABOR_LOAD_FACTOR = 1.0

sales = costs[costs["series_id"].eq("SM72251XUSN")].copy()
food_price = costs[costs["series_id"].eq("PCU311311")].copy()

food_price["value_raw"] = pd.to_numeric(food_price["value"], errors="coerce")
food_price = food_price.sort_values("date")
food_price["base_date"] = food_price["date"].iloc[0]
food_price["base_value"] = food_price["value_raw"].iloc[0]
food_price["value"] = food_price["value_raw"] / food_price["base_value"] * 100
food_price["is_index_series"] = True

# -----------------------------
# 2. CLEAN INPUTS   (wages cleaning dropped; no CES labor series anymore)
# -----------------------------
cpi = cpi.copy()
sales = sales.copy()
food_price = food_price.copy()

cpi["date"] = pd.to_datetime(cpi["date"])
sales["date"] = pd.to_datetime(sales["date"])
food_price["date"] = pd.to_datetime(food_price["date"])

cpi["value"] = pd.to_numeric(cpi["value"], errors="coerce")
if "value_raw" in sales.columns:
    sales["value_raw"] = pd.to_numeric(sales["value_raw"], errors="coerce")
if "value" in sales.columns:
    sales["value"] = pd.to_numeric(sales["value"], errors="coerce")
if "value" in food_price.columns:
    food_price["value"] = pd.to_numeric(food_price["value"], errors="coerce")
if "value_raw" in food_price.columns:
    food_price["value_raw"] = pd.to_numeric(food_price["value_raw"], errors="coerce")

# -----------------------------
# 3. REVENUE DOLLARS   (unchanged)
# -----------------------------
sales_value_col = "value_raw" if "value_raw" in sales.columns else "value"
sales_m = (sales[["date", sales_value_col]]
           .rename(columns={sales_value_col: "limited_service_sales"})
           .sort_values("date").drop_duplicates("date", keep="last"))
sales_m["revenue"] = (sales_m["limited_service_sales"] * 1_000_000
                      if SALES_IN_MILLIONS else sales_m["limited_service_sales"])
sales_m["revenue_index"] = sales_m["revenue"] / sales_m["revenue"].dropna().iloc[0] * 100

# -----------------------------
# 4. MENU PRICE INDEX   (unchanged)
# -----------------------------
menu_price = (cpi.loc[cpi["series_id"].eq(MENU_CPI_SERIES), ["date", "value"]]
              .rename(columns={"value": "menu_price_index"})
              .sort_values("date").drop_duplicates("date", keep="last"))

# -----------------------------
# 5. FOOD INPUT INDEX   (unchanged)
# -----------------------------
food_value_col = "value" if "value" in food_price.columns else "value_raw"
food_input = (food_price[["date", food_value_col]]
              .rename(columns={food_value_col: "food_input_price"})
              .sort_values("date").drop_duplicates("date", keep="last"))
if FOOD_PRICE_ALREADY_INDEXED:
    food_input["food_input_index"] = food_input["food_input_price"]
else:
    food_input["food_input_index"] = (food_input["food_input_price"]
                                      / food_input["food_input_price"].dropna().iloc[0] * 100)
food_input = food_input[["date", "food_input_index"]]

# -----------------------------
# 6. LABOR COST DOLLARS  (OBSERVED -- replaces the entire index/proxy block)
#    `monthly` is the output of qcew_ls_monthly(): national 722513 payroll,
#    split to months by QCEW headcount. wage_bill_month is actual dollars.
# -----------------------------
labor = (monthly[["date", "wage_bill_month"]]
         .rename(columns={"wage_bill_month": "labor_cost_observed"})
         .sort_values("date").drop_duplicates("date", keep="last"))
labor["date"] = pd.to_datetime(labor["date"])
labor["labor_cost_observed"] *= LABOR_LOAD_FACTOR

# -----------------------------
# 7. MERGE   (labor joined as dollars, inner -> series ends where QCEW ends)
# -----------------------------
profit = (sales_m[["date", "revenue", "revenue_index"]]
          .merge(menu_price, on="date", how="left")
          .merge(food_input, on="date", how="left")
          .merge(labor, on="date", how="left")
          .sort_values("date"))

base_row = profit.dropna(subset=["revenue_index", "menu_price_index", "food_input_index"]).iloc[0]
base_date = base_row["date"]
for col in ["revenue_index", "menu_price_index", "food_input_index"]:
    profit[col] = profit[col] / base_row[col] * 100
profit["base_date"] = base_date

# -----------------------------
# 8. MACRO PRESSURE VARIABLES   (labor_pressure removed; food unchanged)
# -----------------------------
profit["implied_volume_index"] = profit["revenue_index"] / profit["menu_price_index"] * 100
profit["food_pressure_vs_revenue"] = (profit["implied_volume_index"] / 100
                                      * profit["food_input_index"] / 100
                                      / (profit["revenue_index"] / 100))

# -----------------------------
# 9. COST % ASSUMPTIONS   (labor now observed; food/occupancy/other unchanged)
# -----------------------------
profit["food_cost_pct_revenue"] = BASE_FOOD_COST_PCT * profit["food_pressure_vs_revenue"]
profit["labor_cost_pct_revenue"] = profit["labor_cost_observed"] / profit["revenue"]
profit["other_opex_pct_revenue"] = BASE_OTHER_OPEX_PCT

# -----------------------------
# 10. COST DOLLARS   (labor = actual dollars, not revenue * pct)
# -----------------------------
profit["food_cost"] = profit["revenue"] * profit["food_cost_pct_revenue"]
profit["labor_cost"] = profit["labor_cost_observed"]
profit["other_opex"] = profit["revenue"] * profit["other_opex_pct_revenue"]

# -----------------------------
# 11. PROFITABILITY DOLLARS   (unchanged)
# -----------------------------
profit["restaurant_profit"] = (profit["revenue"] - profit["food_cost"] - profit["labor_cost"]
                               - profit["other_opex"])
profit["restaurant_profit_index"] = (profit["restaurant_profit"]
                                     / profit["restaurant_profit"].dropna().iloc[0] * 100)
profit["profit_margin"] = profit["restaurant_profit"] / profit["revenue"]

# -----------------------------
# 12. FINAL OUTPUT
# -----------------------------
profitability = profit[[
    "date", "base_date", "revenue", "revenue_index", "menu_price_index",
    "implied_volume_index", "food_input_index",
    "food_pressure_vs_revenue",
    "food_cost_pct_revenue", "labor_cost_pct_revenue",
    "other_opex_pct_revenue",
    "food_cost", "labor_cost", "other_opex",
    "restaurant_profit", "restaurant_profit_index", "profit_margin",
]].copy()

def add_growth(df, cols, date_col='date', ppy=12):
    """y/y growth and annualized 2y CAGR (both as rates) for each column in `cols`.
    Monthly data -> ppy=12. Assumes a single series sorted by date.
    """
    df = df.sort_values(date_col).copy()
    for c in cols:
        df[f'{c}_yoy']     = df[c].pct_change(ppy)
        df[f'{c}_2y_cagr'] = (1 + df[c].pct_change(2 * ppy)) ** (1 / 2) - 1
    return df

growth_cols = [
    "revenue", "revenue_index", "menu_price_index", "implied_volume_index",
    "food_input_index", "food_cost", "labor_cost", "other_opex",
    "restaurant_profit", "profit_margin"
]
profitability = add_growth(profitability, growth_cols)

"""## Load Store Counts Data From Filings"""

# utilize AI to fetch the filings from image on 10K
import pandas as pd

counts = {
    "WA": 67, "OR": 155, "ID": 37, "CA": 154, "NV": 30, "UT": 25,
    "CO": 45, "AZ": 80, "NM": 11, "TX": 166, "OK": 22, "KS": 11,
    "MO": 7, "KY": 2, "TN": 18, "AL": 1,
}

census = {
 "WA":("West","Pacific"),"OR":("West","Pacific"),"CA":("West","Pacific"),
 "AK":("West","Pacific"),"HI":("West","Pacific"),
 "ID":("West","Mountain"),"NV":("West","Mountain"),"UT":("West","Mountain"),
 "CO":("West","Mountain"),"AZ":("West","Mountain"),"NM":("West","Mountain"),
 "MT":("West","Mountain"),"WY":("West","Mountain"),
 "KS":("Midwest","West North Central"),"MO":("Midwest","West North Central"),
 "IA":("Midwest","West North Central"),"MN":("Midwest","West North Central"),
 "NE":("Midwest","West North Central"),"ND":("Midwest","West North Central"),
 "SD":("Midwest","West North Central"),
 "IL":("Midwest","East North Central"),"IN":("Midwest","East North Central"),
 "MI":("Midwest","East North Central"),"OH":("Midwest","East North Central"),
 "WI":("Midwest","East North Central"),
 "TX":("South","West South Central"),"OK":("South","West South Central"),
 "AR":("South","West South Central"),"LA":("South","West South Central"),
 "AL":("South","East South Central"),"KY":("South","East South Central"),
 "MS":("South","East South Central"),"TN":("South","East South Central"),
 "DE":("South","South Atlantic"),"FL":("South","South Atlantic"),
 "GA":("South","South Atlantic"),"MD":("South","South Atlantic"),
 "NC":("South","South Atlantic"),"SC":("South","South Atlantic"),
 "VA":("South","South Atlantic"),"WV":("South","South Atlantic"),"DC":("South","South Atlantic"),
 "CT":("Northeast","New England"),"ME":("Northeast","New England"),
 "MA":("Northeast","New England"),"NH":("Northeast","New England"),
 "RI":("Northeast","New England"),"VT":("Northeast","New England"),
 "NJ":("Northeast","Middle Atlantic"),"NY":("Northeast","Middle Atlantic"),
 "PA":("Northeast","Middle Atlantic"),
}

by_state = pd.DataFrame({"shops": counts}).rename_axis("state").reset_index()
by_state["region"]   = by_state["state"].map(lambda s: census[s][0])
by_state["division"] = by_state["state"].map(lambda s: census[s][1])
by_state = by_state.sort_values("shops", ascending=False).reset_index(drop=True)

bros_locations = (by_state.groupby("region")["shops"].sum()
             .reindex(["Northeast","Midwest","South","West"]).fillna(0).astype(int))

import pandas as pd

# total store count by state from filing image
shak_counts = {
    "AL": 1,  "AZ": 9,  "CA": 50, "CO": 10, "CT": 9,  "DE": 1,
    "DC": 7,  "FL": 27, "GA": 9,  "IL": 14, "IN": 5,  "KS": 2,
    "KY": 1,  "LA": 4,  "MD": 11, "MA": 17, "MI": 8,  "MN": 5,
    "MO": 10, "NV": 8,  "NH": 1,  "NJ": 28, "NY": 67, "NC": 10,
    "OH": 11, "OK": 1,  "OR": 4,  "PA": 20, "RI": 3,  "TN": 6,
    "TX": 29, "UT": 7,  "VA": 14, "WA": 8,  "WI": 3,
}

shak_by_state = (
    pd.DataFrame({"shops": shak_counts})
      .rename_axis("state")
      .reset_index()
)

shak_by_state["region"] = shak_by_state["state"].map(lambda s: census[s][0])
shak_by_state["division"] = shak_by_state["state"].map(lambda s: census[s][1])

shak_by_state = (
    shak_by_state
    .sort_values("shops", ascending=False)
    .reset_index(drop=True)
)

shak_locations = (
    shak_by_state
    .groupby("region")["shops"]
    .sum()
    .reindex(["Northeast", "Midwest", "South", "West"])
    .fillna(0)
    .astype(int)
)

shak_locations_pct = shak_locations / shak_locations.sum()

import pandas as pd

# Top 10 from ScrapeHero table; all other states estimated from map dots
cava_counts = {
    "TX": 72, "CA": 45, "FL": 43, "VA": 36, "NC": 30, "GA": 27,
    "MD": 26, "NY": 20, "MA": 20, "PA": 15,

    # map-implied estimates for non-top-10 states
    "TN": 14, "NJ": 14, "SC": 13, "OK": 12, "AL": 9,  "LA": 9,
    "AZ": 7,  "CO": 7,  "DC": 7,  "IL": 7,  "CT": 5,  "IN": 5,
    "KS": 4,  "OH": 4,  "MI": 3,  "RI": 3,  "AR": 2,  "MO": 2,
    "NH": 2,  "DE": 1,
}

top_10_states = {"TX", "CA", "FL", "VA", "NC", "GA", "MD", "NY", "MA", "PA"}

cava_by_state = (
    pd.DataFrame({"shops": cava_counts})
      .rename_axis("state")
      .reset_index()
)

cava_by_state["region"] = cava_by_state["state"].map(lambda s: census[s][0])
cava_by_state["division"] = cava_by_state["state"].map(lambda s: census[s][1])

cava_by_state["source"] = cava_by_state["state"].map(
    lambda s: "ScrapeHero top 10 table" if s in top_10_states
    else "map proxy"
)

cava_by_state = (
    cava_by_state
    .sort_values("shops", ascending=False)
    .reset_index(drop=True)
)

cava_locations = (
    cava_by_state
    .groupby("region")["shops"]
    .sum()
    .reindex(["Northeast", "Midwest", "South", "West"])
    .fillna(0)
    .astype(int)
)

"""## Load Actuals"""

df = pd.read_excel('actuals.xlsx', parse_dates=['Period Start','Period End']).sort_values(['Ticker', 'Period End'])
df = df[df.Ticker != 'WING']
calendar = df[['Ticker', 'Period', 'Period Start', 'Period End']]
df = df.dropna()
# df

import pandas as pd
import numpy as np

def aggregate_to_quarters(panel, calendar, ticker_to_col=None, agg='mean',
                          value_name='search_interest',
                          qtd_projection=False, extrap_lag_years=1):
    """Roll a weekly/monthly panel to a per-ticker fiscal-quarter calendar.

    If qtd_projection=True, the partially observed ending quarter is completed by
    extrapolation: take the elapsed N weeks, compare to the first N weeks of the
    comparable period extrap_lag_years back to get a QTD growth rate, then fill the
    remaining weeks as comparable_week * (1 + growth) before aggregating.
    """
    cal = calendar.copy()
    cal['Period Start'] = pd.to_datetime(cal['Period Start'])
    cal['Period End']   = pd.to_datetime(cal['Period End'])
    ticker_to_col = ticker_to_col or {}

    idx       = pd.DatetimeIndex(panel.index)
    span      = int(pd.Series(idx).diff().dt.days.dropna().mode().iloc[0])
    mid       = idx + pd.to_timedelta((span - 1) / 2, unit='D')
    last_date = idx.max()

    cal[value_name] = np.nan
    if qtd_projection:
        cal['is_projected'] = False
        cal['qtd_growth']   = np.nan

    agg_of = lambda a: getattr(pd.Series(np.asarray(a, float)), agg)() if len(a) else np.nan

    for ticker, qrows in cal.groupby('Ticker'):
        col = ticker_to_col.get(ticker, ticker)
        if col not in panel.columns:
            continue
        qrows = qrows.sort_values('Period Start')
        vals  = panel[col].to_numpy()
        which = pd.IntervalIndex.from_arrays(qrows['Period Start'], qrows['Period End'],
                                             closed='both').get_indexer(mid)   # -1 if none

        by_q = (pd.DataFrame({'q': which, 'v': vals})
                .query('q >= 0').dropna(subset=['v']).groupby('q')['v'].agg(agg))
        cal.loc[qrows.index[by_q.index], value_name] = by_q.to_numpy()

        if not qtd_projection:
            continue

        starts  = qrows['Period Start'].to_numpy()
        ends    = qrows['Period End'].to_numpy()
        periods = qrows['Period'].to_numpy()

        begun = np.flatnonzero(starts <= last_date)
        if len(begun) == 0:
            continue
        cur = begun[-1]                                        # latest quarter that has begun
        if ends[cur] <= last_date:
            continue                                           # already complete

        qtr, yr  = periods[cur].split()                        # "Q1 2026" -> comparable "Q1 2025"
        comp_hit = np.flatnonzero(periods == f"{qtr} {int(yr) - extrap_lag_years}")
        if len(comp_hit) == 0:
            continue

        actual = vals[which == cur]                            # date-ordered (panel sorted asc)
        comp   = vals[which == comp_hit[-1]]
        n = len(actual)
        if n == 0 or len(comp) == 0:
            continue
        base = agg_of(comp[:n])
        if not base:                                           # 0 -> no rate
            continue
        growth = agg_of(actual) / base - 1.0
        cal.loc[qrows.index[cur], value_name]     = agg_of(np.r_[actual, comp[n:] * (1.0 + growth)])
        cal.loc[qrows.index[cur], 'is_projected'] = True
        cal.loc[qrows.index[cur], 'qtd_growth']   = growth

    return cal

def build_estimates(quarterly, last_date, value_name='search_interest',
                    agg='mean', lag_years=1):
    """Walk-forward estimate at every quarter: comparable(lag yrs back) x (1 + growth),
    growth = trailing growth knowable at Q-1 (PIT, carried forward). The growth column
    is the cumulative growth over lag_years applied to the comparable (YoY when lag=1).
    """
    q = quarterly.copy()
    q['Period Start'] = pd.to_datetime(q['Period Start'])
    q['Period End']   = pd.to_datetime(q['Period End'])
    last_date = pd.Timestamp(last_date)
    w = lag_years * 4

    q['period_type'] = np.where(q['Period End'] <= last_date, 'realized',
                        np.where(q['Period Start'] <= last_date, 'partial', 'forecast'))
    q = q.rename(columns={value_name: 'actual'})

    parts = []
    for _, g in q.groupby('Ticker', sort=False):
        g = g.sort_values('Period Start').copy()
        actual = g['actual'].to_numpy(float)
        pt     = g['period_type'].to_numpy()
        complete = np.where(pt == 'realized', actual, np.nan)
        roll  = pd.Series(complete).rolling(w, min_periods=w).agg(agg)
        g_cum = (roll.shift(1) / roll.shift(1 + w) - 1).ffill().to_numpy()
        est    = np.full(len(g), np.nan)
        growth = np.full(len(g), np.nan)
        for p in range(len(g)):
            c = p - w
            if c < 0:
                continue
            comp = actual[c] if pt[c] == 'realized' else est[c]
            if np.isnan(comp) or np.isnan(g_cum[p]):
                continue
            est[p]    = comp * (1 + g_cum[p])
            growth[p] = g_cum[p]
        g['estimate'] = est
        g['growth']   = growth
        parts.append(g)

    out = pd.concat(parts)
    return out[['Ticker', 'Period', 'Period Start', 'Period End',
                'actual', 'estimate', 'growth', 'period_type']].reindex(quarterly.index)

def geo_normalize(values, geo_table, by='region', how='mean', weight_col='shops'):
    """Collapse the geographic columns of `values` into one shop-weighted series.

    values    : DataFrame whose COLUMNS are geographies at level `by`
                (region / division / state), rows are whatever you want collapsed
                (e.g. dates). A single cross-section may be passed as a Series
                indexed by geography.
    geo_table : table like the shops-by-state frame, with columns [by, weight_col].
    by        : geographic level to weight over; must match `values` column labels.
    how       : 'mean' -> sum(w*v)/sum(w)
                'sum'  -> sum(w*v)
    Returns a Series indexed like the rows of `values`.
    """
    if isinstance(values, pd.Series):
        values = values.to_frame().T
    v = values.copy()

    w = geo_table.groupby(by)[weight_col].sum()      # roll shop counts to the chosen level
    w = w.reindex(v.columns)                         # align weights to value columns

    missing = list(v.columns[w.isna()])
    if missing:
        print(f"  no shop weight for {missing} (excluded from the weighting)")

    wv  = v.mul(w, axis=1)                            # w_g * v_g
    num = wv.sum(axis=1, skipna=True)                 # over geographies present in the row
    if how == 'sum':
        return num
    den = v.notna().mul(w, axis=1).sum(axis=1)        # weights only where value present
    return num / den

import pandas as pd, numpy as np

def cagr(df, n, interval, value_col='search_interest',
         ticker_col='Ticker', order_col='Period Start', extrap_col=None):
    order = df.sort_values([ticker_col, order_col])
    if extrap_col is None:
        extrap_col = f'{value_col}_extrap'
    vals = order[value_col]
    if extrap_col in order:
        vals = vals.where(vals.notna(), order[extrap_col])   # actual, else projected
    prior = vals.groupby(order[ticker_col], sort=False).shift(n * interval)
    out   = (vals / prior) ** (1 / n) - 1
    return out.reindex(df.index)

# get a revenue growth figure on y/y and y/2y CAGR
limited_service_price = cpi[cpi.series_id == 'CUUR0000SEFV02'].set_index('date')

for t in ['SHAK','CAVA','BROS']:
  limited_service_price[t] = limited_service_price['value']

prices = limited_service_price[['SHAK','CAVA','BROS']]
prices.index = pd.to_datetime(prices.index)

def scale_by_capture(df, capture_col='capture', cols=('actual', 'estimate'),
                     lag_years=1, max_lookback_years=2):
    """Project capture from the year-ago q/q step, then back out a level as
    col / projected_capture. If the data is lagged far enough that the year-ago
    step anchors are themselves unreported, step the growth source back another
    year (up to max_lookback_years) so the projection still forms. Projected
    capture chains forward through unreported quarters via cap_est[t-1].
    """
    w   = lag_years * 4
    out = df.copy()
    out['Period Start'] = pd.to_datetime(out['Period Start'])
    out['capture_est'] = np.nan
    for c in cols:
        out[f'{c}_scaled'] = np.nan

    for _, g in out.groupby('Ticker', sort=False):
        g   = g.sort_values('Period Start')
        cap = g[capture_col].to_numpy(float)
        cap_est = np.full(len(g), np.nan)
        for p in range(w + 1, len(g)):
            prev = cap[p - 1] if not np.isnan(cap[p - 1]) else cap_est[p - 1]
            if np.isnan(prev) or prev == 0:
                continue
            growth = np.nan
            for k in range(1, max_lookback_years + 1):   # year-ago step, then a year further back
                b = p - k * w - 1
                if b < 0:
                    break
                base, comp = cap[b], cap[p - k * w]
                if base and not np.isnan(base) and not np.isnan(comp):
                    growth = comp / base - 1.0
                    break
            if np.isnan(growth):
                continue
            cap_est[p] = (1.0 + growth) * prev
        out.loc[g.index, 'capture_est'] = cap_est
        for c in cols:
            out.loc[g.index, f'{c}_scaled'] = g[c].to_numpy(float) / cap_est

    return out

# get a raw macro number
limited_service_sales = costs[costs.series_id == 'SM72251XUSN'].set_index('date')

for t in ['SHAK','CAVA','BROS']:
  limited_service_sales[t] = limited_service_sales['value']

revs = limited_service_sales[['SHAK','CAVA','BROS']]
revs.index = pd.to_datetime(revs.index)
rev_q = aggregate_to_quarters(
    revs, calendar,
    value_name='revenue', agg='sum', qtd_projection=True)

# panel_q = build_estimates(
#     panel_q, last_date=panel.index.max(),
#     value_name='price', agg='mean')

rev_q = build_estimates(
    rev_q, last_date=prices.index.max(),
    value_name='revenue', agg='sum')

rev_merged = rev_q.merge(df, on=['Ticker','Period','Period Start','Period End'], how='left')
rev_merged['capture'] = rev_merged['actual'] / rev_merged['Restaurant Sales ($K)']

rev_mdled = scale_by_capture(rev_merged)

"""## Cost Line Items"""

# get a raw macro number
ppi_costs = costs[~costs.series_id.isin(['SM72251XUSN','PCU311311'])].set_index('date')

# mapping
item_to_series = {
    'WPU026301': 'BROS',
    'WPU022101': 'SHAK',
    'WPU022': 'CAVA',
    # 'WPU011302': 'CAVA' -- worse estimate quality than just general Processed Foods
}

ppi_costs['Ticker'] = ppi_costs['series_id'].map(item_to_series)

# Group by date and Ticker, take the mean, then unstack
ppi_costs_wide = ppi_costs.groupby(['date', 'Ticker'])['value'].mean().unstack()

food_cost_q = aggregate_to_quarters(
    ppi_costs_wide, calendar,
    value_name='price_cost', agg='mean', qtd_projection=True)

# panel_q = build_estimates(
#     panel_q, last_date=panel.index.max(),
#     value_name='price', agg='mean')

food_cost_q = build_estimates(
    food_cost_q, last_date=ppi_costs_wide.index.max(),
    value_name='price_cost', agg='mean')

food_cost_merged = food_cost_q.merge(df, on=['Ticker','Period','Period Start','Period End'], how='left')
food_cost_merged['capture'] = food_cost_merged['actual'] / food_cost_merged['Food & Distribution ($K)']

food_cost_mdled = scale_by_capture(food_cost_merged)

"""## Labor Input Modeling"""

# what this data captures: cost pressure per employee
# what this data misses: increases/decreases in the labor force, large mix shifts will make the estimate unreliable
labor_monthly = qcew_ls_state_monthly()

# Pivot with date as index and state as columns
wage_monthly = labor_monthly.pivot(index='date', columns='state', values='avg_wkly_wage_qtr')
# wage_monthly = labor_monthly[['state','date','avg_wkly_wage_qtr']].set_index('date')
shak_weighted = geo_normalize(wage_monthly, shak_by_state, by='state').to_frame('SHAK')
cava_weighted = geo_normalize(wage_monthly, cava_by_state, by='state').to_frame('CAVA')
bros_weighted = geo_normalize(wage_monthly, by_state, by='state').to_frame('BROS')

wage_blended = shak_weighted.merge(cava_weighted, on='date', how='outer')
wage_blended = wage_blended.merge(bros_weighted, on='date', how='outer')

labor_q = aggregate_to_quarters(
    wage_blended, calendar,
    value_name='wage', agg='mean', qtd_projection=True)

# panel_q = build_estimates(
#     panel_q, last_date=panel.index.max(),
#     value_name='price', agg='mean')

labor_q = build_estimates(
    labor_q, last_date=wage_blended.index.max(),
    value_name='wage', agg='sum')

labor_merged = labor_q.merge(df, on=['Ticker','Period','Period Start','Period End'], how='left')
labor_merged['capture'] = labor_merged['actual'] / labor_merged['Labor ($K)']

labor_cost_mdled = scale_by_capture(labor_merged)

actuals_opex_mix = df.merge(calendar, on=['Ticker','Period','Period Start','Period End'], how='right')
actuals_opex_mix['total_cogs'] = actuals_opex_mix['Other Costs in COGS ($K)']

w = 4
actuals_opex_mix['Period Start']    = pd.to_datetime(actuals_opex_mix['Period Start'])
actuals_opex_mix['actual_scaled']   = actuals_opex_mix['total_cogs']   # realized, NaN where unreported
actuals_opex_mix['estimate_scaled'] = np.nan                            # model fills in below

for _, g in actuals_opex_mix.groupby('Ticker', sort=False):
    g    = g.sort_values('Period Start')
    val  = g['total_cogs'].to_numpy(float)
    roll = pd.Series(val).rolling(w, min_periods=w).mean()
    growth = (roll.shift(1) / roll.shift(1 + w) - 1).ffill().to_numpy()
    pred = np.full(len(g), np.nan)
    for p in range(w, len(g)):
        comp = val[p - w] if not np.isnan(val[p - w]) else pred[p - w]
        if np.isnan(comp) or np.isnan(growth[p]):
            continue
        pred[p] = comp * (1.0 + growth[p])
    actuals_opex_mix.loc[g.index, 'estimate_scaled'] = pred

# combine together the estimates to get to the profit number
base_cols = ['Ticker', 'Period', 'Period Start', 'Period End']
# base_cols_woth_estimate = base_cols + ['estimate_scaled']
base_cols_with_estimate = base_cols + ['actual_scaled', 'estimate_scaled']
frames = {'rev': rev_mdled, 'labor': labor_cost_mdled, 'food': food_cost_mdled, 'opex': actuals_opex_mix}
profit_actuals = df[base_cols + ['Restaurant-Level Profit ($K)']].merge(calendar, on=base_cols, how='right')

parts = [
    df[base_cols + ['actual_scaled', 'estimate_scaled']]
    for k, df in frames.items()
]

f_k = list(frames.keys())
merged = profit_actuals.copy()
for i, part in enumerate(parts):
    part_renamed = part.rename(columns={'actual_scaled':f'actual_{f_k[i]}','estimate_scaled':f'estimate_{f_k[i]}'})
    merged = merged.merge(part_renamed, on=base_cols, how='outer')

merged['Profit Estimate'] = merged['estimate_rev'] - merged['estimate_labor'] - merged['estimate_food'] - merged['estimate_opex']

merged['Period Start'] = pd.to_datetime(merged['Period Start'])
merged = merged.sort_values(['Ticker', 'Period Start'])

ape = (merged['Profit Estimate'] - merged['Restaurant-Level Profit ($K)']).abs() \
      / merged['Restaurant-Level Profit ($K)']

merged['mape_4q'] = (
    ape.groupby(merged['Ticker'])
       .transform(lambda s: s.rolling(4, min_periods=4).mean().ffill())
)