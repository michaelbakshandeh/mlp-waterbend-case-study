"""Run the case-study pipeline once and pickle the dataframes the Streamlit app needs.

The pipeline hits BLS, FRED, and QCEW. Expect a multi-minute runtime. Re-run on
transient API failures.
"""

import pathlib
import runpy

BASE = pathlib.Path(__file__).parent
DATA = BASE / "data"
DATA.mkdir(exist_ok=True)

ns = runpy.run_path(str(BASE / "mlp_waterbend_case_study.py"), run_name="__main__")

base_cols = ["Ticker", "Period", "Period Start", "Period End"]
filing_actuals = ns["df"][base_cols + [
    "Restaurant Sales ($K)", "Food & Distribution ($K)", "Labor ($K)"
]]
merged = ns["merged"].merge(filing_actuals, on=base_cols, how="left")

ns["profitability"].to_pickle(DATA / "profitability.pkl")
merged.to_pickle(DATA / "merged.pkl")

print(f"profitability: {ns['profitability'].shape}")
print(f"merged: {merged.shape}, tickers: {sorted(merged['Ticker'].unique())}")
