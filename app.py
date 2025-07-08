import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime, date
import base64

st.set_page_config(page_title="Options Pricing Model", layout="wide")

st.markdown(
    """
<style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 250px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 250px;
        margin-left: -250px;
    }
    .streamlit-expanderHeader {
        font-size: 0.9em !important;
    }
    .stNumberInput label, .stSlider label, .stDateInput label, .stRadio label {
        font-size: 0.85em !important;
    }
    .stNumberInput input, .stSlider input, .stDateInput input, .stRadio input {
        font-size: 0.8em !important;
    }
    /* Hide the increment/decrement buttons */
    .stNumberInput div[data-baseweb="input"] div[role="spinbutton"] button {
        display: none !important;
    }
    .stNumberInput div[data-baseweb="input"] div[role="spinbutton"] input {
        text-align: center;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Options Pricing Model")

linkedin_url = "https://www.linkedin.com/in/abhishek-upadhya-647a58257/"


def get_base64(bin_data):
    return base64.b64encode(bin_data).decode("utf-8")


def set_background(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    b64_str = get_base64(data)
    st.sidebar.markdown(
        f"""
    <div style="margin-bottom: 20px;">
        <p style="font-size: 0.9em; color: #888;">
            Created by: Abhishek Upadhya
            <a href="{linkedin_url}" target="_blank">
                <img src="data:image/png;base64,{b64_str}" 
                     alt="LinkedIn" 
                     width="20" 
                     style="vertical-align: middle; margin-left: 5px;">
            </a>
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

set_background("linkedin-icon.png")

# Sidebar for input parameters
st.sidebar.header("Input Parameters")

# Input parameters
spot_price = float(st.sidebar.text_input("Spot Price (₹)", value=1000.0))
strike_price = float(st.sidebar.text_input("Strike Price (₹)", value=1000.0))

st.sidebar.markdown("---")

# Buy date selection
buy_date_option = st.sidebar.radio("Select Purchase Date", ["Today", "Custom Date"])

if buy_date_option == "Today":
    buy_date = date.today()
else:
    buy_date = st.sidebar.date_input("Purchase Date", value=date.today())

# Expiration date input
expiration_date = st.sidebar.date_input(
    "Expiration Date", min_value=buy_date, value=buy_date + pd.Timedelta(days=30)
)

# Calculate time to expiry in years
business_days = pd.bdate_range(start=buy_date, end=expiration_date).shape[0] - 1
delta_days = (expiration_date - buy_date).days
time_to_expiry = delta_days / 365

# Display calculated time to expiry
st.sidebar.write(
    f"Time to Expiry: {business_days} business days ({time_to_expiry:.2f} years)"
)

st.sidebar.markdown("---")

# Risk-free rate input
risk_free_rate = st.sidebar.text_input("Risk-free Rate (%)", value="5.0")
risk_free_rate = float(risk_free_rate) / 100 if risk_free_rate else 0.05

# Volatility input
volatility = st.sidebar.text_input("Volatility (%)", value="20.0")
volatility = float(volatility) / 100 if volatility else 0.20

st.sidebar.markdown("---")

# Purchase price inputs
call_purchase_price = float(st.sidebar.text_input(
    "Call Option Purchase Price (₹)", value=10.0
))
put_purchase_price = float(st.sidebar.text_input(
    "Put Option Purchase Price (₹)", value=10.0
))

st.sidebar.markdown("---")

# Heatmap settings
st.sidebar.subheader("Heatmap Settings")
spot_range_min = float(st.sidebar.text_input(
    "Minimum Spot Price", value=spot_price * 0.8
))
spot_range_max = float(st.sidebar.text_input(
    "Maximum Spot Price", value=spot_price * 1.2
))
vol_range_min = float(st.sidebar.text_input(
    "Minimum Volatility (%)", value=volatility * 80
))
vol_range_max = float(st.sidebar.text_input(
    "Maximum Volatility (%)", value=volatility * 120
))

# Main content area
def black_scholes(S, K, T, r, sigma, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "Call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price


# Calculate option prices
call_price = black_scholes(
    spot_price, strike_price, time_to_expiry, risk_free_rate, volatility, "Call"
)
put_price = black_scholes(
    spot_price, strike_price, time_to_expiry, risk_free_rate, volatility, "Put"
)

# Custom CSS for colored boxes and labels
st.markdown(
    """
<style>
    .price-container {
        text-align: center;
        margin-bottom: 20px;
    }
    .price-label {
        font-size: 1em;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .price-box {
        padding: 10px;
        border-radius: 5px;
        color: white;
        text-align: center;
        font-weight: bold;
        width: 100%;
        height: 50px;  /* Reduced height */
        display: flex;
        justify-content: center;
        align-items: center;
        box-sizing: border-box;
    }
    .call-price {
        background-color: #28a745;
    }
    .put-price {
        background-color: #dc3545;
    }
    .price-box h2 {
        font-size: 1.2em;
        margin: 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Display option prices in colored boxes with labels outside
col1, col2 = st.columns(2)
with col1:
    st.markdown(
        """
    <div class="price-container">
        <div class="price-label">Call Option Price</div>
        <div class="price-box call-price">
            <h2>₹{:.2f}</h2>
        </div>
    </div>
    """.format(
            call_price
        ),
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        """
    <div class="price-container">
        <div class="price-label">Put Option Price</div>
        <div class="price-box put-price">
            <h2>₹{:.2f}</h2>
        </div>
    </div>
    """.format(
            put_price
        ),
        unsafe_allow_html=True,
    )

import matplotlib.colors as mcolors

# Heatmap section
st.subheader("Option Price and PnL Heatmaps")

# Create a range of values for spot price and volatility
spot_range = np.round(np.linspace(spot_range_min, spot_range_max, 10), 2)
vol_range = np.round(np.linspace(vol_range_min, vol_range_max, 10), 2)[::-1]

# Calculate option prices and PnL for different spot prices and volatilities
call_prices = [
    [
        black_scholes(s, strike_price, time_to_expiry, risk_free_rate, v / 100, "Call")
        for s in spot_range
    ]
    for v in vol_range
]
put_prices = [
    [
        black_scholes(s, strike_price, time_to_expiry, risk_free_rate, v / 100, "Put")
        for s in spot_range
    ]
    for v in vol_range
]
call_pnl = [[price - call_purchase_price for price in row] for row in call_prices]
put_pnl = [[price - put_purchase_price for price in row] for row in put_prices]

# Create DataFrames for the heatmaps
df_call = pd.DataFrame(call_prices, index=vol_range, columns=spot_range)
df_put = pd.DataFrame(put_prices, index=vol_range, columns=spot_range)
df_call_pnl = pd.DataFrame(call_pnl, index=vol_range, columns=spot_range)
df_put_pnl = pd.DataFrame(put_pnl, index=vol_range, columns=spot_range)

# Create custom colormaps
price_cmap = mcolors.LinearSegmentedColormap.from_list(
    "price", ["#dc3545", "white", "#28a745"]
)
pnl_cmap = mcolors.LinearSegmentedColormap.from_list(
    "pnl", ["#dc3545", "white", "#28a745"]
)

# Create and display the heatmaps
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

# Call Option Price Heatmap
sns.heatmap(df_call, ax=ax1, cmap=price_cmap, annot=True, fmt=".2f", cbar=False)
ax1.set_title("Call Option Price Heatmap")
ax1.set_xlabel("Spot Price")
ax1.set_ylabel("Volatility (%)")

# Put Option Price Heatmap
sns.heatmap(df_put, ax=ax2, cmap=price_cmap, annot=True, fmt=".2f", cbar=False)
ax2.set_title("Put Option Price Heatmap")
ax2.set_xlabel("Spot Price")
ax2.set_ylabel("Volatility (%)")

# Normalize PnL heatmaps symmetrically around zero
call_pnl_max = df_call_pnl.max().max()
put_pnl_max = df_put_pnl.max().max()
call_pnl_min = df_call_pnl.min().min()
put_pnl_min = df_put_pnl.min().min()

# Ensure vmin and vmax are different and on opposite sides of zero
if call_pnl_min >= 0 and call_pnl_max > 0:
    vmin, vmax = -call_pnl_max, call_pnl_max
elif call_pnl_min < 0 and call_pnl_max <= 0:
    vmin, vmax = call_pnl_min, -call_pnl_min
else:
    vmin, vmax = call_pnl_min, call_pnl_max
call_norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

# Ensure vmin and vmax are different and on opposite sides of zero
if put_pnl_min >= 0 and put_pnl_max > 0:
    vmin, vmax = -put_pnl_max, put_pnl_max
elif put_pnl_min < 0 and put_pnl_max <= 0:
    vmin, vmax = put_pnl_min, -put_pnl_min
else:
    vmin, vmax = put_pnl_min, put_pnl_max
put_norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

# Call Option PnL Heatmap
sns.heatmap(
    df_call_pnl,
    ax=ax3,
    cmap=pnl_cmap,
    annot=True,
    fmt=".2f",
    cbar=False,
    norm=call_norm,
)
ax3.set_title("Call Option PnL Heatmap")
ax3.set_xlabel("Spot Price")
ax3.set_ylabel("Volatility (%)")

# Put Option PnL Heatmap
sns.heatmap(
    df_put_pnl, ax=ax4, cmap=pnl_cmap, annot=True, fmt=".2f", cbar=False, norm=put_norm
)
ax4.set_title("Put Option PnL Heatmap")
ax4.set_xlabel("Spot Price")
ax4.set_ylabel("Volatility (%)")

# Adjust layout and display the plot
plt.tight_layout()
st.pyplot(fig)
