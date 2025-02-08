import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

st.set_page_config(layout="wide")

st.title("Options Pricing Model")

# Sidebar for input parameters
st.sidebar.header("Input Parameters")

# Input parameters
spot_price = st.sidebar.number_input("Spot Price (₹)", min_value=0.01, value=100.0)
strike_price = st.sidebar.number_input("Strike Price (₹)", min_value=0.01, value=100.0)
time_to_expiry = st.sidebar.number_input(
    "Time to Expiry (years)", min_value=0.01, value=1.0
)

# Risk-free rate input
if "risk_free_rate" not in st.session_state:
    st.session_state.risk_free_rate = 5.0


def update_rf_rate():
    st.session_state.risk_free_rate = st.session_state.rf_slider


def update_rf_input():
    st.session_state.risk_free_rate = st.session_state.rf_input


st.sidebar.write("Risk-free Rate (%)")
col1, col2 = st.sidebar.columns([1, 1])
with col1:
    risk_free_rate_input = st.number_input(
        "",
        min_value=0.0,
        max_value=10.0,
        value=st.session_state.risk_free_rate,
        step=0.05,
        key="rf_input",
        on_change=update_rf_input,
    )
with col2:
    risk_free_rate_slider = st.slider(
        "",
        min_value=0.0,
        max_value=10.0,
        value=st.session_state.risk_free_rate,
        step=0.05,
        key="rf_slider",
        on_change=update_rf_rate,
    )

risk_free_rate = st.session_state.risk_free_rate / 100

# Volatility input
if "volatility" not in st.session_state:
    st.session_state.volatility = 20.0


def update_volatility():
    st.session_state.volatility = st.session_state.vol_slider


def update_vol_input():
    st.session_state.volatility = st.session_state.vol_input


st.sidebar.write("Volatility (%)")
col1, col2 = st.sidebar.columns([1, 1])
with col1:
    volatility_input = st.number_input(
        "",
        min_value=0.0,
        max_value=100.0,
        value=st.session_state.volatility,
        step=0.01,
        key="vol_input",
        on_change=update_vol_input,
    )
with col2:
    volatility_slider = st.slider(
        "",
        min_value=0.0,
        max_value=100.0,
        value=st.session_state.volatility,
        step=0.01,
        key="vol_slider",
        on_change=update_volatility,
    )

volatility = st.session_state.volatility / 100


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
st.subheader("Option Price Heatmaps")

# Create a range of values for spot price and volatility
spot_range = np.round(np.linspace(spot_price * 0.8, spot_price * 1.2, 10), 2)
vol_range = np.round(np.linspace(volatility * 0.8, volatility * 1.2, 10) * 100, 2)

# Calculate option prices for different spot prices and volatilities
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

# Create DataFrames for the heatmaps
df_call = pd.DataFrame(call_prices, index=vol_range, columns=spot_range)
df_put = pd.DataFrame(put_prices, index=vol_range, columns=spot_range)

# Create a custom colormap
put_color = "#dc3545"  # Red
call_color = "#28a745"  # Green
custom_cmap = mcolors.LinearSegmentedColormap.from_list(
    "custom", [put_color, "white", call_color]
)

# Create and display the heatmaps side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Call Option Heatmap
sns.heatmap(df_call, ax=ax1, cmap=custom_cmap, annot=True, fmt=".2f", cbar=False)
ax1.set_title("Call Option Price Heatmap")
ax1.set_xlabel("Spot Price")
ax1.set_ylabel("Volatility (%)")

# Put Option Heatmap
sns.heatmap(df_put, ax=ax2, cmap=custom_cmap, annot=True, fmt=".2f", cbar=False)
ax2.set_title("Put Option Price Heatmap")
ax2.set_xlabel("Spot Price")
ax2.set_ylabel("Volatility (%)")

# Adjust layout and display the plot
plt.tight_layout()
st.pyplot(fig)
