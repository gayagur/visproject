# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import logging

from dash import Dash, html, dcc, Input, Output, State, ALL, callback_context
import dash_bootstrap_components as dbc

import plotly.graph_objects as go
import plotly.io as pio

# Disable Jupyter integration
os.environ["DASH_DISABLE_JUPYTER"] = "True"

# Disable Flask/Dash request logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# ========================= CONFIG =========================
CSV_PATH = os.getenv(
    "CAR_CSV_PATH",
    os.path.join(os.path.dirname(__file__), "cars_dataset_cleaned.csv"),
)
YEAR_NOW = 2025


# ========================= DATA LOADING =========================
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)

    # Convert price to numeric (handle strings with commas)
    if df["price"].dtype == "object":
        df["price"] = pd.to_numeric(
            df["price"].astype(str).str.replace(",", ""), errors="coerce"
        )
    else:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Convert mileage to numeric (handle strings with commas)
    if df["mileage"].dtype == "object":
        df["mileage"] = pd.to_numeric(
            df["mileage"].astype(str).str.replace(",", ""), errors="coerce"
        )
    else:
        df["mileage"] = pd.to_numeric(df["mileage"], errors="coerce")

    # Convert owner_count to numeric if it exists
    if "owner_count" in df.columns:
        if df["owner_count"].dtype == "object":
            df["owner_count"] = pd.to_numeric(
                df["owner_count"].astype(str).str.replace(",", ""), errors="coerce"
            )
        else:
            df["owner_count"] = pd.to_numeric(df["owner_count"], errors="coerce")

    # Calculate vehicle age
    df["age"] = YEAR_NOW - df["on_road_year"]

    # Extract manufacturer from vehicle (first token) - only if column doesn't exist
    if "manufacturer" not in df.columns:
        df["manufacturer"] = df["vehicle"].apply(
            lambda x: str(x).split()[0] if pd.notna(x) else "Unknown"
        )

    # Remove rows with missing critical data
    df = df.dropna(subset=["price", "mileage", "on_road_year"]).copy()

    # Ensure selected columns are strings (avoid mixed types in dropdowns)
    for col in ["vehicle", "transmission", "fuel_type", "body_type", "color", "drive_type"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df


df = load_data(CSV_PATH)

COUNTRIES = sorted(df["country"].unique().tolist())
FEATURES = [
    ("fuel_type", "Fuel Type"),
    ("body_type", "Body Type"),
    ("transmission", "Transmission"),
    ("color", "Color"),
    ("drive_type", "Drive Type"),
]
TOP_MODELS = df["vehicle"].value_counts().head(20).index.tolist()
YEAR_MIN = int(df["on_road_year"].min())
YEAR_MAX = int(df["on_road_year"].max())

# ========================= PLOTLY THEME =========================
# Distinct high-contrast color palette for charts
COLORS = {
    # Primary blues - from dark to light (for UI elements)
    "navy":         "#001D39",  # Darkest navy
    "dark_blue":    "#0A4174",  # Dark blue
    "medium_blue":  "#49769F",  # Medium blue
    "teal":         "#4E8EA2",  # Teal blue
    "light_teal":   "#6EA2B3",  # Light teal
    "sky_blue":     "#7BBDE8",  # Sky blue
    "pale_blue":    "#BDD8E9",  # Pale blue
    
    # Chart colors - DISTINCT and HIGH CONTRAST
    "cyan":    "#0A4174",   # Dark blue
    "orange":  "#E07B39",   # Warm orange
    "purple":  "#7B4B94",   # Rich purple
    "lime":    "#2D8659",   # Forest green
    "indigo":  "#C74B50",   # Coral red
    "magenta": "#D4A03A",   # Golden yellow
    "blue":    "#3B7BC0",   # Medium blue

    # Status colors
    "green":   "#2D8659",  # Forest green - Positive
    "yellow":  "#D4A03A",  # Golden - Neutral
    "red":     "#C74B50",  # Coral red - Caution
}

# High-contrast colors for line charts - easily distinguishable
COLOR_SCALE = [
    "#0A4174",   # Dark navy blue
    "#E07B39",   # Warm orange
    "#2D8659",   # Forest green
    "#7B4B94",   # Rich purple
    "#C74B50",   # Coral red
    "#D4A03A",   # Golden yellow
    "#3B7BC0",   # Medium blue
    "#1D9A8C",   # Teal
]

car_template = go.layout.Template(
    layout=dict(
        font=dict(family="Inter, system-ui, sans-serif", size=13, color="#0F172A"),
        paper_bgcolor="rgba(255, 255, 255, 0.0)",
        plot_bgcolor="rgba(255, 255, 255, 0.0)",
        margin=dict(l=60, r=40, t=80, b=60),
        title=dict(
            x=0.5,
            xref="paper",
            xanchor="center",
            y=0.95,
            yanchor="top",
            font=dict(size=18, color="#001D39", family="Inter"),
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.08,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.98)",
            bordercolor="rgba(10, 65, 116, 0.1)",
            borderwidth=1,
            font=dict(color="#0F172A"),
        ),
        xaxis=dict(
            gridcolor="rgba(10, 65, 116, 0.06)",
            zeroline=False,
            linecolor="rgba(10, 65, 116, 0.1)",
            tickcolor="rgba(10, 65, 116, 0.1)",
            color="#334155",
            fixedrange=True,
        ),
        yaxis=dict(
            gridcolor="rgba(10, 65, 116, 0.06)",
            zeroline=False,
            linecolor="rgba(10, 65, 116, 0.1)",
            tickcolor="rgba(10, 65, 116, 0.1)",
            color="#334155",
            fixedrange=True,
        ),
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 0.98)",
            bordercolor="rgba(10, 65, 116, 0.15)",
            font=dict(family="Inter", color="#001D39"),
        ),
    )
)
pio.templates["car_theme"] = car_template
pio.templates.default = "car_theme"

# ========================= APP =========================
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server
app.title = "Premium Car Analytics"


# Add custom CSS for premium hover effects and animations
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Premium hover effects */
            #calc-info-button:hover,
            #buyer-info-button:hover {
                background: rgba(10, 65, 116, 0.08) !important;
                border-color: rgba(10, 65, 116, 0.3) !important;
                color: #0A4174 !important;
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(0, 29, 57, 0.08);
            }
            .deal-card-hover {
                position: relative;
                background: #FFFFFF !important;
            }
            .deal-card-hover:hover {
                transform: translateY(-8px) !important;
                box-shadow: 0 16px 40px rgba(0, 29, 57, 0.12) !important;
                border-color: #3B82F6 !important;
                z-index: 10 !important;
            }
            .best-deals-container {
                position: relative;
            }
            .best-deals-container > * {
                position: relative;
            }
            .deal-card-hover:hover .deal-card-hover-text {
                opacity: 1 !important;
            }
            .deal-card-hover:hover > *:not(.deal-card-hover-text) {
                opacity: 0.8;
            }
            /* Link hover effects */
            a:hover {
                color: #0A4174 !important;
            }
            /* Smooth transitions for all interactive elements */
            .graph-card, .filter-card, .kpi-card {
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            }
            
            /* === DEPRECIATION ANALYSIS ANIMATIONS === */
            .depreciation-item {
                animation: slideInUp 0.4s ease-out forwards;
                opacity: 0;
            }
            .depreciation-item:nth-child(1) { animation-delay: 0.1s; }
            .depreciation-item:nth-child(2) { animation-delay: 0.2s; }
            .depreciation-item:nth-child(3) { animation-delay: 0.3s; }
            .depreciation-item:nth-child(4) { animation-delay: 0.4s; }
            .depreciation-item:nth-child(5) { animation-delay: 0.5s; }
            
            @keyframes slideInUp {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .depreciation-item:hover {
                transform: translateX(8px) !important;
                box-shadow: 0 8px 24px rgba(0, 29, 57, 0.1) !important;
            }
            
            /* Progress bar animation */
            .depreciation-bar-fill {
                animation: growWidth 1s ease-out forwards;
                transform-origin: left;
            }
            
            @keyframes growWidth {
                from {
                    transform: scaleX(0);
                }
                to {
                    transform: scaleX(1);
                }
            }
            
            /* Percentage counter animation */
            .depreciation-percentage {
                animation: fadeInScale 0.5s ease-out forwards;
            }
            
            @keyframes fadeInScale {
                from {
                    opacity: 0;
                    transform: scale(0.8);
                }
                to {
                    opacity: 1;
                    transform: scale(1);
                }
            }
            
             /* Pulse effect for status dot */
             .status-dot {
                 animation: pulse 2s ease-in-out infinite;
             }
             
             @keyframes pulse {
                 0%, 100% {
                     box-shadow: 0 0 0 0 currentColor;
                     opacity: 1;
                 }
                 50% {
                     box-shadow: 0 0 0 6px transparent;
                     opacity: 0.8;
                 }
             }
             
             /* Landing page animations */
             @keyframes fadeInUp {
                 from {
                     opacity: 0;
                     transform: translateY(30px);
                 }
                 to {
                     opacity: 1;
                     transform: translateY(0);
                 }
             }
             
             .feature-card-animated {
                 animation: fadeInUp 0.6s ease-out forwards;
                 opacity: 0;
             }
             
             .feature-card-animated:nth-child(1) { animation-delay: 0.1s; }
             .feature-card-animated:nth-child(2) { animation-delay: 0.2s; }
             .feature-card-animated:nth-child(3) { animation-delay: 0.3s; }
             
             .feature-card-modern {
                 transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                 position: relative;
                 overflow: visible;
             }
             
             .feature-card-modern:hover {
                 transform: translateY(-4px) scale(1.02);
                 box-shadow: 0 20px 40px rgba(0, 29, 57, 0.15) !important;
             }
             
             .feature-icon-circle {
                 width: 80px;
                 height: 80px;
                 border-radius: 50%;
                 display: flex;
                 align-items: center;
                 justify-content: center;
                 margin: 0 auto -40px auto;
                 position: relative;
                 z-index: 10;
                 box-shadow: 0 8px 24px rgba(59, 130, 246, 0.3);
                 transition: all 0.3s ease;
             }
             
             .feature-card-modern:hover .feature-icon-circle {
                transform: scale(1.1);
                 box-shadow: 0 12px 32px rgba(59, 130, 246, 0.4);
             }
             
             .gradient-text {
                 background: linear-gradient(135deg, #3B82F6 0%, #1E40AF 50%, #0A4174 100%);
                 -webkit-background-clip: text;
                 -webkit-text-fill-color: transparent;
                 background-clip: text;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


# ========================= KPI CARD =========================
def kpi_card(label, value, sub, icon_text, gradient="blue-purple"):
    # Premium Ocean Blue palette with hierarchy
    # First card (blue-purple) is primary/featured
    card_styles = {
        "blue-purple": {
            "bg": "linear-gradient(145deg, #FFFFFF 0%, #F0F7FA 100%)",
            "text": "#001D39",  # Darkest for primary
            "icon_bg": "linear-gradient(135deg, #0A4174 0%, #49769F 100%)",
            "icon_color": "#FFFFFF",
            "is_primary": True,
        },
        "cyan-blue": {
            "bg": "#FFFFFF",
            "text": "#0A4174",
            "icon_bg": "rgba(10, 65, 116, 0.08)",
            "icon_color": "#0A4174",
            "is_primary": False,
        },
        "orange-red": {
            "bg": "#FFFFFF",
            "text": "#49769F",
            "icon_bg": "rgba(73, 118, 159, 0.08)",
            "icon_color": "#49769F",
            "is_primary": False,
        },
        "green-cyan": {
            "bg": "#FFFFFF",
            "text": "#4E8EA2",
            "icon_bg": "rgba(78, 142, 162, 0.08)",
            "icon_color": "#4E8EA2",
            "is_primary": False,
        },
    }
    
    style = card_styles.get(gradient, card_styles["blue-purple"])
    
    # Primary card gets extra emphasis
    card_style = {
        "background": style["bg"],
    }
    if style.get("is_primary"):
        card_style["boxShadow"] = "0 8px 32px rgba(0, 29, 57, 0.12)"

    return html.Div(
        className="kpi-card",
        style=card_style,
        children=[
            html.Div(
                icon_text,
                className="kpi-icon",
                style={
                    "background": style["icon_bg"], 
                    "color": style["icon_color"],
                    "boxShadow": "0 4px 12px rgba(0, 29, 57, 0.08)" if style.get("is_primary") else "none",
                },
            ),
            html.Div(value, className="kpi-value", style={"color": style["text"]}),
            html.Div(label, className="kpi-label"),
        ],
    )


# ========================= FIGURES =========================
def fig_donut(counts: pd.Series, title: str, subtitle: str) -> go.Figure:
    labels = counts.index.tolist()
    values = counts.values.tolist()

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.65,
                textinfo="label+percent",
                textposition="outside",
                textfont=dict(size=13, color="#1E293B", family="Inter"),
                hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Percent: %{percent}<extra></extra>",
                marker=dict(
                    colors=COLOR_SCALE,
                    line=dict(color="rgba(255, 255, 255, 0.9)", width=2),
                ),
                pull=[0.02] * len(labels),
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size:14px;color:#64748B'>{subtitle}</span>"
        ),
        height=550,
        showlegend=False,
    )

    return fig


def fig_smart_buyer_matrix(
    data: pd.DataFrame,
    selected_vehicles: list = None,
    max_price: float = None,
    min_price: float = None,
    max_mileage: float = None,
        country: list = None,
        transmission: list = None,
        fuel_type: list = None,
        owner_count: list = None,
    year_range: list = None,
    max_vehicles: int = 12,
) -> go.Figure:
    # Smart Buyer Matrix: a bubble scatter of vehicle-level avg mileage vs avg price,
    # with a derived "value score" and uncertainty bands.

    dff = data.copy()

    # Price filters
    if min_price:
        dff = dff[dff["price"] >= min_price]
    if max_price:
        dff = dff[dff["price"] <= max_price]

    # Mileage filter
    if max_mileage:
        dff = dff[dff["mileage"] <= max_mileage]

    # Country filter (multiple selection)
    if country and len(country) > 0:
        dff = dff[dff["country"].isin(country)]

    # Transmission filter (multiple selection)
    if transmission and len(transmission) > 0:
        dff = dff[dff["transmission"].isin(transmission)]

    # Fuel filter (multiple selection)
    if fuel_type and len(fuel_type) > 0:
        dff = dff[dff["fuel_type"].isin(fuel_type)]

    # Owner count filter (multiple selection)
    if owner_count and len(owner_count) > 0 and "owner_count" in dff.columns:
        dff = dff[dff["owner_count"].isin(owner_count)]

    # Year filter
    if year_range:
        dff = dff[
            (dff["on_road_year"] >= year_range[0]) & (dff["on_road_year"] <= year_range[1])
        ]

    # Vehicle selection (or fallback to top N by frequency)
    if selected_vehicles and len(selected_vehicles) > 0:
        dff = dff[dff["vehicle"].isin(selected_vehicles)]
    else:
        top_vehicles = dff["vehicle"].value_counts().head(max_vehicles).index
        dff = dff[dff["vehicle"].isin(top_vehicles)]

    if len(dff) == 0:
        fig = go.Figure()
        fig.update_layout(
            title=dict(
                text="<b>No data matches your filters</b><br><span style='font-size:14px;color:#6B7280'>Try adjusting your criteria</span>"
            ),
            height=650,
        )
        return fig, []

    # Aggregate per vehicle model
    # Compute: AvgPrice, StdPrice, AvgMileageKm, StdMileageKm, Count
    vehicle_stats = (
        dff.groupby("vehicle")
        .agg({"price": ["mean", "std", "count"], "mileage": ["mean", "std"]})
        .reset_index()
    )
    vehicle_stats.columns = ["vehicle", "avg_price", "price_std", "count", "avg_mileage_km", "mileage_std"]
    
    # Rename for clarity (mileage is already in km)
    vehicle_stats = vehicle_stats.rename(columns={"avg_mileage_km": "avg_mileage"})

    # Filter out models with insufficient data or invalid mileage
    # Exclude models with Count < 2 (low confidence) or AvgMileageKm <= 0
    vehicle_stats = vehicle_stats[
        (vehicle_stats["count"] >= 2) & (vehicle_stats["avg_mileage"] > 0)
    ].copy()

    if len(vehicle_stats) == 0:
        fig = go.Figure()
        fig.update_layout(
            title=dict(
                text="<b>No models with sufficient data</b><br><span style='font-size:14px;color:#6B7280'>Need at least 2 listings per model with valid mileage</span>"
            ),
            height=650,
        )
        return fig, []

    # Compute Price per km (PPK) per model
    # AvgPPK = AvgPrice / AvgMileageKm
    # If AvgMileageKm <= 0, set AvgPPK = NaN and exclude from scoring
    # Since we already filtered avg_mileage > 0, division is safe
    vehicle_stats["avg_ppk"] = vehicle_stats["avg_price"] / vehicle_stats["avg_mileage"]
    
    # Sanity check: all avg_mileage should be > 0 after filtering
    assert (vehicle_stats["avg_mileage"] > 0).all(), "All avg_mileage must be > 0"
    
    # Drop any NaN cases (shouldn't happen after filtering, but for robustness)
    vehicle_stats = vehicle_stats.dropna(subset=["avg_ppk"])

    # Scoring: Use AvgPPK directly as "worse if larger"
    # RawScore = AvgPPK (units: currency/km)
    # Normalize to 0-100 and invert so higher means better value
    raw_score = vehicle_stats["avg_ppk"]
    min_raw = raw_score.min()
    max_raw = raw_score.max()
    
    if max_raw > min_raw:
        # Invert: lower PPK = better value, so higher normalized score
        vehicle_stats["value_normalized"] = 100 * (1 - (raw_score - min_raw) / (max_raw - min_raw))
    else:
        # All models have same PPK
        vehicle_stats["value_normalized"] = 50
    
    # Clip to [0, 100] for safety
    vehicle_stats["value_normalized"] = vehicle_stats["value_normalized"].clip(0, 100)
    
    # Sanity check: value_normalized should be in [0, 100]
    assert (vehicle_stats["value_normalized"] >= 0).all() and (vehicle_stats["value_normalized"] <= 100).all(), \
        "value_normalized must be in [0, 100]"

    # Bubble size based on listing count
    min_count = vehicle_stats["count"].min()
    max_count = vehicle_stats["count"].max()
    if max_count > min_count:
        vehicle_stats["bubble_size"] = 20 + (
            (vehicle_stats["count"] - min_count) / (max_count - min_count) * 50
        )
    else:
        vehicle_stats["bubble_size"] = 35

    fig = go.Figure()

    for _, row in vehicle_stats.iterrows():
        color_val = row["value_normalized"]

        # Traffic-light colors based on meaning
        if color_val >= 75:
            color = "#16A34A"  # Green - Excellent value (best deals)
            category = "Excellent Value"
        elif color_val >= 55:
            color = "#65A30D"  # Lime green - Good value
            category = "Good Value"
        elif color_val >= 35:
            color = "#EAB308"  # Yellow/Amber - Fair value
            category = "Fair Value"
        else:
            color = "#DC2626"  # Red - Poor value (overpriced)
            category = "Poor Value"

        fig.add_trace(
            go.Scatter(
                x=[row["avg_mileage"]],
                y=[row["avg_price"]],
                mode="markers+text",
                name=row["vehicle"],
                marker=dict(
                    size=row["bubble_size"],
                    color=color,
                    opacity=0.8,
                    line=dict(color="rgba(255, 255, 255, 0.9)", width=2),
                    sizemode="diameter",
                ),
                text=row["vehicle"].split()[0] if len(row["vehicle"].split()) > 0 else row["vehicle"],
                textposition="top center",
                textfont=dict(size=11, color="#001D39", family="Inter"),
                error_x=dict(
                    type="data",
                    array=[row["mileage_std"]],
                    visible=True,
                    color="rgba(10, 65, 116, 0.2)",
                    thickness=1.5,
                ),
                error_y=dict(
                    type="data",
                    array=[row["price_std"]],
                    visible=True,
                    color="rgba(10, 65, 116, 0.2)",
                    thickness=1.5,
                ),
                hovertemplate=f"<b style='color:#001D39;font-size:15px;'>%{{fullData.name}}</b><br>"
                + f"<b style='color:{color};'>{category}</b><br><br>"
                + f"<span style='color:#001D39;'>Avg Price: ₪{row['avg_price']:,.0f}</span><br>"
                + f"<span style='color:#49769F;'>Std Price: ₪{row['price_std']:,.0f}</span><br>"
                + f"<span style='color:#001D39;'>Avg Mileage: {row['avg_mileage']:,.0f} km</span><br>"
                + f"<span style='color:#49769F;'>Std Mileage: {row['mileage_std']:,.0f} km</span><br>"
                + f"<span style='color:#001D39;'>Available: {row['count']:,} cars</span><br>"
                + f"<span style='color:#49769F;'>Avg Price/km: ₪{row['avg_ppk']:.2f}</span><br>"
                + f"<span style='color:{color};font-weight:600;'>Value Score: {color_val:.0f}</span><br>"
                + "<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text="<b>Smart Buyer Matrix</b><br>"
                 + f"<span style='font-size:14px;color:#6B7280'>Analyzing {len(dff):,} vehicles across {len(vehicle_stats)} models</span><br>"
                 + "<span style='font-size:11px;color:#94A3B8;font-weight:normal'>Drag to pan • Double-click to reset</span>"
        ),
        xaxis_title="<b>Average Mileage (km)</b>",
        yaxis_title="<b>Average Price (₪)</b>",
        height=650,
        hovermode="closest",
        showlegend=False,
    )

    # --- SPECIFIC UNLOCK FOR THIS CHART ONLY ---
    fig.update_xaxes(fixedrange=False)
    fig.update_yaxes(fixedrange=False)

    # Return the figure and the list of vehicles displayed
    displayed_vehicles = vehicle_stats["vehicle"].tolist()
    return fig, displayed_vehicles


def create_best_deals_cards(data: pd.DataFrame, max_results: int = 10, displayed_vehicles: list = None):
    # Best Deals: compute a per-model z-score for price_per_km and surface listings that
    # are significantly cheaper per km than their model mean.
    # Only shows deals from vehicles that are displayed in the Smart Buyer Matrix.

    dff = data.copy()
    
    # Filter to only vehicles displayed in the matrix
    if displayed_vehicles and len(displayed_vehicles) > 0:
        dff = dff[dff["vehicle"].isin(displayed_vehicles)]
    
    # Calculate Price per km (PPK) for each listing
    # PPK_listing = price / max(mileage_km, 1) to avoid division by zero
    dff["ppk_listing"] = dff["price"] / dff["mileage"].clip(lower=1.0)
    
    # Calculate per-model statistics for PPK
    dff["ppk_zscore"] = np.nan

    for model in dff["vehicle"].unique():
        model_data = dff[dff["vehicle"] == model]
        # Need at least 2 listings to compute std (minimum for meaningful comparison)
        if len(model_data) >= 2:
            # Compute mean and std of PPK for this model
            mean_ppk = model_data["ppk_listing"].mean()
            std_ppk = model_data["ppk_listing"].std()
            
            if std_ppk > 0:
                # Z-score: (PPK_listing - model_mean_PPK) / model_std_PPK
                # Negative z-score = PPK below model mean = better deal
                dff.loc[dff["vehicle"] == model, "ppk_zscore"] = (
                    dff.loc[dff["vehicle"] == model, "ppk_listing"] - mean_ppk
                ) / std_ppk

    # Keep deals that are at least 0.5 std below their model mean PPK
    # Lower PPK = better value, so negative z-score is good
    best_deals = dff[dff["ppk_zscore"] < -0.5].nsmallest(max_results, "ppk_zscore")

    if len(best_deals) == 0:
        return html.Div(
            [
                html.H3(
                    "No significant deals found",
                    style={
                        "fontSize": "18px",
                        "fontWeight": 600,
                        "marginBottom": "8px",
                        "color": "#1A202C",
                    },
                ),
                html.P(
                    "All vehicles are fairly priced within their model range",
                    style={"fontSize": "14px", "color": "#718096"},
                ),
            ],
            style={"padding": "32px", "textAlign": "center"},
        )

    best_deals = best_deals.sort_values("ppk_zscore")

    # Calculate normalized values for color (0-1 range)
    # Invert z-score so more negative (better deal) = higher normalized value
    z_scores_neg = -best_deals["ppk_zscore"].values
    z_min, z_max = z_scores_neg.min(), z_scores_neg.max()
    if z_max > z_min:
        z_normalized = (z_scores_neg - z_min) / (z_max - z_min)
    else:
        z_normalized = np.ones(len(z_scores_neg)) * 0.5

    # Create cards in a grid (4 per row)
    cards = []
    for idx, (_, row) in enumerate(best_deals.iterrows()):
        z_norm = z_normalized[idx]
        z_score_neg = -row["ppk_zscore"]
        
        # Convert normalized z-score (0-1) to Value Score (0-100) to match legend thresholds
        value_score = z_norm * 100
        
        # Color palette matching Value Score legend EXACTLY
        # Use same thresholds as legend: ≥75, 55-74, 35-54, <35
        if value_score >= 75:
            color = "#22c55e"  # Dark green - Excellent (≥75)
            quality = "Excellent"
        elif value_score >= 55:
            color = "#84cc16"  # Light green - Good (55-74)
            quality = "Good"
        elif value_score >= 35:
            color = "#eab308"  # Yellow - Fair (35-54)
            quality = "Fair"
        else:
            color = "#ef4444"  # Red - Poor (<35)
            quality = "Poor"
        
        # Store row index in the card ID for callback
        card_id = f"deal-card-{idx}"
        cards.append(
            html.Div(
                id={"type": "deal-card", "index": idx},
                className="graph-card deal-card-hover",
                n_clicks=0,
                style={
                    "padding": "20px",
                    "minWidth": "260px",
                    "width": "260px",
                    "flexShrink": 0,
                    "border": "1.5px solid #93C5FD",
                    "borderRadius": "10px",
                    "background": "#FFFFFF",
                    "boxShadow": "0 1px 3px rgba(0, 0, 0, 0.05)",
                    "transition": "all 0.3s ease",
                    "cursor": "pointer",
                    "marginRight": "14px",
                    "position": "relative",
                },
                children=[
                    # Hover text overlay
                    html.Div(
                        "View details →",
                        className="deal-card-hover-text",
                        style={
                            "position": "absolute",
                            "top": "50%",
                            "left": "50%",
                            "transform": "translate(-50%, -50%)",
                            "color": "#FFFFFF",
                            "fontSize": "13px",
                            "fontWeight": 600,
                            "opacity": 0,
                            "transition": "opacity 0.25s ease",
                            "pointerEvents": "none",
                            "zIndex": 10,
                            "textAlign": "center",
                            "background": "linear-gradient(135deg, #0A4174 0%, #49769F 100%)",
                            "padding": "12px 20px",
                            "borderRadius": "10px",
                            "boxShadow": "0 8px 24px rgba(0, 29, 57, 0.2)",
                        },
                    ),
                    # Color bar at top
                    html.Div(
                        style={
                            "width": "40px",
                            "height": "4px",
                            "background": color,
                            "borderRadius": "2px",
                            "margin": "0 auto 16px auto",
                        },
                    ),
                    html.Div(
                        row["vehicle"][:30] + ("..." if len(row["vehicle"]) > 30 else ""),
                        style={
                            "fontSize": "15px",
                            "fontWeight": 700,
                            "color": "#1A202C",
                            "textAlign": "center",
                            "marginBottom": "16px",
                            "minHeight": "40px",
                        },
                    ),
                    html.Div(
                        [
                            html.Span("Price ", style={"color": "#718096", "fontSize": "12px"}),
                            html.Span(
                                f"₪{row['price']:,.0f}",
                                style={"color": "#1A202C", "fontWeight": 700, "fontSize": "18px"},
                            ),
                        ],
                        style={"marginBottom": "12px", "textAlign": "center"},
                    ),
                    html.Div(
                        f"Below avg: {z_score_neg:.2f} std",
                        style={
                            "color": "#1A202C",
                            "fontSize": "12px",
                            "marginBottom": "10px",
                            "textAlign": "center",
                        },
                    ),
                    html.Div(
                        [
                            html.Span("Rating: ", style={"color": "#1A202C", "fontSize": "12px"}),
                            html.Span(
                                quality,
                                style={"color": color, "fontWeight": 600, "fontSize": "13px"},
                            ),
                        ],
                        style={"textAlign": "center", "marginBottom": "6px"},
                    ),
                    html.Div(
                        "Significantly below model average",
                        style={
                            "color": "#9CA3AF",
                            "fontSize": "10px",
                            "textAlign": "center",
                            "fontStyle": "italic",
                        },
                    ),
                ],
            )
        )

    # Update subtitle based on whether we're filtering by displayed vehicles
    if displayed_vehicles and len(displayed_vehicles) > 0:
        subtitle = f"Best deals from {len(displayed_vehicles)} models shown above"
    else:
        subtitle = "Cars with price per km significantly below their model average"

    return html.Div(
        [
            html.Div(
                [
                    html.H3(
                        f"Top {len(best_deals)} Best Deals",
                        style={
                            "fontSize": "20px",
                            "fontWeight": 700,
                            "marginBottom": "6px",
                            "color": "#374151",
                        },
                    ),
                    html.P(
                        subtitle,
                        style={"fontSize": "13px", "color": "#9CA3AF", "marginBottom": "20px"},
                    ),
                ],
                style={"marginBottom": "20px"},
            ),
            html.Div(
                cards,
                style={
                    "display": "flex",
                    "overflowX": "auto",
                    "overflowY": "visible",
                    "paddingBottom": "20px",
                    "paddingTop": "8px",
                    "paddingLeft": "8px",
                    "paddingRight": "8px",
                    "gap": "0",
                },
                className="best-deals-container",
            ),
        ],
        style={"padding": "22px"},
    )


def fig_price_depreciation(manufacturers: list[str], data: pd.DataFrame) -> tuple[go.Figure, dict]:
    manufacturers = (manufacturers or [])[:5]
    dff = data[data["manufacturer"].isin(manufacturers)].copy()

    if dff.empty:
        fig = go.Figure()
        fig.update_layout(title=dict(text="<b>No data for selection</b>"), height=550)
        return fig, {}

    # Bin mileage into 7 segments, then compute mean price per bin
    dff["mileage_bin"] = pd.cut(dff["mileage"], bins=7)
    line_data = (
        dff.groupby(["manufacturer", "mileage_bin"], observed=False)["price"]
        .mean()
        .reset_index()
    )
    line_data["mileage_center"] = line_data["mileage_bin"].apply(lambda x: x.mid)

    fig = go.Figure()
    depreciation_data = {}

    for idx, manufacturer in enumerate(manufacturers):
        md = line_data[line_data["manufacturer"] == manufacturer].sort_values("mileage_center")
        color = COLOR_SCALE[idx % len(COLOR_SCALE)]

        fig.add_trace(
            go.Scatter(
                x=md["mileage_center"],
                y=md["price"] / 1000,
                mode="lines+markers",
                name=manufacturer,
                line=dict(width=4, color=color, shape="spline"),
                marker=dict(
                    size=10,
                    color=color,
                    line=dict(width=2, color="rgba(31, 41, 55, 0.3)"),
                ),
                hovertemplate="<b>%{fullData.name}</b><br>Mileage: %{x:,.0f} km<br>Price: ₪%{y:,.1f}K<extra></extra>",
            )
        )

        # --- UPDATED DEPRECIATION LOGIC START ---
        # Compare average price of lowest-mileage 20% vs highest-mileage 20%
        raw_manufacturer_data = dff[dff["manufacturer"] == manufacturer].copy()
        if len(raw_manufacturer_data) >= 5:
            sorted_data = raw_manufacturer_data.sort_values("mileage")
            n = len(sorted_data)
            bottom_20pct = sorted_data.head(max(3, int(n * 0.2)))
            top_20pct = sorted_data.tail(max(3, int(n * 0.2)))

            # 1. Calculate Prices
            low_mileage_price = bottom_20pct["price"].mean()
            high_mileage_price = top_20pct["price"].mean()

            # 2. Calculate Mileages (New)
            low_mileage_avg = bottom_20pct["mileage"].mean()
            high_mileage_avg = top_20pct["mileage"].mean()
            
            # Calculate the gap in usage
            mileage_diff = high_mileage_avg - low_mileage_avg

            # Calculate normalized depreciation if valid
            if low_mileage_price > 0 and mileage_diff > 5000:
                # A) Total percentage drop
                total_drop_pct = (low_mileage_price - high_mileage_price) / low_mileage_price
                
                # B) Normalize to "Drop per 10,000 km"
                # Formula: (Total Drop % / Mileage Difference) * 10,000
                score_per_10k = total_drop_pct * (10000 / mileage_diff) * 100
                
                depreciation_data[manufacturer] = {
                    "depreciation_pct": max(0, score_per_10k), # Normalized rate
                    "first_price": low_mileage_price,
                    "last_price": high_mileage_price,
                    "color": color,
                    "sample_size": n
                }
            else:
                # Fallback if not enough data/distance
                depreciation_data[manufacturer] = {
                    "depreciation_pct": 0,
                    "first_price": low_mileage_price,
                    "last_price": high_mileage_price,
                    "color": color,
                    "sample_size": n
                }
        # --- UPDATED DEPRECIATION LOGIC END ---

    fig.update_layout(
        title=dict(text="<b>Price Depreciation by Mileage</b>"),
        height=550,
        hovermode="x unified",
        xaxis_title="<b>Mileage (km)</b>",
        yaxis_title="<b>Price (₪ Thousands)</b>",
    )
    fig.update_xaxes(tickformat=",")

    return fig, depreciation_data


def fig_group_comparison(group_a: pd.DataFrame, group_b: pd.DataFrame):
    def value_for_money(g):
        clean = g[g["mileage"] > 0]
        return float((clean["price"] / clean["mileage"]).mean()) if len(clean) > 0 else 0.0

    metrics_data = {
        "Price per KM": [value_for_money(group_a), value_for_money(group_b)],
        "Price Stability (σ)": [float(group_a["price"].std()), float(group_b["price"].std())],
        "Avg Mileage": [float(group_a["mileage"].mean()), float(group_b["mileage"].mean())],
        "Avg Price": [float(group_a["price"].mean()), float(group_b["price"].mean())],
    }

    metrics = list(metrics_data.keys())
    a_norm, b_norm = [], []

    for m in metrics:
        va, vb = metrics_data[m]
        mx = max(va, vb) if max(va, vb) > 0 else 1
        a_norm.append((va / mx) * 100)
        b_norm.append((vb / mx) * 100)

    fig = go.Figure()

    # Vibrant, professional colors for value comparison
    # Group A: Rich ocean blue with subtle gradient effect
    group_a_color = "#4A90D9"  # Vibrant ocean blue
    group_a_darker = "#3A7BC4"  # Slightly darker for depth
    
    # Group B: Warm, vibrant purple with subtle gradient effect
    group_b_color = "#9F7AEA"  # Warm vibrant purple
    group_b_darker = "#8B6BD9"  # Slightly darker for depth

    fig.add_trace(
        go.Bar(
            y=metrics,
            x=[-v for v in a_norm],
            name="Group A",
            orientation="h",
            marker=dict(
                color=group_a_color,
                line=dict(color="rgba(255, 255, 255, 0.6)", width=1.5),
                # Create subtle gradient effect using a slightly darker base
                opacity=0.95,
            ),
            text=[
                f"₪{metrics_data[m][0]:,.2f}" if ("KM" in m or "Stability" in m) else f"{metrics_data[m][0]:,.0f}"
                for m in metrics
            ],
            textposition="inside",
            textfont=dict(size=13, color="#FFFFFF", family="Inter", weight="bold"),
            hovertemplate="<b>Group A</b><br>%{y}: %{text}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            y=metrics,
            x=b_norm,
            name="Group B",
            orientation="h",
            marker=dict(
                color=group_b_color,
                line=dict(color="rgba(255, 255, 255, 0.6)", width=1.5),
                # Create subtle gradient effect using a slightly darker base
                opacity=0.95,
            ),
            text=[
                f"₪{metrics_data[m][1]:,.2f}" if ("KM" in m or "Stability" in m) else f"{metrics_data[m][1]:,.0f}"
                for m in metrics
            ],
            textposition="inside",
            textfont=dict(size=13, color="#FFFFFF", family="Inter", weight="bold"),
            hovertemplate="<b>Group B</b><br>%{y}: %{text}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text="<b>Vehicle Value Comparison</b>"),
        barmode="relative",
        height=500,
        margin=dict(l=140, r=40, t=80, b=60),  # Increased left margin to prevent label clipping
        xaxis_title="<b>Relative Performance</b>",
        xaxis=dict(
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor="rgba(107, 114, 128, 0.3)",
            range=[-120, 120],
        ),
    )
    fig.update_yaxes(
        autorange="reversed",
        tickfont=dict(size=13, family="Inter"),
        # Ensure labels are not clipped
        automargin=True,
    )

    return fig, metrics_data


# ========================= LAYOUT =========================
app.layout = dbc.Container(
    fluid=True,
    children=[
        # HERO with car illustration
        html.Div(
            className="hero",
            children=[
                # Car illustration with bar chart
                html.Div(
                    style={
                        "display": "flex",
                        "justifyContent": "center",
                        "alignItems": "center",
                        "marginBottom": "20px",
                        "gap": "16px",
                    },
                    children=[
                        # Layered cars - SVG
                        html.Img(
                            src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='60' viewBox='0 0 120 60' fill='none'%3E%3C!-- Back car (grey/faded) --%3E%3Cg opacity='0.4'%3E%3Cellipse cx='25' cy='42' rx='20' ry='8' fill='%234A5568'/%3E%3Crect x='10' y='28' width='30' height='16' rx='4' fill='%234A5568'/%3E%3Crect x='14' y='20' width='22' height='10' rx='3' fill='%234A5568'/%3E%3Ccircle cx='16' cy='44' r='5' fill='%232D3748'/%3E%3Ccircle cx='34' cy='44' r='5' fill='%232D3748'/%3E%3Crect x='16' y='22' width='6' height='5' rx='1' fill='%2390CDF4'/%3E%3Crect x='24' y='22' width='6' height='5' rx='1' fill='%2390CDF4'/%3E%3C/g%3E%3C!-- Middle car (blue) --%3E%3Cg opacity='0.7'%3E%3Cellipse cx='50' cy='44' rx='22' ry='9' fill='%232B6CB0'/%3E%3Crect x='33' y='28' width='34' height='18' rx='5' fill='%233182CE'/%3E%3Crect x='38' y='18' width='24' height='12' rx='3' fill='%233182CE'/%3E%3Ccircle cx='40' cy='46' r='6' fill='%231A365D'/%3E%3Ccircle cx='60' cy='46' r='6' fill='%231A365D'/%3E%3Crect x='40' y='21' width='7' height='6' rx='1' fill='%2390CDF4'/%3E%3Crect x='49' y='21' width='7' height='6' rx='1' fill='%2390CDF4'/%3E%3C/g%3E%3C!-- Front car (red) --%3E%3Cellipse cx='82' cy='46' rx='26' ry='10' fill='%23C53030'/%3E%3Crect x='62' y='28' width='40' height='20' rx='6' fill='%23E53E3E'/%3E%3Crect x='68' y='14' width='28' height='16' rx='4' fill='%23E53E3E'/%3E%3Ccircle cx='72' cy='48' r='7' fill='%231A202C'/%3E%3Ccircle cx='92' cy='48' r='7' fill='%231A202C'/%3E%3Crect x='71' y='18' width='9' height='8' rx='2' fill='%2390CDF4'/%3E%3Crect x='82' y='18' width='9' height='8' rx='2' fill='%2390CDF4'/%3E%3Crect x='64' y='34' width='6' height='4' rx='1' fill='%23FBD38D'/%3E%3Crect x='94' y='34' width='6' height='4' rx='1' fill='%23FBD38D'/%3E%3C/svg%3E",
                            style={"height": "60px"},
                        ),
                        # Separator line
                        html.Div(
                            style={
                                "width": "2px",
                                "height": "50px",
                                "background": "linear-gradient(180deg, transparent, rgba(255,255,255,0.5), transparent)",
                            }
                        ),
                        # Bar chart SVG
                        html.Img(
                            src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='70' height='50' viewBox='0 0 70 50' fill='none'%3E%3Crect x='2' y='30' width='10' height='18' rx='2' fill='%234E8EA2'/%3E%3Crect x='16' y='20' width='10' height='28' rx='2' fill='%2349769F'/%3E%3Crect x='30' y='8' width='10' height='40' rx='2' fill='%230A4174'/%3E%3Crect x='44' y='15' width='10' height='33' rx='2' fill='%2349769F'/%3E%3Crect x='58' y='25' width='10' height='23' rx='2' fill='%237BBDE8'/%3E%3C/svg%3E",
                            style={"height": "50px"},
                        ),
                    ],
                ),
                html.H1("PREMIUM CAR ANALYTICS"),
                html.P("Advanced Vehicle Intelligence & Market Insights Platform"),
            ],
        ),

        # OVERVIEW SECTION
        html.Div(
            className="section",
            children=[
                html.Div("MARKET OVERVIEW", className="section-title"),
                html.Div(
                    "Key performance indicators and market trends at a glance",
                    className="section-sub",
                ),
            ],
        ),

        # KPI ROW
        dbc.Row(
            className="g-4 mb-4",
            children=[
                dbc.Col(
                    kpi_card("Total Listings", f"{len(df):,}", "Active market records", "📋", "blue-purple"),
                    md=3,
                ),
                dbc.Col(
                    kpi_card("Avg Price", f"₪{df['price'].mean():,.0f}", "Mean asking price", "💰", "cyan-blue"),
                    md=3,
                ),
                dbc.Col(
                    kpi_card("Avg Mileage", f"{df['mileage'].mean():,.0f}", "Kilometers driven", "🛣️", "orange-red"),
                    md=3,
                ),
                dbc.Col(
                    kpi_card("Avg Age", f"{df['age'].mean():.1f} yrs", "Vehicle age", "📅", "green-cyan"),
                    md=3,
                ),
            ],
        ),

        # TABS
        dbc.Tabs(
            id="tabs",
            active_tab="tab-home",
            children=[
                dbc.Tab(label="Home", tab_id="tab-home"),
                dbc.Tab(label="Manufacturers Comparison", tab_id="tab-model"),
                dbc.Tab(label="Group Comparison", tab_id="tab-group"),
                dbc.Tab(label="Buyer's Guide", tab_id="tab-buyer"),
            ],
        ),

        html.Div(id="tab-content", style={"marginTop": "24px"}),

        # Store for best deals data
        dcc.Store(id="best-deals-store", data={}),

        # Modal for vehicle details
        dbc.Modal(
            [
                dbc.ModalHeader(
                    html.Div(
                        [
                            html.Div(id="vehicle-modal-title", children="Vehicle Details",
                                     className="vehicle-modal-header-wrapper"),
                            html.Button(
                                "×",
                                className="vehicle-modal-close",
                                n_clicks=0,
                                id="vehicle-modal-close-btn",
                            ),
                        ],
                        className="vehicle-modal-header",
                        style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start",
                               "width": "100%"},
                    ),
                    close_button=False,
                ),
                dbc.ModalBody(id="vehicle-modal-body", className="vehicle-modal-body"),
            ],
            id="vehicle-modal",
            is_open=False,
            size="xl",
            backdrop=True,
            scrollable=True,
            className="vehicle-modal",
        ),

        # FOOTER
        html.Div(
            className="small-muted",
            style={"textAlign": "center", "padding": "24px 0 32px 0"},
            children="© 2025 Premium Car Analytics | Powered by Dash & Plotly | Real-time Market Intelligence",
        ),
    ],
)


# ========================= CALLBACKS =========================
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab"),
)
def render_tab(active_tab):
    if active_tab == "tab-home":
        return html.Div(
            [
                # Welcome Section - Clean professional styling
                html.Div(
                    className="graph-card",
                    style={
                        "padding": "56px 40px", 
                        "marginBottom": "48px",
                        "background": "#F8FAFC",
                        "borderTop": "4px solid #0A4174",
                    },
                    children=[
                        html.H1(
                            "WELCOME TO PREMIUM CAR ANALYTICS",
                            style={
                                "fontSize": "32px",
                                "fontWeight": 800,
                                "marginBottom": "16px",
                                "textTransform": "uppercase",
                                "letterSpacing": "2px",
                                "color": "#0F172A",
                            },
                        ),
                        html.P(
                            "Advanced Vehicle Intelligence & Market Insights Platform",
                            style={
                                "fontSize": "17px", 
                                "color": "#49769F", 
                                "marginBottom": "0", 
                                "fontWeight": 500,
                            },
                        ),
                    ],
                ),
                
                # Features Section - Enhanced cards with icons
                dbc.Row(
                    className="g-4 mb-4",
                            children=[
                        dbc.Col(
                                html.Div(
                                className="graph-card feature-card-modern feature-card-animated",
                                    style={
                                    "padding": "48px 40px 32px 40px", 
                                    "height": "100%",
                                    "borderTop": "4px solid #3B82F6",
                                    "background": "linear-gradient(145deg, #FFFFFF 0%, #FAFCFF 100%)",
                                    "boxShadow": "0 4px 16px rgba(0, 29, 57, 0.08)",
                                        "borderRadius": "16px",
                                    "marginTop": "40px",
                                    },
                                    children=[
                                    # Icon circle with gradient
                                html.Div(
                                        className="feature-icon-circle",
                                    style={
                                            "background": "linear-gradient(135deg, #3B82F6 0%, #1E40AF 100%)",
                                    },
                                    children=[
                                            html.Img(
                                                src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='40' height='40' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M3 3v18h18'/%3E%3Cpath d='M18 7v10'/%3E%3Cpath d='M13 7v10'/%3E%3Cpath d='M8 7v10'/%3E%3C/svg%3E",
                                                style={"width": "40px", "height": "40px"},
                                            ),
                                    ],
                                ),
                                html.Div(
                                        "Manufacturers Comparison",
                                    style={
                                            "fontSize": "22px", 
                                            "fontWeight": 700, 
                                            "color": "#001D39",
                                            "letterSpacing": "-0.3px",
                                            "marginTop": "48px",
                                            "marginBottom": "16px",
                                            "textAlign": "center",
                                        },
                                    ),
                                    html.P(
                                        "Compare up to 5 different car models side by side. Analyze price depreciation trends, "
                                        "mileage impact, and value retention over time. Get detailed insights into which models "
                                        "maintain their value best.",
                                        style={"fontSize": "16px", "lineHeight": "1.6", "color": "#64748B", "margin": 0, "textAlign": "center"},
                                    ),
                                ],
                            ),
                            md=4,
                        ),
                        dbc.Col(
                            html.Div(
                                className="graph-card feature-card-modern feature-card-animated",
                                style={
                                    "padding": "48px 40px 32px 40px", 
                                    "height": "100%",
                                    "borderTop": "4px solid #49769F",
                                    "background": "linear-gradient(145deg, #FFFFFF 0%, #FAFCFF 100%)",
                                    "boxShadow": "0 4px 16px rgba(0, 29, 57, 0.08)",
                                    "borderRadius": "16px",
                                    "marginTop": "40px",
                                },
                                children=[
                                    # Icon circle with gradient
                                    html.Div(
                                        className="feature-icon-circle",
                                        style={
                                            "background": "linear-gradient(135deg, #49769F 0%, #1E40AF 100%)",
                                        },
                                        children=[
                                            html.Img(
                                                src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='40' height='40' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='3' width='7' height='7'/%3E%3Crect x='14' y='3' width='7' height='7'/%3E%3Crect x='14' y='14' width='7' height='7'/%3E%3Crect x='3' y='14' width='7' height='7'/%3E%3C/svg%3E",
                                                style={"width": "40px", "height": "40px"},
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        "Group Comparison",
                                        style={
                                            "fontSize": "22px", 
                                            "fontWeight": 700, 
                                            "color": "#001D39",
                                            "letterSpacing": "-0.3px",
                                            "marginTop": "48px",
                                            "marginBottom": "16px",
                                            "textAlign": "center",
                                        },
                                    ),
                                    html.P(
                                        "Compare two different vehicle groups based on model, year range, and transmission type. "
                                        "Analyze price stability, value for money, and average metrics to make informed decisions "
                                        "between different vehicle segments.",
                                        style={"fontSize": "16px", "lineHeight": "1.6", "color": "#64748B", "margin": 0, "textAlign": "center"},
                                    ),
                                ],
                            ),
                            md=4,
                        ),
                        dbc.Col(
                            html.Div(
                                className="graph-card feature-card-modern feature-card-animated",
                                style={
                                    "padding": "48px 40px 32px 40px", 
                                    "height": "100%",
                                    "borderTop": "4px solid #4E8EA2",
                                    "background": "linear-gradient(145deg, #FFFFFF 0%, #FAFCFF 100%)",
                                    "boxShadow": "0 4px 16px rgba(0, 29, 57, 0.08)",
                                    "borderRadius": "16px",
                                    "marginTop": "40px",
                                },
                                children=[
                                    # Icon circle with gradient
                                    html.Div(
                                        className="feature-icon-circle",
                                        style={
                                            "background": "linear-gradient(135deg, #4E8EA2 0%, #1E40AF 100%)",
                                        },
                                        children=[
                                            html.Img(
                                                src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='40' height='40' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 2L2 7l10 5 10-5-10-5z'/%3E%3Cpath d='M2 17l10 5 10-5'/%3E%3Cpath d='M2 12l10 5 10-5'/%3E%3C/svg%3E",
                                                style={"width": "40px", "height": "40px"},
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        "Buyer's Guide",
                                        style={
                                            "fontSize": "22px", 
                                            "fontWeight": 700, 
                                            "color": "#001D39",
                                            "letterSpacing": "-0.3px",
                                            "marginTop": "48px",
                                            "marginBottom": "16px",
                                            "textAlign": "center",
                                        },
                                    ),
                                    html.P(
                                        "Smart Buyer Matrix helps you find the best value deals. Use advanced filters to narrow down "
                                        "your search, and discover vehicles priced significantly below their model average. Get real-time "
                                        "recommendations for the best deals available.",
                                        style={"fontSize": "16px", "lineHeight": "1.6", "color": "#64748B", "margin": 0, "textAlign": "center"},
                                    ),
                                ],
                            ),
                            md=4,
                        ),
                    ],
                ),
                
                # Team Section - Redesigned with profile images
                html.Div(
                    className="graph-card",
                    style={
                        "padding": "48px 36px", 
                        "marginTop": "72px",
                        "background": "linear-gradient(145deg, #FFFFFF 0%, #F8FBFC 100%)",
                        "borderRadius": "16px",
                    },
                    children=[
                        html.H2(
                            "DEVELOPED BY",
                            style={
                                "fontSize": "17px",
                                "fontWeight": 700,
                                "marginBottom": "40px",
                                "color": "#64748B",
                                "textTransform": "uppercase",
                                "letterSpacing": "2px",
                                "textAlign": "center",
                            },
                        ),
                        html.Div(
                            style={
                                "display": "flex",
                                "justifyContent": "center",
                                "alignItems": "flex-start",
                                "gap": "40px",
                                "flexWrap": "wrap",
                            },
                            children=[
                                # Gaya Gur
                                html.Div(
                                    style={
                                        "display": "flex",
                                        "flexDirection": "column",
                                        "alignItems": "center",
                                        "gap": "16px",
                                    },
                                    children=[
                                        html.Img(
                                            src=app.get_asset_url("gaya.png"),
                                            style={
                                                "width": "110px",
                                                "height": "110px",
                                                "borderRadius": "50%",
                                                "border": "3px solid #4A90D9",
                                                "boxShadow": "0 4px 16px rgba(74, 144, 217, 0.2)",
                                                "objectFit": "cover",
                                                "transition": "all 0.3s ease",
                                            },
                                            className="team-member-image",
                                        ),
                                        html.Div(
                                            "Gaya Gur",
                                            style={
                                                "fontSize": "15px",
                                                "fontWeight": 600,
                                                "color": "#001D39",
                                                "textAlign": "center",
                                            },
                                        ),
                                    ],
                                ),
                                # Moran Shavit
                                html.Div(
                                    style={
                                        "display": "flex",
                                        "flexDirection": "column",
                                        "alignItems": "center",
                                        "gap": "16px",
                                    },
                                    children=[
                                        html.Img(
                                            src=app.get_asset_url("moran.png"),
                                            style={
                                                "width": "110px",
                                                "height": "110px",
                                                "borderRadius": "50%",
                                                "border": "3px solid #4A90D9",
                                                "boxShadow": "0 4px 16px rgba(74, 144, 217, 0.2)",
                                                "objectFit": "cover",
                                                "transition": "all 0.3s ease",
                                            },
                                            className="team-member-image",
                                        ),
                                        html.Div(
                                            "Moran Shavit",
                                            style={
                                                "fontSize": "15px",
                                                "fontWeight": 600,
                                                "color": "#001D39",
                                                "textAlign": "center",
                                            },
                                        ),
                                    ],
                                ),
                                # Matias Guernik
                                html.Div(
                                    style={
                                        "display": "flex",
                                        "flexDirection": "column",
                                        "alignItems": "center",
                                        "gap": "16px",
                                    },
                                    children=[
                                        html.Img(
                                            src=app.get_asset_url("matias.png"),
                                            style={
                                                "width": "110px",
                                                "height": "110px",
                                                "borderRadius": "50%",
                                                "border": "3px solid #4A90D9",
                                                "boxShadow": "0 4px 16px rgba(74, 144, 217, 0.2)",
                                                "objectFit": "cover",
                                                "transition": "all 0.3s ease",
                                            },
                                            className="team-member-image",
                                        ),
                                        html.Div(
                                            "Matias Guernik",
                                            style={
                                                "fontSize": "15px",
                                                "fontWeight": 600,
                                                "color": "#001D39",
                                                "textAlign": "center",
                                            },
                                        ),
                                    ],
                                ),
                                # Tamar Hagbi
                                html.Div(
                                    style={
                                        "display": "flex",
                                        "flexDirection": "column",
                                        "alignItems": "center",
                                        "gap": "16px",
                                    },
                                    children=[
                                        html.Img(
                                            src=app.get_asset_url("tamar.png"),
                                            style={
                                                "width": "110px",
                                                "height": "110px",
                                                "borderRadius": "50%",
                                                "border": "3px solid #4A90D9",
                                                "boxShadow": "0 4px 16px rgba(74, 144, 217, 0.2)",
                                                "objectFit": "cover",
                                                "transition": "all 0.3s ease",
                                            },
                                            className="team-member-image",
                                        ),
                                        html.Div(
                                            "Tamar Hagbi",
                                            style={
                                                "fontSize": "15px",
                                                "fontWeight": 600,
                                                "color": "#001D39",
                                                "textAlign": "center",
                                            },
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ]
        )
    
    if active_tab == "tab-market":
        return html.Div(
            [
                dbc.Row(
                    className="g-4",
                    children=[
                        dbc.Col(
                            html.Div(
                                className="filter-card",
                                children=[
                                    html.Div(
                                        "🎯 Filters",
                                        style={"fontWeight": 900, "marginBottom": "16px", "fontSize": "18px"},
                                    ),
                                    html.Label("Feature Category"),
                                    dcc.Dropdown(
                                        id="market-feature",
                                        options=[{"label": lbl, "value": val} for val, lbl in FEATURES],
                                        value="fuel_type",
                                        clearable=False,
                                    ),
                                    html.Div(style={"height": "16px"}),
                                    html.Label("Country of Origin"),
                                    dcc.Dropdown(
                                        id="market-country",
                                        options=[{"label": c, "value": c} for c in COUNTRIES],
                                        value=COUNTRIES[0] if COUNTRIES else None,
                                        clearable=False,
                                    ),
                                    html.Hr(style={"opacity": 0.2, "margin": "20px 0"}),
                                    html.Div(id="market-stats", className="small-muted"),
                                ],
                            ),
                            md=3,
                        ),
                        dbc.Col(
                            html.Div(
                                className="graph-card",
                                # The donut chart is intentionally not displayed (kept as commented code).
                                children=[
                                    # dcc.Graph(id="market-donut", config={"displayModeBar": False})
                                    html.Div(
                                        "Donut chart is currently hidden (commented out).",
                                        className="small-muted",
                                        style={"padding": "16px"},
                                    )
                                ],
                            ),
                            md=9,
                        ),
                    ],
                ),
            ]
        )

    if active_tab == "tab-model":
        # 1. Get Top 15 Manufacturers
        top_15 = df["manufacturer"].value_counts().head(15).index.tolist()

        # 2. Get the rest and sort alphabetically
        all_manus = df["manufacturer"].dropna().unique().tolist()
        rest_sorted = sorted([m for m in all_manus if m not in top_15])

        # 3. Combine
        final_manufacturer_order = top_15 + rest_sorted

        return dbc.Row(
            className="g-4",
            children=[
                dbc.Col(
                    html.Div(
                        className="filter-card",
                        children=[
                            html.Div(
                                "🔄 Model Comparison",
                                style={
                                    "fontWeight": 900,
                                    "marginBottom": "12px",
                                    "fontSize": "18px",
                                    "textAlign": "center",
                                    "width": "100%",
                                    "display": "block"
                                },
                            ),
                            html.Div(
                                "Select up to 5 manufacturers for optimal visualization",
                                className="small-muted",
                                style={"marginBottom": "16px", "textAlign": "center", "width": "100%"},
                            ),
                            dcc.Dropdown(
                                id="model-selected",
                                options=[{"label": m, "value": m} for m in final_manufacturer_order],
                                value=final_manufacturer_order[:3] if len(df) > 0 else [],
                                multi=True,
                            ),
                            html.Hr(style={"opacity": 0.2, "margin": "20px 0"}),
                            html.Div(id="model-mini-kpis"),
                        ],
                    ),
                    md=3,
                ),
                dbc.Col(
                    [
                        html.Div(
                            className="graph-card",
                            children=[dcc.Graph(id="model-line", config={"displayModeBar": False})],
                        ),
                        html.Div(id="depreciation-trends", style={"marginTop": "16px"}),
                    ],
                    md=9,
                ),
            ],
        )

    if active_tab == "tab-buyer":
        return html.Div(
            [
                # Methodology (collapsible)
                html.Div(
                    className="graph-card",
                    style={"padding": "24px", "marginBottom": "16px"},
                    children=[
                        html.Div(
                            style={"display": "flex", "alignItems": "center", "gap": "12px", "marginBottom": "8px"},
                            children=[
                                html.Div(
                                    "Buyer's Guide Methodology",
                                    style={
                                        "fontSize": "20px",
                                        "fontWeight": 700,
                                        "color": "#374151",
                                    },
                                ),
                                html.Div(
                                    id="buyer-info-button",
                                    children="ℹ️",
                                    style={
                                        "width": "40px",
                                        "height": "40px",
                                        "borderRadius": "50%",
                                        "background": "rgba(100, 116, 139, 0.1)",
                                        "border": "1px solid rgba(100, 116, 139, 0.25)",
                                        "display": "flex",
                                        "alignItems": "center",
                                        "justifyContent": "center",
                                        "cursor": "pointer",
                                        "fontSize": "16px",
                                        "transition": "all 0.25s ease",
                                    },
                                    title="Click to see calculation methodology",
                                ),
                            ],
                        ),
                        html.Div(
                            "How the Smart Buyer Matrix and Best Deals are computed",
                            style={"fontSize": "13px", "color": "#9CA3AF", "marginBottom": "12px"},
                        ),
                        dbc.Collapse(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H6(
                                            "Smart Buyer Matrix (Value for Money)",
                                            style={"fontWeight": 700, "marginBottom": "10px", "color": "#7C9CB5"},
                                        ),
                                        html.Ol(
                                            [
                                                html.Li(
                                                    [
                                                        html.Strong("Filtering: "),
                                                        "Apply the selected filters (price range, max mileage, country, transmission, fuel type, year range).",
                                                    ],
                                                    style={"marginBottom": "8px", "fontSize": "13px"},
                                                ),
                                                html.Li(
                                                    [
                                                        html.Strong("Model selection: "),
                                                        "If you select models explicitly, only those models are used. Otherwise the top-N models by frequency are selected (N = slider value).",
                                                    ],
                                                    style={"marginBottom": "8px", "fontSize": "13px"},
                                                ),
                                                html.Li(
                                                    [
                                                        html.Strong("Aggregation per model: "),
                                                        "Compute average price (AvgPrice), price std (StdPrice), average mileage in km (AvgMileageKm), mileage std (StdMileageKm), and listing count (Count) per model.",
                                                    ],
                                                    style={"marginBottom": "8px", "fontSize": "13px"},
                                                ),
                                                html.Li(
                                                    [
                                                        html.Strong("Filtering: "),
                                                        "Exclude models with Count < 2 (low confidence) or AvgMileageKm <= 0 (invalid data).",
                                                    ],
                                                    style={"marginBottom": "8px", "fontSize": "13px"},
                                                ),
                                                html.Li(
                                                    [
                                                        html.Strong("Price per km: "),
                                                        "Compute average price per km per model:",
                                                        html.Code(
                                                            "AvgPPK = AvgPrice / AvgMileageKm",
                                                            style={
                                                                "background": "rgba(168, 150, 168, 0.15)",
                                                                "padding": "2px 6px",
                                                                "borderRadius": "4px",
                                                                "fontSize": "11px",
                                                                "marginLeft": "6px",
                                                                "color": "#374151",
                                                            },
                                                        ),
                                                        " (units: currency/km). Lower AvgPPK means better value.",
                                                    ],
                                                    style={"marginBottom": "8px", "fontSize": "13px"},
                                                ),
                                                html.Li(
                                                    [
                                                        html.Strong("Normalization: "),
                                                        "Normalize AvgPPK to 0–100 scale and invert so higher means better value:",
                                                        html.Code(
                                                            "ValueNorm = 100 * (1 - (AvgPPK - min)/(max - min))",
                                                            style={
                                                                "background": "rgba(168, 150, 168, 0.15)",
                                                                "padding": "2px 6px",
                                                                "borderRadius": "4px",
                                                                "fontSize": "11px",
                                                                "marginLeft": "6px",
                                                                "color": "#374151",
                                                            },
                                                        ),
                                                        " If max==min, set ValueNorm = 50 for all. Clip to [0, 100].",
                                                    ],
                                                    style={"marginBottom": "8px", "fontSize": "13px"},
                                                ),
                                                html.Li(
                                                    [
                                                        html.Strong("Bubble size: "),
                                                        "Scale by listing count to visualize availability. Error bars show within-model variability (std) for mileage and price.",
                                                    ],
                                                    style={"marginBottom": "8px", "fontSize": "13px"},
                                                ),
                                            ],
                                            style={"paddingLeft": "20px", "marginBottom": "14px"},
                                        ),
                                        html.H6(
                                            "Best Deals (Model-relative value)",
                                            style={"fontWeight": 700, "marginBottom": "10px", "color": "#6B9080"},
                                        ),
                                        html.Ol(
                                            [
                                                html.Li(
                                                    [
                                                        html.Strong("Per-model normalization: ", style={"color": "#374151"}),
                                                        html.Span("For each model with at least 5 listings, compute mean and standard deviation for both price and mileage.", style={"color": "#6B7280"}),
                                                    ],
                                                    style={"marginBottom": "8px", "fontSize": "13px"},
                                                ),
                                                html.Li(
                                                    [
                                                        html.Strong("Price per km per listing: ", style={"color": "#374151"}),
                                                        html.Span("Compute PPK for each listing: ", style={"color": "#6B7280"}),
                                                        html.Code(
                                                            "PPK_listing = price / max(mileage_km, 1)",
                                                            style={
                                                                "background": "rgba(107, 144, 128, 0.12)",
                                                                "padding": "2px 6px",
                                                                "borderRadius": "4px",
                                                                "fontSize": "11px",
                                                                "marginLeft": "6px",
                                                                "color": "#374151",
                                                            },
                                                        ),
                                                        " (units: currency/km). Use max(...,1) only to avoid division by zero.",
                                                    ],
                                                    style={"marginBottom": "8px", "fontSize": "13px"},
                                                ),
                                                html.Li(
                                                    [
                                                        html.Strong("Model statistics: ", style={"color": "#374151"}),
                                                        html.Span("For each model, compute mean and std of PPK across all listings. Need at least 2 listings per model.", style={"color": "#6B7280"}),
                                                    ],
                                                    style={"marginBottom": "8px", "fontSize": "13px"},
                                                ),
                                                html.Li(
                                                    [
                                                        html.Strong("PPK z-score: ", style={"color": "#374151"}),
                                                        html.Span("Normalize PPK relative to model: ", style={"color": "#6B7280"}),
                                                        html.Code(
                                                            "Z_PPK = (PPK_listing - model_mean_PPK) / model_std_PPK",
                                                            style={
                                                                "background": "rgba(107, 144, 128, 0.12)",
                                                                "padding": "2px 6px",
                                                                "borderRadius": "4px",
                                                                "fontSize": "11px",
                                                                "marginLeft": "6px",
                                                                "color": "#374151",
                                                            },
                                                        ),
                                                        " Negative z-score = PPK below model mean = better deal.",
                                                    ],
                                                    style={"marginBottom": "8px", "fontSize": "13px"},
                                                ),
                                                html.Li(
                                                    [
                                                        html.Strong("Deal threshold: ", style={"color": "#374151"}),
                                                        html.Span("Keep listings where Z_PPK < -0.5 (PPK at least 0.5 std below model mean).", style={"color": "#6B7280"}),
                                                    ],
                                                    style={"marginBottom": "8px", "fontSize": "13px"},
                                                ),
                                                html.Li(
                                                    [
                                                        html.Strong("Ranking: "),
                                                        "Sort by most negative Z_PPK (best relative deal) and display the top results.",
                                                    ],
                                                    style={"marginBottom": "8px", "fontSize": "13px"},
                                                ),
                                            ],
                                            style={"paddingLeft": "20px"},
                                        ),
                                        html.Div(
                                            [
                                                html.Strong("Note: ", style={"color": "#B5A67D"}),
                                                "These are statistical signals based on listing prices. They do not account for trims, accidents, ownership history, or condition unless those fields are included and modeled.",
                                            ],
                                            style={
                                                "background": "rgba(181, 166, 125, 0.08)",
                                                "padding": "12px",
                                                "borderRadius": "8px",
                                                "border": "1px solid rgba(181, 166, 125, 0.2)",
                                                "fontSize": "13px",
                                                "marginTop": "14px",
                                                "color": "#6B7280",
                                            },
                                        ),
                                    ]
                                ),
                                style={
                                    "background": "rgba(255, 255, 253, 0.95)",
                                    "border": "1px solid rgba(148, 163, 184, 0.2)",
                                },
                            ),
                            id="buyer-methodology-collapse",
                            is_open=False,
                        ),
                    ],
                ),

                # Filters Row
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(
                                    className="filter-card",
                                    children=[
                                        html.Div(
                                            "🎯 Smart Filters",
                                            style={"fontWeight": 900, "marginBottom": "16px", "fontSize": "18px"},
                                        ),
                                        html.Label("Search by Vehicle Model"),
                                        dcc.Dropdown(
                                            id="buyer-vehicles",
                                            options=[{"label": m, "value": m} for m in sorted(df["vehicle"].unique())],
                                            value=[],
                                            multi=True,
                                            placeholder="Leave empty to see top models",
                                        ),
                                        html.Div(style={"height": "12px"}),
                                        html.Label("Country of Origin"),
                                        dcc.Dropdown(
                                            id="buyer-country",
                                            options=[{"label": c, "value": c} for c in COUNTRIES],
                                            value=[],
                                            multi=True,
                                            placeholder="Select countries",
                                        ),
                                        html.Div(style={"height": "12px"}),
                                        html.Label("Transmission Type"),
                                        dcc.Dropdown(
                                            id="buyer-transmission",
                                            options=[{"label": t, "value": t} for t in
                                                     sorted(df["transmission"].unique())],
                                            value=[],
                                            multi=True,
                                            placeholder="Select transmission types",
                                        ),
                                        html.Div(style={"height": "12px"}),
                                        html.Label("Fuel Type"),
                                        dcc.Dropdown(
                                            id="buyer-fuel",
                                            options=[{"label": f, "value": f} for f in
                                                     sorted(df["fuel_type"].unique())],
                                            value=[],
                                            multi=True,
                                            placeholder="Select fuel types",
                                        ),
                                        html.Div(style={"height": "12px"}),
                                        html.Label("מספר ידיים (Owner Count)"),
                                        dcc.Dropdown(
                                            id="buyer-owner-count",
                                            options=[{"label": str(int(o)), "value": int(o)} for o in
                                                     sorted(df["owner_count"].dropna().unique()) if pd.notna(o)],
                                            value=[],
                                            multi=True,
                                            placeholder="Select owner count",
                                        ),
                                    ],
                                )
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                html.Div(
                                    className="filter-card",
                                    children=[
                                        html.Div(
                                            "💰 Price & Mileage",
                                            style={"fontWeight": 900, "marginBottom": "16px", "fontSize": "18px"},
                                        ),
                                        html.Label("Price Range (₪)"),
                                        dcc.RangeSlider(
                                            id="buyer-price-range",
                                            min=0,
                                            max=int(df["price"].quantile(0.95)),
                                            step=5000,
                                            value=[0, int(df["price"].quantile(0.95))],
                                            marks={
                                                0: "₪0",
                                                int(df["price"].quantile(
                                                    0.25)): f"₪{int(df['price'].quantile(0.25) / 1000)}K",
                                                int(df["price"].quantile(
                                                    0.5)): f"₪{int(df['price'].quantile(0.5) / 1000)}K",
                                                int(df["price"].quantile(
                                                    0.75)): f"₪{int(df['price'].quantile(0.75) / 1000)}K",
                                                int(df["price"].quantile(
                                                    0.95)): f"₪{int(df['price'].quantile(0.95) / 1000)}K",
                                            },
                                            tooltip={"placement": "bottom", "always_visible": False},
                                        ),
                                        html.Div(style={"height": "20px"}),
                                        html.Label("Max Mileage (km)"),
                                        dcc.Slider(
                                            id="buyer-max-mileage",
                                            min=0,
                                            max=int(df["mileage"].quantile(0.95)),
                                            step=10000,
                                            value=int(df["mileage"].quantile(0.95)),
                                            marks={
                                                0: "0",
                                                int(df["mileage"].quantile(
                                                    0.5)): f"{int(df['mileage'].quantile(0.5) / 1000)}K",
                                                int(df["mileage"].quantile(
                                                    0.95)): f"{int(df['mileage'].quantile(0.95) / 1000)}K",
                                            },
                                            tooltip={"placement": "bottom", "always_visible": False},
                                        ),
                                        html.Div(style={"height": "20px"}),
                                        html.Label("Year Range"),
                                        dcc.RangeSlider(
                                            id="buyer-year-range",
                                            min=YEAR_MIN,
                                            max=YEAR_MAX,
                                            step=1,
                                            value=[YEAR_MIN, YEAR_MAX],
                                            marks={
                                                YEAR_MIN: str(YEAR_MIN),
                                                int((YEAR_MIN + YEAR_MAX) / 2): str(int((YEAR_MIN + YEAR_MAX) / 2)),
                                                YEAR_MAX: str(YEAR_MAX),
                                            },
                                            tooltip={"placement": "bottom", "always_visible": False},
                                        ),
                                    ],
                                )
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                html.Div(
                                    className="filter-card",
                                    children=[
                                        html.Div(
                                            "📊 Display Options",
                                            style={"fontWeight": 900, "marginBottom": "16px", "fontSize": "18px"},
                                        ),
                                        html.Label("Number of Models to Show"),
                                        dcc.Slider(
                                            id="buyer-max-vehicles",
                                            min=5,
                                            max=20,
                                            step=1,
                                            value=12,
                                            marks={5: "5", 10: "10", 15: "15", 20: "20"},
                                            tooltip={"placement": "bottom", "always_visible": False},
                                        ),
                                        html.Div(style={"height": "20px"}),
                                        html.Div(
                                            [
                                                html.H6("Quick Tips",
                                                        style={"fontWeight": 600, "marginBottom": "10px", "color": "#374151", "fontSize": "13px"}),
                                                html.Ul(
                                                    [
                                                        html.Li("Sage = Best value",
                                                                style={"marginBottom": "4px", "fontSize": "12px", "color": "#6B7280"}),
                                                        html.Li("Larger = More available",
                                                                style={"marginBottom": "4px", "fontSize": "12px", "color": "#6B7280"}),
                                                        html.Li("Hover for details",
                                                                style={"marginBottom": "4px", "fontSize": "12px", "color": "#6B7280"}),
                                                        html.Li(
                                                            "Target: Large sage bubbles",
                                                            style={"marginBottom": "4px", "fontWeight": 600,
                                                                   "color": "#6B9080", "fontSize": "12px"},
                                                        ),
                                                    ],
                                                    style={"paddingLeft": "18px"},
                                                ),
                                            ]
                                        ),
                                        html.Div(style={"height": "12px"}),
                                        dbc.Button(
                                            "🔄 Reset All Filters",
                                            id="buyer-reset-btn",
                                            color="primary",
                                            size="sm",
                                            style={"width": "100%"},
                                        ),
                                    ],
                                )
                            ],
                            md=3,
                        ),
                    ],
                    className="g-4 mb-4",
                ),

                # Matrix Graph
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(
                                    className="graph-card",
                                    children=[
                                        dcc.Loading(
                                            id="loading-matrix",
                                            type="circle",
                                            children=[dcc.Graph(id="buyer-matrix", config={"displayModeBar": False})],
                                        ),
                                        # Color Legend for Smart Buyer Matrix
                                        html.Div(
                                            style={
                                                "display": "flex",
                                                "justifyContent": "center",
                                                "alignItems": "center",
                                                "gap": "28px",
                                                "padding": "16px 24px",
                                                "marginTop": "24px",
                                                "background": "#f1f5f9",
                                                "borderRadius": "8px",
                                                "border": "1px solid #e2e8f0",
                                                "flexWrap": "wrap",
                                            },
                                            children=[
                                                html.Span(
                                                    "Value Score:",
                                                    style={
                                                        "fontSize": "15px",
                                                        "fontWeight": "700",
                                                        "color": "#1F2937",
                                                        "marginRight": "8px",
                                                    }
                                                ),
                                                # Excellent Value - Green
                                                html.Div(
                                                    style={"display": "flex", "alignItems": "center", "gap": "8px"},
                                                    children=[
                                                        html.Div(style={
                                                            "width": "14px",
                                                            "height": "14px",
                                                            "borderRadius": "50%",
                                                            "background": "#16A34A",
                                                        }),
                                                        html.Span("Excellent (≥75)", style={"fontSize": "13px", "color": "#374151", "fontWeight": "500"}),
                                                    ]
                                                ),
                                                # Good Value - Lime
                                                html.Div(
                                                    style={"display": "flex", "alignItems": "center", "gap": "8px"},
                                                    children=[
                                                        html.Div(style={
                                                            "width": "14px",
                                                            "height": "14px",
                                                            "borderRadius": "50%",
                                                            "background": "#65A30D",
                                                        }),
                                                        html.Span("Good (55-74)", style={"fontSize": "13px", "color": "#374151", "fontWeight": "500"}),
                                                    ]
                                                ),
                                                # Fair Value - Yellow
                                                html.Div(
                                                    style={"display": "flex", "alignItems": "center", "gap": "8px"},
                                                    children=[
                                                        html.Div(style={
                                                            "width": "14px",
                                                            "height": "14px",
                                                            "borderRadius": "50%",
                                                            "background": "#EAB308",
                                                        }),
                                                        html.Span("Fair (35-54)", style={"fontSize": "13px", "color": "#374151", "fontWeight": "500"}),
                                                    ]
                                                ),
                                                # Poor Value - Red
                                                html.Div(
                                                    style={"display": "flex", "alignItems": "center", "gap": "8px"},
                                                    children=[
                                                        html.Div(style={
                                                            "width": "14px",
                                                            "height": "14px",
                                                            "borderRadius": "50%",
                                                            "background": "#DC2626",
                                                        }),
                                                        html.Span("Poor (<35)", style={"fontSize": "13px", "color": "#374151", "fontWeight": "500"}),
                                                    ]
                                                ),
                                            ],
                                        ),
                                    ],
                                )
                            ],
                            md=12,
                        )
                    ],
                    className="g-4 mb-4",
                ),

                # Best Deals
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(
                                    className="graph-card",
                                    children=[
                                        dcc.Loading(
                                            id="loading-deals",
                                            type="circle",
                                            children=[html.Div(id="best-deals")],
                                        )
                                    ],
                                )
                            ],
                            md=12,
                        )
                    ],
                    className="g-4",
                ),
            ]
        )

    # Default: Group Comparison tab

    # 1. Calculate Model Order: Top 20 Popular -> Then Alphabetical
    top_models = df["vehicle"].value_counts().head(20).index.tolist()
    all_models = df["vehicle"].dropna().unique().tolist()
    rest_models = sorted([m for m in all_models if m not in top_models])
    final_model_order = top_models + rest_models

    return dbc.Row(
        className="g-4",
        children=[
            dbc.Col(
                html.Div(
                    className="filter-card",
                    children=[
                        html.Div(
                            "Group A Configuration",
                            style={
                                "fontWeight": 900,
                                "marginBottom": "16px",
                                "fontSize": "16px",
                                "color": COLORS["blue"],
                                "textAlign": "center",
                                "width": "100%",
                                "display": "block"
                            },
                        ),
                        html.Label("Model"),
                        dcc.Dropdown(
                            id="ga-model",
                            options=[{"label": t, "value": t} for t in final_model_order],
                            value=final_model_order[0] if len(df) else None,
                            clearable=False,
                        ),
                        html.Div(style={"height": "12px"}),
                        html.Label("Year Range"),
                        dcc.RangeSlider(
                            id="ga-year",
                            min=YEAR_MIN,
                            max=YEAR_MAX,
                            step=1,
                            value=[max(YEAR_MIN, 2019), min(YEAR_MAX, 2022)],
                            marks=None,
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        html.Div(style={"height": "12px"}),
                        html.Label("Transmission"),
                        dcc.Dropdown(
                            id="ga-trans",
                            options=[
                                {"label": t, "value": t} 
                                for t in sorted(df["transmission"].unique()) 
                                if any(keyword in str(t) for keyword in
                                       ["אוטומטי", "ידני", "automatic", "manual", "אוטומט"])
                            ],
                            value=next((t for t in sorted(df["transmission"].unique()) if
                                        any(keyword in str(t) for keyword in ["אוטומטי", "automatic", "אוטומט"])),
                                       sorted(df["transmission"].unique())[0] if len(df) > 0 else None),
                            clearable=False,
                        ),
                    ],
                ),
                md=6,
            ),
            dbc.Col(
                html.Div(
                    className="filter-card",
                    children=[
                        html.Div(
                            "Group B Configuration",
                            style={
                                "fontWeight": 900,
                                "marginBottom": "16px",
                                "fontSize": "16px",
                                "color": COLORS["purple"],
                                "textAlign": "center",
                                "width": "100%",
                                "display": "block"
                            },
                        ),
                        html.Label("Model"),
                        dcc.Dropdown(
                            id="gb-model",
                            options=[{"label": t, "value": t} for t in final_model_order],
                            value=final_model_order[0] if len(df) else None,
                            clearable=False,
                        ),
                        html.Div(style={"height": "12px"}),
                        html.Label("Year Range"),
                        dcc.RangeSlider(
                            id="gb-year",
                            min=YEAR_MIN,
                            max=YEAR_MAX,
                            step=1,
                            value=[max(YEAR_MIN, 2021), min(YEAR_MAX, 2024)],
                            marks=None,
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        html.Div(style={"height": "12px"}),
                        html.Label("Transmission"),
                        dcc.Dropdown(
                            id="gb-trans",
                            options=[
                                {"label": t, "value": t} 
                                for t in sorted(df["transmission"].unique()) 
                                if any(keyword in str(t) for keyword in
                                       ["אוטומטי", "ידני", "automatic", "manual", "אוטומט"])
                            ],
                            value=next((t for t in sorted(df["transmission"].unique()) if
                                        any(keyword in str(t) for keyword in ["אוטומטי", "automatic", "אוטומט"])),
                                       sorted(df["transmission"].unique())[0] if len(df) > 0 else None),
                            clearable=False,
                        ),
                    ],
                ),
                md=6,
            ),
            dbc.Col(
                html.Div(
                    className="graph-card",
                    children=[dcc.Graph(id="group-butterfly", config={"displayModeBar": False})],
                ),
                md=12,
            ),
            dbc.Col(html.Div(id="group-details"), md=12),
        ],
    )


@app.callback(
    Output("market-donut", "figure"),
    Output("market-stats", "children"),
    Input("market-feature", "value"),
    Input("market-country", "value"),
)
def update_market(feature, country):
    # Note: The donut graph component is currently commented out in the layout.
    # This callback can remain as-is; Dash will simply not render the output component.

    if not feature or not country:
        fig = go.Figure()
        fig.update_layout(title=dict(text="<b>No selection</b>"), height=550)
        return fig, "No selection"

    dff = df[df["country"] == country].copy()
    if dff.empty or feature not in dff.columns:
        fig = go.Figure()
        fig.update_layout(title=dict(text="<b>No data available</b>"), height=550)
        return fig, "No data"

    counts = dff[feature].value_counts().head(10)
    fig = fig_donut(counts, f"{feature.replace('_', ' ').title()} Distribution", f"Market: {country}")

    stats = html.Div(
        style={"lineHeight": "1.8"},
        children=[
            html.Div([html.Span("📊 Total Listings: ", style={"fontWeight": 600}), html.B(f"{len(dff):,}")]),
            html.Div([html.Span("💰 Avg Price: ", style={"fontWeight": 600}), html.B(f"₪{dff['price'].mean():,.0f}")]),
            html.Div(
                [html.Span("🛣️ Avg Mileage: ", style={"fontWeight": 600}), html.B(f"{dff['mileage'].mean():,.0f} km")]),
        ],
    )

    return fig, stats


@app.callback(
    Output("buyer-vehicles", "value"),
    Output("buyer-country", "value"),
    Output("buyer-transmission", "value"),
    Output("buyer-fuel", "value"),
    Output("buyer-owner-count", "value"),
    Output("buyer-price-range", "value"),
    Output("buyer-max-mileage", "value"),
    Output("buyer-year-range", "value"),
    Output("buyer-max-vehicles", "value"),
    Input("buyer-reset-btn", "n_clicks"),
    prevent_initial_call=True,
)
def reset_filters(n_clicks):
    # Reset all buyer-guide filters to default values
    return (
        [],  # vehicles
        [],  # country
        [],  # transmission
        [],  # fuel
        [],  # owner_count
        [0, int(df["price"].quantile(0.95))],  # price range
        int(df["mileage"].quantile(0.95)),  # max mileage
        [YEAR_MIN, YEAR_MAX],  # year range
        12,  # max vehicles
    )


@app.callback(
    Output("buyer-matrix", "figure"),
    Output("best-deals", "children"),
    Output("best-deals-store", "data"),
    Input("buyer-vehicles", "value"),
    Input("buyer-price-range", "value"),
    Input("buyer-max-mileage", "value"),
    Input("buyer-country", "value"),
    Input("buyer-transmission", "value"),
    Input("buyer-fuel", "value"),
    Input("buyer-owner-count", "value"),
    Input("buyer-year-range", "value"),
    Input("buyer-max-vehicles", "value"),
)
def update_buyer_guide(vehicles, price_range, max_mileage, country, transmission, fuel, owner_count, year_range,
                       max_vehicles):
    # Check if any filters are active (user has made selections)
    price_max_default = int(df["price"].quantile(0.95))
    mileage_max_default = int(df["mileage"].quantile(0.95))
    year_min_default = YEAR_MIN
    year_max_default = YEAR_MAX

    has_price_filter = price_range[0] > 0 or price_range[1] < price_max_default
    has_mileage_filter = max_mileage < mileage_max_default
    has_country_filter = country and len(country) > 0
    has_transmission_filter = transmission and len(transmission) > 0
    has_fuel_filter = fuel and len(fuel) > 0
    has_owner_filter = owner_count and len(owner_count) > 0
    has_year_filter = year_range and (year_range[0] > year_min_default or year_range[1] < year_max_default)
    has_vehicle_filter = vehicles and len(vehicles) > 0

    has_any_filter = (has_price_filter or has_mileage_filter or has_country_filter or
                      has_transmission_filter or has_fuel_filter or has_owner_filter or
                      has_year_filter or has_vehicle_filter)

    # Only apply filters if user has made selections
    min_price = price_range[0] if (has_price_filter and price_range[0] > 0) else None
    max_price = price_range[1] if has_price_filter else None
    max_mileage_filtered = max_mileage if has_mileage_filter else None
    country_filtered = country if has_country_filter else None
    transmission_filtered = transmission if has_transmission_filter else None
    fuel_filtered = fuel if has_fuel_filter else None
    owner_count_filtered = owner_count if has_owner_filter else None
    year_range_filtered = year_range if has_year_filter else None

    matrix_fig, displayed_vehicles = fig_smart_buyer_matrix(
        df,
        selected_vehicles=vehicles if has_vehicle_filter else None,
        max_price=max_price,
        min_price=min_price,
        max_mileage=max_mileage_filtered,
        country=country_filtered,
        transmission=transmission_filtered,
        fuel_type=fuel_filtered,
        owner_count=owner_count_filtered,
        year_range=year_range_filtered,
        max_vehicles=max_vehicles,
    )

    # Apply the same filters to the deals chart input
    dff_deals = df.copy()
    if min_price:
        dff_deals = dff_deals[dff_deals["price"] >= min_price]
    if max_price:
        dff_deals = dff_deals[dff_deals["price"] <= max_price]
    if max_mileage_filtered:
        dff_deals = dff_deals[dff_deals["mileage"] <= max_mileage_filtered]
    if country_filtered and len(country_filtered) > 0:
        dff_deals = dff_deals[dff_deals["country"].isin(country_filtered)]
    if transmission_filtered and len(transmission_filtered) > 0:
        dff_deals = dff_deals[dff_deals["transmission"].isin(transmission_filtered)]
    if fuel_filtered and len(fuel_filtered) > 0:
        dff_deals = dff_deals[dff_deals["fuel_type"].isin(fuel_filtered)]
    if owner_count_filtered and len(owner_count_filtered) > 0 and "owner_count" in dff_deals.columns:
        dff_deals = dff_deals[dff_deals["owner_count"].isin(owner_count_filtered)]
    if year_range_filtered:
        dff_deals = dff_deals[
            (dff_deals["on_road_year"] >= year_range_filtered[0]) & (
                        dff_deals["on_road_year"] <= year_range_filtered[1])
        ]

    # Only show best deals from vehicles displayed in the matrix
    deals_cards = create_best_deals_cards(dff_deals, displayed_vehicles=displayed_vehicles)

    # Store best deals data for modal
    dff_deals_filtered = dff_deals.copy()
    if displayed_vehicles and len(displayed_vehicles) > 0:
        dff_deals_filtered = dff_deals_filtered[dff_deals_filtered["vehicle"].isin(displayed_vehicles)]

    # Calculate Price per km (PPK) z-scores for best deals
    # PPK_listing = price / max(mileage_km, 1)
    dff_deals_filtered["ppk_listing"] = dff_deals_filtered["price"] / dff_deals_filtered["mileage"].clip(lower=1.0)
    dff_deals_filtered["ppk_zscore"] = np.nan
    
    for model in dff_deals_filtered["vehicle"].unique():
        model_data = dff_deals_filtered[dff_deals_filtered["vehicle"] == model]
        # Need at least 2 listings to compute std
        if len(model_data) >= 2:
            mean_ppk = model_data["ppk_listing"].mean()
            std_ppk = model_data["ppk_listing"].std()
            if std_ppk > 0:
                dff_deals_filtered.loc[dff_deals_filtered["vehicle"] == model, "ppk_zscore"] = (
                    dff_deals_filtered.loc[dff_deals_filtered["vehicle"] == model, "ppk_listing"] - mean_ppk
                ) / std_ppk

    best_deals_data = dff_deals_filtered[dff_deals_filtered["ppk_zscore"] < -0.5].nsmallest(10, "ppk_zscore")
    best_deals_data = best_deals_data.sort_values("ppk_zscore")

    # Convert to dict for storage
    deals_store_data = best_deals_data.to_dict("records") if len(best_deals_data) > 0 else {}

    return matrix_fig, deals_cards, deals_store_data


@app.callback(
    Output("vehicle-modal", "is_open"),
    Output("vehicle-modal-title", "children"),
    Output("vehicle-modal-body", "children"),
    Input({"type": "deal-card", "index": ALL}, "n_clicks"),
    Input("vehicle-modal-close-btn", "n_clicks"),
    State("best-deals-store", "data"),
    State("vehicle-modal", "is_open"),
    prevent_initial_call=True,
)
def open_vehicle_modal(n_clicks_list, close_clicks, deals_data, is_open):
    ctx = callback_context
    if not ctx.triggered:
        return False, "Vehicle Details", html.Div()

    # Handle close button
    triggered_id = ctx.triggered[0]["prop_id"]
    if "vehicle-modal-close-btn" in triggered_id:
        return False, "Vehicle Details", html.Div()

    if not deals_data:
        return is_open, "Vehicle Details", html.Div()

    # Find which card was clicked by extracting index from triggered_id
    # triggered_id format: '{"type":"deal-card","index":2}.n_clicks'
    if "deal-card" not in triggered_id or not deals_data:
        return is_open, "Vehicle Details", html.Div()

    # Extract the index from the triggered_id JSON string
    import json
    try:
        # Parse the JSON part of the triggered_id (before .n_clicks)
        json_part = triggered_id.split('.')[0]
        card_info = json.loads(json_part)
        card_index = card_info.get("index")
        
        # Validate index
        if card_index is None or card_index < 0 or card_index >= len(deals_data):
            return is_open, "Vehicle Details", html.Div()
        
        # CRITICAL: Only open modal if there was an actual click (n_clicks > 0)
        # This prevents auto-opening on page load or state changes
        if not n_clicks_list or card_index >= len(n_clicks_list):
            return is_open, "Vehicle Details", html.Div()
        
        if not n_clicks_list[card_index] or n_clicks_list[card_index] <= 0:
            return is_open, "Vehicle Details", html.Div()
    except (json.JSONDecodeError, KeyError, ValueError, IndexError):
        # Fallback: if parsing fails, return current state
        return is_open, "Vehicle Details", html.Div()

    vehicle_data = deals_data[card_index]

    # Extract values with formatters
    def get_value(field, formatter=None):
        value = vehicle_data.get(field)
        if pd.notna(value) and value is not None:
            return formatter(value) if formatter else str(value)
        return "N/A"

    # A) Modal Header - Title and subtitle
    vehicle_name = vehicle_data.get('vehicle', 'Unknown Vehicle')
    year = get_value('on_road_year', lambda x: str(int(x)))
    mileage = get_value('mileage', lambda x: f"{x:,.0f} km")
    transmission = get_value('transmission')
    fuel = get_value('fuel_type')

    modal_title = html.Div(
        [
            html.Div(
                [
                    html.Span("🚙", style={"fontSize": "20px", "marginRight": "8px", "color": "#475569"}),
                    html.Span(
                        vehicle_name,
                        style={
                            "fontSize": "24px",
                            "fontWeight": 700,
                            "color": "#1F2937",
                        },
                    ),
                ],
                style={"display": "flex", "alignItems": "center"},
            ),
            html.Div(
                f"{year} • {mileage} • {transmission} • {fuel}",
                className="vehicle-modal-subtitle",
                style={
                    "fontSize": "12px",
                    "color": "#6B7280",
                    "marginTop": "6px",
                },
            ),
        ],
        className="vehicle-modal-header-content",
    )

    # B) Hero KPI Row (4 compact KPIs)
    price = get_value('price', lambda x: f"₪{x:,.0f}")
    owners = get_value('owner_count', lambda x: f"{int(x)} ידיים")

    kpi_row = html.Div(
        [
            html.Div(
                [
                    html.Div("💰", className="kpi-icon"),
                    html.Div(
                        [
                            html.Div("Price", className="kpi-label"),
                            html.Div(price, className="kpi-value"),
                        ],
                    ),
                ],
                className="vehicle-kpi-card vehicle-kpi-card-price",
            ),
            html.Div(
                [
                    html.Div("🛣️", className="kpi-icon"),
                    html.Div(
                        [
                            html.Div("Mileage", className="kpi-label"),
                            html.Div(mileage, className="kpi-value"),
                        ],
                    ),
                ],
                className="vehicle-kpi-card",
            ),
            html.Div(
                [
                    html.Div("📅", className="kpi-icon"),
                    html.Div(
                        [
                            html.Div("Year", className="kpi-label"),
                            html.Div(year, className="kpi-value"),
                        ],
                    ),
                ],
                className="vehicle-kpi-card",
            ),
            html.Div(
                [
                    html.Div("👤", className="kpi-icon kpi-icon-owners"),
                    html.Div(
                        [
                            html.Div("Owners", className="kpi-label"),
                            html.Div(owners, className="kpi-value"),
                        ],
                    ),
                ],
                className="vehicle-kpi-card",
            ),
        ],
        className="vehicle-kpi-row",
    )

    # C) Details Section (two-column layout)
    # Left column: Vehicle Specs
    specs_rows = []
    spec_fields = [
        ("transmission", "Transmission"),
        ("fuel_type", "Fuel Type"),
        ("body_type", "Body Type"),
        ("drive_type", "Drive Type"),
        ("color", "Color"),
        ("manufacturer", "Manufacturer"),
    ]

    for field, label in spec_fields:
        value = get_value(field)
        if value != "N/A":
            specs_rows.append(
                html.Div(
                    [
                        html.Span(label, className="detail-label"),
                        html.Span(value, className="detail-value"),
                    ],
                    className="detail-row",
                )
            )

    # Right column: Listing & Context
    context_rows = []

    # Country
    country = get_value('country')
    if country != "N/A":
        context_rows.append(
            html.Div(
                [
                    html.Span("Country", className="detail-label"),
                    html.Span(country, className="detail-value"),
                ],
                className="detail-row",
            )
        )

    # Additional fields
    additional_fields = set(vehicle_data.keys()) - {
        "price", "mileage", "on_road_year", "owner_count", "transmission",
        "fuel_type", "body_type", "drive_type", "color", "manufacturer",
        "country", "url", "vehicle", "ppk_zscore"
    }
    for field in sorted(additional_fields):
        value = vehicle_data.get(field)
        if pd.notna(value) and str(value).strip() and str(value) != "nan":
            context_rows.append(
                html.Div(
                    [
                        html.Span(field.replace("_", " ").title(), className="detail-label"),
                        html.Span(str(value), className="detail-value"),
                    ],
                    className="detail-row",
                )
            )

    # URL handling
    url_value = None
    if "url" in vehicle_data and pd.notna(vehicle_data.get("url")) and str(vehicle_data.get("url")).strip():
        url_value = str(vehicle_data.get("url")).strip()
        if not url_value.startswith(("http://", "https://")):
            url_value = "https://" + url_value

    # D) Primary CTA Button
    cta_button = None
    if url_value:
        cta_button = html.A(
            "Open Listing →",
            href=url_value,
            target="_blank",
            rel="noopener noreferrer",
            className="vehicle-cta-button",
        )

    # Build two-column layout
    details_section = html.Div(
        [
            html.Div(
                [
                    html.H6("Vehicle Specs", className="details-section-title"),
                    html.Div(specs_rows, className="detail-list"),
                ],
                className="vehicle-details-column",
            ),
            html.Div(
                [
                    html.H6("Listing & Context", className="details-section-title"),
                    html.Div(context_rows, className="detail-list"),
                    cta_button if cta_button else html.Div(),
                ],
                className="vehicle-details-column",
            ),
        ],
        className="vehicle-details-grid",
    )

    # Combine all sections
    modal_body = html.Div(
        [
            kpi_row,
            details_section,
        ],
        className="vehicle-modal-body-content",
    )

    return True, modal_title, modal_body


@app.callback(
    Output("model-line", "figure"),
    Output("model-mini-kpis", "children"),
    Output("depreciation-trends", "children"),
    Input("model-selected", "value"),
)
def update_model(manufacturers):
    manufacturers = (manufacturers or [])[:5]
    fig, depreciation_data = fig_price_depreciation(manufacturers, df)

    # KPI cards for up to 5 manufacturers
    cards = []
    dff = df[df["manufacturer"].isin(manufacturers)].copy()

    for idx, m in enumerate(manufacturers[:5]):
        md = dff[dff["manufacturer"] == m]
        if md.empty:
            continue

        # High-contrast distinct colors for manufacturer cards (matching chart colors)
        card_colors = [
            {"bg": "#FFFFFF", "border": "#0A4174", "text": "#0A4174", "accent": "rgba(10, 65, 116, 0.04)"},      # Dark navy
            {"bg": "#FFFFFF", "border": "#E07B39", "text": "#C06830", "accent": "rgba(224, 123, 57, 0.04)"},    # Warm orange
            {"bg": "#FFFFFF", "border": "#2D8659", "text": "#236B47", "accent": "rgba(45, 134, 89, 0.04)"},     # Forest green
            {"bg": "#FFFFFF", "border": "#7B4B94", "text": "#6A3F80", "accent": "rgba(123, 75, 148, 0.04)"},    # Rich purple
            {"bg": "#FFFFFF", "border": "#C74B50", "text": "#A83E43", "accent": "rgba(199, 75, 80, 0.04)"},     # Coral red
        ]
        card_style = card_colors[idx % len(card_colors)]
        
        cards.append(
            html.Div(
                style={
                    "background": f"linear-gradient(135deg, {card_style['bg']} 0%, {card_style['accent']} 100%)",
                    "borderLeft": f"4px solid {card_style['border']}",
                    "border": f"1px solid rgba(10, 65, 116, 0.06)",
                    "borderLeftWidth": "4px",
                    "borderLeftColor": card_style['border'],
                    "borderRadius": "14px",
                    "padding": "18px 18px 18px 22px",
                    "marginBottom": "14px",
                    "boxShadow": "0 4px 16px rgba(0, 29, 57, 0.06)",
                    "transition": "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
                },
                children=[
                    html.Div(m, style={
                        "fontWeight": 700, 
                        "marginBottom": "12px", 
                        "color": card_style["text"], 
                        "fontSize": "16px",
                        "letterSpacing": "-0.3px",
                    }),
                    html.Div(f"💰 Avg Price: ₪{md['price'].mean():,.0f}", style={"color": "#334155", "fontSize": "13px", "marginBottom": "4px"}),
                    html.Div(f"🚗 Avg Mileage: {md['mileage'].mean():,.0f} km", style={"color": "#64748B", "fontSize": "13px", "marginBottom": "4px"}),
                    html.Div(f"📊 Listings: {len(md):,}", style={"color": "#64748B", "fontSize": "13px"}),
                ],
            )
        )

    if not cards:
        cards = [html.Div("Select manufacturers to view detailed insights", className="small-muted")]

    # Depreciation trends section (with collapsible methodology, as you already had)
    if depreciation_data:
        trend_items = []
        for manufacturer, data_ in depreciation_data.items():
            color = data_["color"]
            dep_pct = data_["depreciation_pct"]

            # Meaningful status colors
            if dep_pct < 3.5:
                status_color = "#2D8659"  # Green - Excellent retention
                sentiment = "Excellent Retention"
            elif dep_pct < 4.5:
                status_color = "#F59E0B"  # Orange - Normal depreciation
                sentiment = "Normal Depreciation"
            else:
                status_color = "#DC2626"  # Red - High depreciation
                sentiment = "High Depreciation"

            # Create the card with manufacturer color and animations
            trend_items.append(
                html.Div(
                    className="depreciation-item",
                    style={
                        "background": "#FFFFFF",
                        "borderLeft": f"4px solid {color}",
                        "border": f"1px solid rgba(10, 65, 116, 0.08)",
                        "borderLeftWidth": "4px",
                        "borderLeftColor": color,
                        "borderRadius": "14px",
                        "padding": "20px 24px",
                        "marginBottom": "12px",
                        "boxShadow": "0 4px 16px rgba(0, 29, 57, 0.06)",
                        "position": "relative",
                        "overflow": "hidden",
                        "transition": "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
                        "cursor": "default",
                    },
                    children=[
                        # Subtle background accent
                        html.Div(
                            style={
                                "position": "absolute",
                                "top": 0,
                                "left": 0,
                                "right": 0,
                                "bottom": 0,
                                "background": f"linear-gradient(135deg, {color}08 0%, transparent 50%)",
                                "pointerEvents": "none",
                            }
                        ),
                        html.Div(
                            style={"position": "relative", "zIndex": 1},
                            children=[
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    [
                                                        # Manufacturer color indicator dot (matches chart line) with pulse
                                                        html.Div(
                                                            className="status-dot",
                                                            style={
                                                                "width": "14px",
                                                                "height": "14px",
                                                                "borderRadius": "50%",
                                                                "background": color,  # Use manufacturer color, not status
                                                                "display": "inline-block",
                                                                "marginRight": "12px",
                                                                "verticalAlign": "middle",
                                                                "boxShadow": f"0 2px 6px {color}40",
                                                                "color": color,  # For pulse animation
                                                            }
                                                        ),
                                                        # Manufacturer name
                                                        html.Span(
                                                            manufacturer,
                                                            style={
                                                                "fontWeight": 700,
                                                                "fontSize": "16px",
                                                                "color": "#001D39",
                                                                "verticalAlign": "middle",
                                                            }, 
                                                        ),
                                                    ],
                                                    style={"marginBottom": "10px", "display": "flex", "alignItems": "center"},
                                                ),
                                                html.Div(
                                                    [
                                                        html.Span(
                                                            "Depreciation Trend: ",
                                                            style={"color": "#6B7280", "fontSize": "12px", "fontWeight": 500},
                                                        ),
                                                        html.Span(
                                                            sentiment,
                                                            style={"color": status_color, "fontSize": "12px", "fontWeight": 600},
                                                        ),
                                                    ]
                                                ),
                                            ],
                                            md=6,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    style={"marginBottom": "8px"},
                                                    children=[
                                                        html.Div(
                                                            f"{dep_pct:.1f}%",
                                                            className="depreciation-percentage",
                                                            style={
                                                                "fontSize": "26px",
                                                                "fontWeight": 800,
                                                                "color": status_color, 
                                                                "textAlign": "right",
                                                                "marginBottom": "8px",
                                                            },
                                                        ),
                                                        # Animated progress bar
                                                        html.Div(
                                                            style={
                                                                "height": "10px",
                                                                "background": "rgba(148, 163, 184, 0.12)",
                                                                "borderRadius": "5px",
                                                                "overflow": "hidden",
                                                            },
                                                            children=[
                                                                html.Div(
                                                                    className="depreciation-bar-fill",
                                                                    style={
                                                                        "width": f"{min((dep_pct / 7) * 100, 100)}%",
                                                                        "height": "100%",
                                                                        "background": f"linear-gradient(90deg, {status_color}, {status_color}CC)",
                                                                        "borderRadius": "5px",
                                                                        "boxShadow": f"0 2px 8px {status_color}40",
                                                                    }
                                                                )
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    "Value Loss Rate (per 10k km)",
                                                    style={
                                                        "fontSize": "10px",
                                                        "color": "#9CA3AF",
                                                        "textAlign": "right",
                                                        "textTransform": "uppercase",
                                                        "letterSpacing": "0.5px",
                                                        "fontWeight": 500,
                                                    },
                                                ),
                                            ],
                                            md=6,
                                        ),
                                    ]
                                ),
                                html.Hr(style={"margin": "14px 0", "opacity": 0.12}),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            html.Div(
                                                [
                                                    html.Div("Initial Price", style={"fontSize": "10px", "color": "#9CA3AF", "textTransform": "uppercase", "marginBottom": "3px", "fontWeight": 500}),
                                                    html.Div(f"₪{data_['first_price']:,.0f}", style={"fontSize": "15px", "fontWeight": 700, "color": "#374151"}),
                                                ]
                                            ),
                                            md=4,
                                        ),
                                        dbc.Col(
                                            html.Div(
                                                [
                                                    html.Div("Final Price", style={"fontSize": "10px", "color": "#9CA3AF", "textTransform": "uppercase", "marginBottom": "3px", "fontWeight": 500}),
                                                    html.Div(f"₪{data_['last_price']:,.0f}", style={"fontSize": "15px", "fontWeight": 700, "color": "#374151"}),
                                                ]
                                            ),
                                            md=4,
                                        ),
                                        dbc.Col(
                                            html.Div(
                                                [
                                                    html.Div("Value Lost", style={"fontSize": "10px", "color": "#9CA3AF", "textTransform": "uppercase", "marginBottom": "3px", "fontWeight": 500}),
                                                    html.Div(f"₪{data_['first_price'] - data_['last_price']:,.0f}", style={"fontSize": "15px", "fontWeight": 700, "color": status_color}),
                                                ]
                                            ),
                                            md=4,
                                        ),
                                    ]
                                ),
                            ],
                        ),
                    ],
                )
            )

        depreciation_section = html.Div(
            className="graph-card",
            style={"padding": "22px"},
            children=[
                html.Div(
                    [
                        html.Div(
                            style={"display": "flex", "alignItems": "center", "gap": "10px", "marginBottom": "8px"},
                            children=[
                                html.Div(
                                    "Depreciation Analysis",
                                    style={
                                        "fontSize": "18px",
                                        "fontWeight": 700,
                                        "color": "#374151",
                                    },
                                ),
                                html.Div(
                                    id="calc-info-button",
                                    children="ℹ️",
                                    style={
                                        "width": "36px",
                                        "height": "36px",
                                        "borderRadius": "50%",
                                        "background": "rgba(100, 116, 139, 0.1)",
                                        "border": "1px solid rgba(100, 116, 139, 0.25)",
                                        "display": "flex",
                                        "alignItems": "center",
                                        "justifyContent": "center",
                                        "cursor": "pointer",
                                        "fontSize": "14px",
                                        "transition": "all 0.25s ease",
                                    },
                                    title="Click to see calculation methodology",
                                ),
                            ],
                        ),
                        html.Div(
                            "Value retention trends across selected models",
                            style={"fontSize": "13px", "color": "#9CA3AF", "marginBottom": "18px"},
                        ),
                        dbc.Collapse(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H6(
                                            "Calculation Methodology",
                                            style={"fontWeight": 700, "marginBottom": "12px", "color": "#374151"},
                                        ),
                                        html.P(
                                            [
                                                "The depreciation percentage is calculated using a robust statistical approach:"],
                                            style={"marginBottom": "12px", "fontSize": "13px", "color": "#6B7280"},
                                        ),
                                        html.Ol(
                                            [
                                                html.Li(
                                                    [
                                                        html.Strong("Data Sorting: ", style={"color": "#374151"}),
                                                        html.Span("All vehicles of the selected model are sorted by mileage (low to high).", style={"color": "#6B7280"}),
                                                    ],
                                                    style={"marginBottom": "8px", "fontSize": "13px"},
                                                ),
                                                html.Li(
                                                    [
                                                        html.Strong("Group Selection: ", style={"color": "#374151"}),
                                                        html.Span("The bottom 20% (lowest mileage, minimum 3 vehicles) and top 20% (highest mileage, minimum 3 vehicles) are selected.", style={"color": "#6B7280"}),
                                                    ],
                                                    style={"marginBottom": "8px", "fontSize": "13px"},
                                                ),
                                                html.Li(
                                                    [
                                                        html.Strong("Average Calculation: ", style={"color": "#374151"}),
                                                        html.Span("The average price is computed for each group separately.", style={"color": "#6B7280"}),
                                                    ],
                                                    style={"marginBottom": "8px", "fontSize": "13px"},
                                                ),
                                                html.Li(
                                                    [
                                                        html.Strong("Depreciation Formula: ", style={"color": "#374151"}),
                                                        html.Code(
                                                            "Depreciation Score = (Price Drop % / Mileage Difference) × 10,000 km",
                                                            style={
                                                                "background": "rgba(168, 150, 168, 0.12)",
                                                                "padding": "2px 6px",
                                                                "borderRadius": "4px",
                                                                "fontSize": "11px",
                                                                "marginLeft": "6px",
                                                                "color": "#374151",
                                                            },
                                                        ),
                                                    ],
                                                    style={"marginBottom": "8px", "fontSize": "13px"},
                                                ),
                                            ],
                                            style={"paddingLeft": "20px", "marginBottom": "12px"},
                                        ),
                                        html.Div(
                                            [
                                                html.Strong("Calculation Method: ", style={"color": "#1F2937", "fontSize": "15px", "fontWeight": "700"}),
                                                html.Span("We normalize the score per 10,000 km to ensure fair comparison between high and low mileage vehicles. We use the average of the top and bottom 20% of listings to eliminate outliers and ensure statistical stability.", style={"color": "#374151", "fontSize": "14px"}),
                                            ],
                                            style={
                                                "background": "#f1f5f9",
                                                "padding": "16px 20px",
                                                "borderRadius": "8px",
                                                "border": "1px solid #e2e8f0",
                                                "marginTop": "24px",
                                            },
                                        ),
                                    ]
                                ),
                                style={
                                    "background": "rgba(255, 255, 253, 0.95)",
                                    "border": "1px solid rgba(148, 163, 184, 0.2)",
                                    "marginBottom": "18px",
                                },
                            ),
                            id="calc-explanation-collapse",
                            is_open=False,
                        ),
                    ]
                ),
                *trend_items,
            ],
        )
    else:
        depreciation_section = html.Div()

    return fig, cards, depreciation_section


@app.callback(
    Output("group-butterfly", "figure"),
    Output("group-details", "children"),
    Input("ga-model", "value"),
    Input("ga-year", "value"),
    Input("ga-trans", "value"),
    Input("gb-model", "value"),
    Input("gb-year", "value"),
    Input("gb-trans", "value"),
)
def update_groups(ma, ya, ta, mb, yb, tb):
    if not all([ma, ya, ta, mb, yb, tb]):
        fig = go.Figure()
        fig.update_layout(title=dict(text="<b>Complete all selections to compare groups</b>"), height=500)
        return fig, dbc.Alert("⚠️ Please select all filter options to proceed with comparison.", color="warning")

    group_a = df[(df["vehicle"] == ma) & (df["on_road_year"].between(ya[0], ya[1])) & (df["transmission"] == ta)]
    group_b = df[(df["vehicle"] == mb) & (df["on_road_year"].between(yb[0], yb[1])) & (df["transmission"] == tb)]

    if len(group_a) == 0 or len(group_b) == 0:
        fig = go.Figure()
        fig.update_layout(title=dict(text="<b>Insufficient data for comparison</b>"), height=500)
        return fig, dbc.Alert("⚠️ One or both groups have no data. Please adjust your filter criteria.", color="danger")

    fig, metrics_data = fig_group_comparison(group_a, group_b)

    # Determine which group is better for each metric (lower is better for Price per KM and Price Stability)
    def get_advantage(metric_name, val_a, val_b):
        if "Price per KM" in metric_name or "Price Stability" in metric_name or "(σ)" in metric_name:
            # Lower is better
            return "A" if val_a < val_b else "B" if val_b < val_a else "Tie"
        else:
            # For Avg Mileage and Avg Price, don't highlight advantage (context-dependent)
            return "Tie"

    # Generate comparison insight
    price_per_km_a = metrics_data['Price per KM'][0]
    price_per_km_b = metrics_data['Price per KM'][1]
    stability_a = metrics_data['Price Stability (σ)'][0]
    stability_b = metrics_data['Price Stability (σ)'][1]

    insights = []
    if price_per_km_a < price_per_km_b:
        insights.append("better value per km")
    elif price_per_km_b < price_per_km_a:
        insights.append("better value per km")

    if stability_a < stability_b:
        insights.append("higher price stability")
    elif stability_b < stability_a:
        insights.append("higher price stability")

    # Determine winning group
    winning_group = None
    if insights:
        if price_per_km_a < price_per_km_b or stability_a < stability_b:
            winning_group = "A"
        else:
            winning_group = "B"

    if insights:
        # Build insight with colored group name (vibrant colors)
        group_name_span = html.Span(
            f"Group {winning_group}",
            style={
                "color": "#4A90D9" if winning_group == "A" else "#9F7AEA",  # Vibrant ocean blue / warm purple
                "fontWeight": 700,
            }
        )
        insight_content = [
            group_name_span,
            html.Span(f" shows {', '.join(insights[:2])}.", style={"fontWeight": 500}),
        ]
        insight_text_structured = insight_content
    else:
        insight_text_structured = [
            html.Span("Both groups show similar value characteristics.", style={"fontWeight": 500}),
        ]

    # Compact KPI strip for Group A/B
    group_summary = dbc.Row(
        className="g-3 mb-4",
        children=[
            dbc.Col(
                html.Div(
                    [
                        html.Div("A", className="group-label-badge",
                                 style={"background": "rgba(74, 144, 217, 0.2)", "color": "#4A90D9"}),
                        html.Div(
                            [
                                html.Div(ma[:30] + ("..." if len(ma) > 30 else ""), className="group-name"),
                                html.Div(f"{len(group_a):,} vehicles", className="group-count"),
                            ],
                            className="group-info",
                        ),
                    ],
                    className="group-summary-card",
                ),
                md=6,
            ),
            dbc.Col(
                html.Div(
                    [
                        html.Div("B", className="group-label-badge",
                                 style={"background": "rgba(159, 122, 234, 0.2)", "color": "#9F7AEA"}),
                                        html.Div(
                            [
                                html.Div(mb[:30] + ("..." if len(mb) > 30 else ""), className="group-name"),
                                html.Div(f"{len(group_b):,} vehicles", className="group-count"),
                            ],
                            className="group-info",
                        ),
                    ],
                    className="group-summary-card",
                ),
                                    md=6,
                                ),
        ],
    )

    # Comparison insight line - Enhanced with icon and colored group
    insight_section = html.Div(
                                    [
                                        html.Div(
                insight_text_structured,
                className="comparison-insight",
            ),
        ],
        className="comparison-insight-container",
    )

    # Comparison table
    metric_rows = []
    metric_labels = {
        "Price per KM": "Price per KM",
        "Price Stability (σ)": "Price Stability (σ)",
        "Avg Mileage": "Avg Mileage",
        "Avg Price": "Avg Price",
    }

    for metric_key, metric_label in metric_labels.items():
        val_a = metrics_data[metric_key][0]
        val_b = metrics_data[metric_key][1]
        advantage = get_advantage(metric_key, val_a, val_b)

        # Format values
        if "KM" in metric_key:
            formatted_a = f"₪{val_a:.2f}"
            formatted_b = f"₪{val_b:.2f}"
        elif "Stability" in metric_key:
            formatted_a = f"₪{val_a:,.0f}"
            formatted_b = f"₪{val_b:,.0f}"
        elif "Mileage" in metric_key:
            formatted_a = f"{val_a:,.0f} km"
            formatted_b = f"{val_b:,.0f} km"
        else:
            formatted_a = f"₪{val_a:,.0f}"
            formatted_b = f"₪{val_b:,.0f}"

        # Determine winner styling - simple bold + checkmark in circle
        # Checkmark badge for Group A (blue circle with white checkmark)
        checkmark_a = html.Span(
            html.Img(
                src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='20 6 9 17 4 12'%3E%3C/polyline%3E%3C/svg%3E",
                style={"width": "14px", "height": "14px"},
            ),
            style={
                "display": "inline-flex",
                "alignItems": "center",
                "justifyContent": "center",
                "width": "22px",
                "height": "22px",
                "borderRadius": "50%",
                "backgroundColor": "#3B82F6",
                "marginLeft": "8px",
                "verticalAlign": "middle",
            },
        ) if advantage == "A" else None
        
        # Checkmark badge for Group B (purple circle with white checkmark)
        checkmark_b = html.Span(
            html.Img(
                src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='20 6 9 17 4 12'%3E%3C/polyline%3E%3C/svg%3E",
                style={"width": "14px", "height": "14px"},
            ),
            style={
                "display": "inline-flex",
                "alignItems": "center",
                "justifyContent": "center",
                "width": "22px",
                "height": "22px",
                "borderRadius": "50%",
                "backgroundColor": "#8B5CF6",
                "marginLeft": "8px",
                "verticalAlign": "middle",
            },
        ) if advantage == "B" else None

        # Build value divs with optional bold and checkmark
        value_a_content = [
            html.Span(formatted_a, style={"fontWeight": 700 if advantage == "A" else 500}),
        ]
        if checkmark_a:
            value_a_content.append(checkmark_a)
        
        value_b_content = [
            html.Span(formatted_b, style={"fontWeight": 700 if advantage == "B" else 500}),
        ]
        if checkmark_b:
            value_b_content.append(checkmark_b)

        metric_rows.append(
            html.Div(
                [
                    html.Div(metric_label, className="comparison-metric-label"),
                    html.Div(
                        value_a_content,
                        className="comparison-metric-value comparison-metric-value-center",
                        style={"display": "flex", "alignItems": "center", "justifyContent": "center"},
                    ),
                    html.Div(
                        value_b_content,
                        className="comparison-metric-value comparison-metric-value-center",
                        style={"display": "flex", "alignItems": "center", "justifyContent": "center"},
                    ),
                ],
                className="comparison-table-row",
            )
        )

    comparison_table = html.Div(
        [
            html.Div(
                [
                    html.Div("Metric", className="comparison-table-header"),
                    html.Div(
                        "Group A", 
                        className="comparison-table-header comparison-table-header-center",
                        style={"color": "#4A90D9"}
                    ),
                    html.Div(
                        "Group B", 
                        className="comparison-table-header comparison-table-header-center",
                        style={"color": "#9F7AEA"}
                    ),
                ],
                className="comparison-table-header-row",
            ),
            html.Div(metric_rows, className="comparison-table-body"),
        ],
        className="comparison-table",
    )

    detail = dbc.Row(
        className="g-4",
        children=[
            dbc.Col(
                html.Div(
                    [
                        group_summary,
                        insight_section,
                        comparison_table,
                    ],
                    className="comparison-container",
                ),
                md=12,
            ),
        ],
    )

    return fig, detail


@app.callback(
    Output("calc-explanation-collapse", "is_open"),
    Input("calc-info-button", "n_clicks"),
    State("calc-explanation-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_calc_explanation(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


@app.callback(
    Output("buyer-methodology-collapse", "is_open"),
    Input("buyer-info-button", "n_clicks"),
    State("buyer-methodology-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_buyer_methodology(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    # Use localhost for local development, 0.0.0.0 for production deployment
    host = "127.0.0.1" if os.environ.get("PORT") is None else "0.0.0.0"
    app.run(host=host, port=port, debug=debug)
