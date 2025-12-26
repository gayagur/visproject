# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd

from dash import Dash, html, dcc, Input, Output, State, ALL, callback_context
import dash_bootstrap_components as dbc

import plotly.graph_objects as go
import plotly.io as pio

# Disable Jupyter integration
os.environ["DASH_DISABLE_JUPYTER"] = "True"

# ========================= CONFIG =========================
CSV_PATH = os.getenv(
    "CAR_CSV_PATH",
    r"C:\Users\gayag\Desktop\לימודים\שנה ג\ויזואליזציה\פרוייקט ויזו\cars_dataset_cleaned.csv",
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
COLORS = {
    "blue": "#60A5FA",
    "purple": "#8B5CF6",
    "cyan": "#06B6D4",
    "orange": "#F59E0B",
    "green": "#10B981",
    "red": "#EF4444",
    "pink": "#EC4899",
    "indigo": "#6366F1",
}

COLOR_SCALE = [
    COLORS["blue"],
    COLORS["purple"],
    COLORS["cyan"],
    COLORS["orange"],
    COLORS["green"],
    COLORS["red"],
    COLORS["pink"],
    COLORS["indigo"],
]

car_template = go.layout.Template(
    layout=dict(
        font=dict(family="Inter, system-ui, sans-serif", size=13, color="#F9FAFB"),
        paper_bgcolor="rgba(17, 24, 39, 0.5)",
        plot_bgcolor="rgba(17, 24, 39, 0.3)",
        margin=dict(l=60, r=40, t=80, b=60),
        title=dict(
            x=0.5,
            xanchor="center",
            font=dict(size=20, color="#F9FAFB", family="Inter"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(17, 24, 39, 0.8)",
            bordercolor="rgba(75, 85, 99, 0.3)",
            borderwidth=1,
            font=dict(color="#F9FAFB"),
        ),
        xaxis=dict(
            gridcolor="rgba(75, 85, 99, 0.2)",
            zeroline=False,
            linecolor="rgba(75, 85, 99, 0.3)",
            tickcolor="rgba(75, 85, 99, 0.3)",
            color="#9CA3AF",
        ),
        yaxis=dict(
            gridcolor="rgba(75, 85, 99, 0.2)",
            zeroline=False,
            linecolor="rgba(75, 85, 99, 0.3)",
            tickcolor="rgba(75, 85, 99, 0.3)",
            color="#9CA3AF",
        ),
        hoverlabel=dict(
            bgcolor="rgba(17, 24, 39, 0.95)",
            bordercolor="rgba(59, 130, 246, 0.5)",
            font=dict(family="Inter", color="#F9FAFB"),
        ),
    )
)
pio.templates["car_theme"] = car_template
pio.templates.default = "car_theme"

# ========================= APP =========================
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "Premium Car Analytics"

# Add custom CSS for info button hover effect
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            #calc-info-button:hover,
            #buyer-info-button:hover {
                background: rgba(59, 130, 246, 0.4) !important;
                border-color: rgba(59, 130, 246, 0.8) !important;
                transform: scale(1.1);
                box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
            }
            .hover-link:hover {
                background: rgba(96, 165, 250, 0.2) !important;
                color: #93C5FD !important;
                transform: scale(1.05);
            }
            .deal-card-hover {
                position: relative;
            }
            .deal-card-hover:hover {
                transform: scale(1.05) !important;
                border-color: #60A5FA !important;
                background: linear-gradient(135deg, rgba(96, 165, 250, 0.2), rgba(96, 165, 250, 0.1)) !important;
                box-shadow: 0 8px 24px rgba(96, 165, 250, 0.4) !important;
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
                opacity: 0.7;
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
    gradients = {
        "blue-purple": "linear-gradient(135deg, #3B82F6, #8B5CF6)",
        "cyan-blue": "linear-gradient(135deg, #06B6D4, #3B82F6)",
        "orange-red": "linear-gradient(135deg, #F59E0B, #EF4444)",
        "green-cyan": "linear-gradient(135deg, #10B981, #06B6D4)",
    }

    return html.Div(
        className="kpi-card",
        children=[
            html.Div(
                icon_text,
                className="kpi-icon",
                style={"background": gradients.get(gradient, gradients["blue-purple"])},
            ),
            html.Div(label, className="kpi-label"),
            html.Div(value, className="kpi-value"),
            html.Div(sub, className="kpi-sub"),
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
                textfont=dict(size=13, color="#F9FAFB", family="Inter"),
                hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Percent: %{percent}<extra></extra>",
                marker=dict(
                    colors=COLOR_SCALE,
                    line=dict(color="rgba(17, 24, 39, 0.8)", width=3),
                ),
                pull=[0.05] * len(labels),
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size:14px;color:#9CA3AF'>{subtitle}</span>"
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
                text="<b>❌ No data matches your filters</b><br><span style='font-size:14px;color:#9CA3AF'>Try adjusting your criteria</span>"
            ),
            height=650,
        )
        return fig, []

    # Aggregate per vehicle
    vehicle_stats = (
        dff.groupby("vehicle")
        .agg({"price": ["mean", "std", "count"], "mileage": ["mean", "std"]})
        .reset_index()
    )
    vehicle_stats.columns = ["vehicle", "avg_price", "price_std", "count", "avg_mileage", "mileage_std"]

    # Value score: avg_price divided by (avg_mileage/1000 + 1)
    vehicle_stats["value_score"] = vehicle_stats["avg_price"] / (vehicle_stats["avg_mileage"] / 1000 + 1)

    # Normalize to 0-100 where higher is better (inverted scale)
    min_val = vehicle_stats["value_score"].min()
    max_val = vehicle_stats["value_score"].max()
    if max_val > min_val:
        vehicle_stats["value_normalized"] = 100 - (
            (vehicle_stats["value_score"] - min_val) / (max_val - min_val) * 100
        )
    else:
        vehicle_stats["value_normalized"] = 50

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

        # Traffic light colors based on deal quality (value score)
        if color_val >= 75:
            color = "#10B981"  # Green - Excellent deal
            category = "🟢 Excellent Value"
        elif color_val >= 50:
            color = "#F59E0B"  # Yellow/Orange - Good deal
            category = "🟡 Good Value"
        elif color_val >= 25:
            color = "#F97316"  # Orange - Fair deal
            category = "🟠 Fair Value"
        else:
            color = "#EF4444"  # Red - Poor deal
            category = "🔴 Basic Value"

        fig.add_trace(
            go.Scatter(
                x=[row["avg_mileage"]],
                y=[row["avg_price"]],
                mode="markers+text",
                name=row["vehicle"],
                marker=dict(
                    size=row["bubble_size"],
                    color=color,
                    opacity=0.75,
                    line=dict(color="white", width=2.5),
                    sizemode="diameter",
                ),
                text=row["vehicle"].split()[0] if len(row["vehicle"].split()) > 0 else row["vehicle"],
                textposition="top center",
                textfont=dict(size=10, color="white", family="Inter"),
                error_x=dict(
                    type="data",
                    array=[row["mileage_std"]],
                    visible=True,
                    color="rgba(255,255,255,0.25)",
                    thickness=2,
                ),
                error_y=dict(
                    type="data",
                    array=[row["price_std"]],
                    visible=True,
                    color="rgba(255,255,255,0.25)",
                    thickness=2,
                ),
                hovertemplate=f"<b style='color:{color};font-size:16px;'>%{{fullData.name}}</b><br>"
                + f"<b style='color:{color};'>{category}</b><br><br>"
                + f"<span style='color:{color};'>💰 Avg Price: ₪{row['avg_price']:,.0f}</span><br>"
                + f"<span style='color:{color};'>🛣️ Avg Mileage: {row['avg_mileage']:,.0f} km</span><br>"
                + f"<span style='color:{color};'>📊 Available: {row['count']:,} cars</span><br>"
                + f"<span style='color:{color};'>🎯 Value Score: {color_val:.0f}/100</span><br>"
                + f"<span style='color:{color};'>💵 Price/1000km: ₪{row['value_score']:.2f}</span><br>"
                + "<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text="<b>🎯 Smart Buyer Matrix</b><br>"
            + f"<span style='font-size:14px;color:#9CA3AF'>Analyzing {len(dff):,} vehicles across {len(vehicle_stats)} models</span>"
        ),
        xaxis_title="<b>Average Mileage (km)</b>",
        yaxis_title="<b>Average Price (₪)</b>",
        height=650,
        hovermode="closest",
        showlegend=False,
    )

    # Return the figure and the list of vehicles displayed
    displayed_vehicles = vehicle_stats["vehicle"].tolist()
    return fig, displayed_vehicles


def create_best_deals_cards(data: pd.DataFrame, max_results: int = 10, displayed_vehicles: list = None):
    # Best Deals: compute a per-model z-score for price and surface listings that
    # are significantly cheaper than their model mean.
    # Only shows deals from vehicles that are displayed in the Smart Buyer Matrix.

    dff = data.copy()
    
    # Filter to only vehicles displayed in the matrix
    if displayed_vehicles and len(displayed_vehicles) > 0:
        dff = dff[dff["vehicle"].isin(displayed_vehicles)]
    dff["price_zscore"] = np.nan

    for model in dff["vehicle"].unique():
        model_data = dff[dff["vehicle"] == model]
        if len(model_data) >= 5:
            mean_price = model_data["price"].mean()
            std_price = model_data["price"].std()
            if std_price > 0:
                dff.loc[dff["vehicle"] == model, "price_zscore"] = (
                    dff.loc[dff["vehicle"] == model, "price"] - mean_price
                ) / std_price

    # Keep deals that are at least 0.5 std below their model mean
    best_deals = dff[dff["price_zscore"] < -0.5].nsmallest(max_results, "price_zscore")

    if len(best_deals) == 0:
        return html.Div(
            [
                html.H3(
                    "No significant deals found",
                    style={
                        "fontSize": "24px",
                        "fontWeight": 900,
                        "marginBottom": "8px",
                        "color": "#F9FAFB",
                    },
                ),
                html.P(
                    "All vehicles are fairly priced",
                    style={"fontSize": "14px", "color": "#9CA3AF"},
                ),
            ],
            style={"padding": "24px", "textAlign": "center"},
        )

    best_deals = best_deals.sort_values("price_zscore")

    # Calculate normalized values for color (0-1 range)
    z_scores_neg = -best_deals["price_zscore"].values
    z_min, z_max = z_scores_neg.min(), z_scores_neg.max()
    if z_max > z_min:
        z_normalized = (z_scores_neg - z_min) / (z_max - z_min)
    else:
        z_normalized = np.ones(len(z_scores_neg)) * 0.5

    # Create cards in a grid (4 per row)
    cards = []
    for idx, (_, row) in enumerate(best_deals.iterrows()):
        z_norm = z_normalized[idx]
        z_score_neg = -row["price_zscore"]
        
        # Determine color based on normalized z-score
        if z_norm >= 0.75:
            color = "#10B981"  # Dark green - Outstanding
            quality = "Outstanding"
        elif z_norm >= 0.5:
            color = "#34D399"  # Medium green - Excellent
            quality = "Excellent"
        elif z_norm >= 0.25:
            color = "#6EE7B7"  # Light green - Very Good
            quality = "Very Good"
        else:
            color = "#A7F3D0"  # Very light green - Good
            quality = "Good"
        
        # Store row index in the card ID for callback
        card_id = f"deal-card-{idx}"
        cards.append(
            html.Div(
                id={"type": "deal-card", "index": idx},
                className="graph-card deal-card-hover",
                n_clicks=0,
                style={
                    "padding": "20px",
                    "minWidth": "280px",
                    "width": "280px",
                    "flexShrink": 0,
                    "border": f"2px solid {color}",
                    "borderRadius": "16px",
                    "background": f"linear-gradient(135deg, {color}15, {color}05)",
                    "transition": "all 0.3s ease",
                    "cursor": "pointer",
                    "marginRight": "16px",
                    "position": "relative",
                },
                children=[
                    # Hover text overlay
                    html.Div(
                        "Press for more details",
                        className="deal-card-hover-text",
                        style={
                            "position": "absolute",
                            "top": "50%",
                            "left": "50%",
                            "transform": "translate(-50%, -50%)",
                            "color": "#60A5FA",
                            "fontSize": "14px",
                            "fontWeight": 700,
                            "opacity": 0,
                            "transition": "opacity 0.3s ease",
                            "pointerEvents": "none",
                            "zIndex": 10,
                            "textAlign": "center",
                            "background": "rgba(17, 24, 39, 0.9)",
                            "padding": "8px 16px",
                            "borderRadius": "8px",
                            "border": "2px solid #60A5FA",
                        },
                    ),
                    html.Div(
                        "🏆",
                        style={
                            "fontSize": "32px",
                            "textAlign": "center",
                            "marginBottom": "12px",
                        },
                    ),
                    html.Div(
                        row["vehicle"][:30] + ("..." if len(row["vehicle"]) > 30 else ""),
                        style={
                            "fontSize": "16px",
                            "fontWeight": 800,
                            "color": "#F9FAFB",
                            "textAlign": "center",
                            "marginBottom": "16px",
                            "minHeight": "48px",
                        },
                    ),
                    html.Hr(style={"opacity": 0.2, "margin": "12px 0"}),
                    html.Div(
                        [
                            html.Span("💰 Price: ", style={"color": "#9CA3AF", "fontSize": "13px"}),
                            html.Span(
                                f"₪{row['price']:,.0f}",
                                style={"color": color, "fontWeight": 800, "fontSize": "18px"},
                            ),
                        ],
                        style={"marginBottom": "12px", "textAlign": "center"},
                    ),
                    html.Div(
                        [
                            html.Span("📉 Below Average: ", style={"color": "#9CA3AF", "fontSize": "13px"}),
                            html.Span(
                                f"{z_score_neg:.2f} std",
                                style={"color": color, "fontWeight": 700, "fontSize": "15px"},
                            ),
                        ],
                        style={"marginBottom": "12px", "textAlign": "center"},
                    ),
                    html.Div(
                        [
                            html.Span("📊 Quality: ", style={"color": "#9CA3AF", "fontSize": "13px"}),
                            html.Span(
                                quality,
                                style={"color": color, "fontWeight": 700, "fontSize": "14px"},
                            ),
                        ],
                        style={"textAlign": "center", "marginBottom": "8px"},
                    ),
                    html.Div(
                        "💡 Significantly below model average",
                        style={
                            "color": "#9CA3AF",
                            "fontSize": "11px",
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
        subtitle = "Cars priced significantly below their model average"

    return html.Div(
        [
            html.Div(
                [
                    html.H3(
                        f"🏆 Top {len(best_deals)} Best Deals Right Now",
                        style={
                            "fontSize": "24px",
                            "fontWeight": 900,
                            "marginBottom": "8px",
                            "background": "linear-gradient(135deg, #60A5FA, #8B5CF6)",
                            "WebkitBackgroundClip": "text",
                            "WebkitTextFillColor": "transparent",
                            "backgroundClip": "text",
                        },
                    ),
                    html.P(
                        subtitle,
                        style={"fontSize": "14px", "color": "#9CA3AF", "marginBottom": "24px"},
                    ),
                ],
                style={"marginBottom": "24px"},
            ),
            html.Div(
                cards,
                style={
                    "display": "flex",
                    "overflowX": "auto",
                    "overflowY": "visible",
                    "paddingBottom": "24px",
                    "paddingTop": "8px",
                    "paddingLeft": "8px",
                    "paddingRight": "8px",
                    "gap": "0",
                },
                className="best-deals-container",
            ),
        ],
        style={"padding": "24px"},
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
                    line=dict(width=2, color="rgba(17, 24, 39, 0.8)"),
                ),
                hovertemplate="<b>%{fullData.name}</b><br>Mileage: %{x:,.0f} km<br>Price: ₪%{y:,.1f}K<extra></extra>",
            )
        )

        # Depreciation estimate: compare average price of lowest-mileage 20% vs highest-mileage 20%
        raw_manufacturer_data = dff[dff["manufacturer"] == manufacturer].copy()
        if len(raw_manufacturer_data) >= 5:
            sorted_data = raw_manufacturer_data.sort_values("mileage")
            n = len(sorted_data)
            bottom_20pct = sorted_data.head(max(3, int(n * 0.2)))
            top_20pct = sorted_data.tail(max(3, int(n * 0.2)))

            low_mileage_price = bottom_20pct["price"].mean()
            high_mileage_price = top_20pct["price"].mean()

            if low_mileage_price > 0:
                depreciation_pct = ((low_mileage_price - high_mileage_price) / low_mileage_price) * 100
                depreciation_data[manufacturer] = {
                    "depreciation_pct": max(0, depreciation_pct),
                    "first_price": low_mileage_price,
                    "last_price": high_mileage_price,
                    "color": color,
                    "sample_size": n,
                }

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
        "Price Stability": [float(group_a["price"].std()), float(group_b["price"].std())],
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

    fig.add_trace(
        go.Bar(
            y=metrics,
            x=[-v for v in a_norm],
            name="Group A",
            orientation="h",
            marker=dict(
                color=COLORS["blue"],
                line=dict(color="rgba(17, 24, 39, 0.8)", width=2),
            ),
            text=[
                f"₪{metrics_data[m][0]:,.2f}" if ("KM" in m or "Stability" in m) else f"{metrics_data[m][0]:,.0f}"
                for m in metrics
            ],
            textposition="inside",
            textfont=dict(size=13, color="white", family="Inter"),
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
                color=COLORS["purple"],
                line=dict(color="rgba(17, 24, 39, 0.8)", width=2),
            ),
            text=[
                f"₪{metrics_data[m][1]:,.2f}" if ("KM" in m or "Stability" in m) else f"{metrics_data[m][1]:,.0f}"
                for m in metrics
            ],
            textposition="inside",
            textfont=dict(size=13, color="white", family="Inter"),
            hovertemplate="<b>Group B</b><br>%{y}: %{text}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text="<b>Vehicle Value Comparison</b>"),
        barmode="relative",
        height=500,
        xaxis_title="<b>Relative Performance</b>",
        xaxis=dict(
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor="rgba(75, 85, 99, 0.5)",
            range=[-120, 120],
        ),
    )
    fig.update_yaxes(autorange="reversed")

    return fig, metrics_data

# ========================= LAYOUT =========================
app.layout = dbc.Container(
    fluid=True,
    children=[
        # HERO
        html.Div(
            className="hero",
            children=[
                html.H1("🚗 Premium Car Analytics"),
                html.P("Advanced Vehicle Intelligence & Market Insights Platform"),
            ],
        ),

        # OVERVIEW SECTION
        html.Div(
            className="section",
            children=[
                html.Div("📊 Market Overview", className="section-title"),
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
                dbc.Tab(label="🏠 Home", tab_id="tab-home"),
                # dbc.Tab(label="🌍 Market Analysis", tab_id="tab-market"),
                dbc.Tab(label="🔄 Manufacturers Comparison", tab_id="tab-model"),
                dbc.Tab(label="⚖️ Group Comparison", tab_id="tab-group"),
                dbc.Tab(label="🛒 Buyer's Guide", tab_id="tab-buyer"),
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
                            html.Div(id="vehicle-modal-title", children="Vehicle Details", className="vehicle-modal-header-wrapper"),
                            html.Button(
                                "×",
                                className="vehicle-modal-close",
                                n_clicks=0,
                                id="vehicle-modal-close-btn",
                            ),
                        ],
                        className="vehicle-modal-header",
                        style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start", "width": "100%"},
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
                # Welcome Section
                html.Div(
                    className="graph-card",
                    style={"padding": "48px", "marginBottom": "32px", "textAlign": "center"},
                    children=[
                        html.H1(
                            "🚗 Welcome to Premium Car Analytics",
                            style={
                                "fontSize": "42px",
                                "fontWeight": 900,
                                "marginBottom": "20px",
                                "background": "linear-gradient(135deg, #60A5FA, #8B5CF6)",
                                "WebkitBackgroundClip": "text",
                                "WebkitTextFillColor": "transparent",
                                "backgroundClip": "text",
                            },
                        ),
                        html.P(
                            "Advanced Vehicle Intelligence & Market Insights Platform",
                            style={"fontSize": "20px", "color": "#9CA3AF", "marginBottom": "40px"},
                        ),
                        html.Div(
                            style={
                                "display": "flex",
                                "justifyContent": "center",
                                "gap": "24px",
                                "flexWrap": "wrap",
                                "marginTop": "40px",
                            },
                            children=[
                                html.Div(
                                    style={
                                        "background": "rgba(59, 130, 246, 0.1)",
                                        "border": "1px solid rgba(59, 130, 246, 0.3)",
                                        "borderRadius": "16px",
                                        "padding": "24px",
                                        "minWidth": "200px",
                                    },
                                    children=[
                                        html.Div("📊", style={"fontSize": "48px", "marginBottom": "12px"}),
                                        html.H3("Data Analysis", style={"fontSize": "18px", "fontWeight": 700, "marginBottom": "8px"}),
                                        html.P("Comprehensive market insights", style={"fontSize": "14px", "color": "#9CA3AF"}),
                                    ],
                                ),
                                html.Div(
                                    style={
                                        "background": "rgba(139, 92, 246, 0.1)",
                                        "border": "1px solid rgba(139, 92, 246, 0.3)",
                                        "borderRadius": "16px",
                                        "padding": "24px",
                                        "minWidth": "200px",
                                    },
                                    children=[
                                        html.Div("🎯", style={"fontSize": "48px", "marginBottom": "12px"}),
                                        html.H3("Smart Insights", style={"fontSize": "18px", "fontWeight": 700, "marginBottom": "8px"}),
                                        html.P("AI-powered recommendations", style={"fontSize": "14px", "color": "#9CA3AF"}),
                                    ],
                                ),
                                html.Div(
                                    style={
                                        "background": "rgba(16, 185, 129, 0.1)",
                                        "border": "1px solid rgba(16, 185, 129, 0.3)",
                                        "borderRadius": "16px",
                                        "padding": "24px",
                                        "minWidth": "200px",
                                    },
                                    children=[
                                        html.Div("💡", style={"fontSize": "48px", "marginBottom": "12px"}),
                                        html.H3("Best Deals", style={"fontSize": "18px", "fontWeight": 700, "marginBottom": "8px"}),
                                        html.P("Find the perfect vehicle", style={"fontSize": "14px", "color": "#9CA3AF"}),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                
                # Features Section
                dbc.Row(
                    className="g-4 mb-4",
                    children=[
                        dbc.Col(
                            html.Div(
                                className="graph-card",
                                style={"padding": "32px", "height": "100%"},
                                children=[
                                    html.Div(
                                        "🔄 Manufacturers Comparison",
                                        style={"fontSize": "24px", "fontWeight": 800, "marginBottom": "16px", "color": "#60A5FA"},
                                    ),
                                    html.P(
                                        "Compare up to 5 different car models side by side. Analyze price depreciation trends, "
                                        "mileage impact, and value retention over time. Get detailed insights into which models "
                                        "maintain their value best.",
                                        style={"fontSize": "15px", "lineHeight": "1.8", "color": "#9CA3AF"},
                                    ),
                                    html.Ul(
                                        [
                                            html.Li("Price depreciation analysis", style={"marginBottom": "8px", "color": "#9CA3AF"}),
                                            html.Li("Mileage impact visualization", style={"marginBottom": "8px", "color": "#9CA3AF"}),
                                            html.Li("Value retention trends", style={"marginBottom": "8px", "color": "#9CA3AF"}),
                                        ],
                                        style={"marginTop": "20px", "paddingLeft": "20px"},
                                    ),
                                ],
                            ),
                            md=4,
                        ),
                        dbc.Col(
                            html.Div(
                                className="graph-card",
                                style={"padding": "32px", "height": "100%"},
                                children=[
                                    html.Div(
                                        "⚖️ Group Comparison",
                                        style={"fontSize": "24px", "fontWeight": 800, "marginBottom": "16px", "color": "#8B5CF6"},
                                    ),
                                    html.P(
                                        "Compare two different vehicle groups based on model, year range, and transmission type. "
                                        "Analyze price stability, value for money, and average metrics to make informed decisions "
                                        "between different vehicle segments.",
                                        style={"fontSize": "15px", "lineHeight": "1.8", "color": "#9CA3AF"},
                                    ),
                                    html.Ul(
                                        [
                                            html.Li("Side-by-side comparison", style={"marginBottom": "8px", "color": "#9CA3AF"}),
                                            html.Li("Price stability analysis", style={"marginBottom": "8px", "color": "#9CA3AF"}),
                                            html.Li("Value for money metrics", style={"marginBottom": "8px", "color": "#9CA3AF"}),
                                        ],
                                        style={"marginTop": "20px", "paddingLeft": "20px"},
                                    ),
                                ],
                            ),
                            md=4,
                        ),
                        dbc.Col(
                            html.Div(
                                className="graph-card",
                                style={"padding": "32px", "height": "100%"},
                                children=[
                                    html.Div(
                                        "🛒 Buyer's Guide",
                                        style={"fontSize": "24px", "fontWeight": 800, "marginBottom": "16px", "color": "#10B981"},
                                    ),
                                    html.P(
                                        "Smart Buyer Matrix helps you find the best value deals. Use advanced filters to narrow down "
                                        "your search, and discover vehicles priced significantly below their model average. Get real-time "
                                        "recommendations for the best deals available.",
                                        style={"fontSize": "15px", "lineHeight": "1.8", "color": "#9CA3AF"},
                                    ),
                                    html.Ul(
                                        [
                                            html.Li("Smart value scoring", style={"marginBottom": "8px", "color": "#9CA3AF"}),
                                            html.Li("Advanced filtering options", style={"marginBottom": "8px", "color": "#9CA3AF"}),
                                            html.Li("Best deals identification", style={"marginBottom": "8px", "color": "#9CA3AF"}),
                                        ],
                                        style={"marginTop": "20px", "paddingLeft": "20px"},
                                    ),
                                ],
                            ),
                            md=4,
                        ),
                    ],
                ),
                
                # Team Section
                html.Div(
                    className="graph-card",
                    style={"padding": "40px", "marginTop": "32px", "textAlign": "center"},
                    children=[
                        html.H2(
                            "👥 Developed By",
                            style={
                                "fontSize": "32px",
                                "fontWeight": 800,
                                "marginBottom": "32px",
                                "background": "linear-gradient(135deg, #60A5FA, #8B5CF6)",
                                "WebkitBackgroundClip": "text",
                                "WebkitTextFillColor": "transparent",
                                "backgroundClip": "text",
                            },
                        ),
                        html.Div(
                            style={
                                "display": "flex",
                                "justifyContent": "center",
                                "gap": "32px",
                                "flexWrap": "wrap",
                                "marginTop": "24px",
                            },
                            children=[
                                html.Div(
                                    style={
                                        "background": "rgba(59, 130, 246, 0.1)",
                                        "border": "1px solid rgba(59, 130, 246, 0.3)",
                                        "borderRadius": "16px",
                                        "padding": "24px 32px",
                                        "minWidth": "180px",
                                    },
                                    children=[
                                        html.Div("👤", style={"fontSize": "36px", "marginBottom": "8px"}),
                                        html.Div("Gaya Gur", style={"fontSize": "18px", "fontWeight": 700, "color": "#F9FAFB"}),
                                    ],
                                ),
                                html.Div(
                                    style={
                                        "background": "rgba(139, 92, 246, 0.1)",
                                        "border": "1px solid rgba(139, 92, 246, 0.3)",
                                        "borderRadius": "16px",
                                        "padding": "24px 32px",
                                        "minWidth": "180px",
                                    },
                                    children=[
                                        html.Div("👤", style={"fontSize": "36px", "marginBottom": "8px"}),
                                        html.Div("Moran Shavit", style={"fontSize": "18px", "fontWeight": 700, "color": "#F9FAFB"}),
                                    ],
                                ),
                                html.Div(
                                    style={
                                        "background": "rgba(16, 185, 129, 0.1)",
                                        "border": "1px solid rgba(16, 185, 129, 0.3)",
                                        "borderRadius": "16px",
                                        "padding": "24px 32px",
                                        "minWidth": "180px",
                                    },
                                    children=[
                                        html.Div("👤", style={"fontSize": "36px", "marginBottom": "8px"}),
                                        html.Div("Matias Gernilk", style={"fontSize": "18px", "fontWeight": 700, "color": "#F9FAFB"}),
                                    ],
                                ),
                                html.Div(
                                    style={
                                        "background": "rgba(245, 158, 11, 0.1)",
                                        "border": "1px solid rgba(245, 158, 11, 0.3)",
                                        "borderRadius": "16px",
                                        "padding": "24px 32px",
                                        "minWidth": "180px",
                                    },
                                    children=[
                                        html.Div("👤", style={"fontSize": "36px", "marginBottom": "8px"}),
                                        html.Div("Tamar Hagbi", style={"fontSize": "18px", "fontWeight": 700, "color": "#F9FAFB"}),
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
        return dbc.Row(
            className="g-4",
            children=[
                dbc.Col(
                    html.Div(
                        className="filter-card",
                        children=[
                            html.Div(
                                "🔄 Model Comparison",
                                style={"fontWeight": 900, "marginBottom": "12px", "fontSize": "18px"},
                            ),
                            html.Div(
                                "Select up to 5 manufacturers for optimal visualization",
                                className="small-muted",
                                style={"marginBottom": "16px"},
                            ),
                            dcc.Dropdown(
                                id="model-selected",
                                options=[{"label": m, "value": m} for m in sorted(df["manufacturer"].unique())],
                                value=sorted(df["manufacturer"].unique())[:3] if len(df) > 0 else [],
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
                                        "fontWeight": 900,
                                        "background": "linear-gradient(135deg, #60A5FA, #8B5CF6)",
                                        "WebkitBackgroundClip": "text",
                                        "WebkitTextFillColor": "transparent",
                                    },
                                ),
                                html.Div(
                                    id="buyer-info-button",
                                    children="ℹ️",
                                    style={
                                        "width": "56px",
                                        "height": "56px",
                                        "borderRadius": "50%",
                                        "background": "rgba(59, 130, 246, 0.2)",
                                        "border": "1px solid rgba(59, 130, 246, 0.5)",
                                        "display": "flex",
                                        "alignItems": "center",
                                        "justifyContent": "center",
                                        "cursor": "pointer",
                                        "fontSize": "20px",
                                        "transition": "all 0.3s ease",
                                    },
                                    title="Click to see calculation methodology",
                                ),
                            ],
                        ),
                        html.Div(
                            "How the Smart Buyer Matrix and Best Deals are computed",
                            style={"fontSize": "14px", "color": "#9CA3AF", "marginBottom": "12px"},
                        ),
                        dbc.Collapse(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H6(
                                            "📌 Smart Buyer Matrix (Value for Money)",
                                            style={"fontWeight": 800, "marginBottom": "10px", "color": "#60A5FA"},
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
                                                        "Compute average price, price std, average mileage, mileage std, and listing count per model.",
                                                    ],
                                                    style={"marginBottom": "8px", "fontSize": "13px"},
                                                ),
                                                html.Li(
                                                    [
                                                        html.Strong("Value score: "),
                                                        "Compute:",
                                                        html.Code(
                                                            "ValueScore = AvgPrice / (AvgMileage/1000 + 1)",
                                                            style={
                                                                "background": "rgba(139, 92, 246, 0.2)",
                                                                "padding": "2px 6px",
                                                                "borderRadius": "4px",
                                                                "fontSize": "11px",
                                                                "marginLeft": "6px",
                                                            },
                                                        ),
                                                    ],
                                                    style={"marginBottom": "8px", "fontSize": "13px"},
                                                ),
                                                html.Li(
                                                    [
                                                        html.Strong("Normalization: "),
                                                        "Convert ValueScore to a 0–100 scale and invert it so higher means better value:",
                                                        html.Code(
                                                            "ValueNorm = 100 - ((ValueScore - min)/(max-min) * 100)",
                                                            style={
                                                                "background": "rgba(139, 92, 246, 0.2)",
                                                                "padding": "2px 6px",
                                                                "borderRadius": "4px",
                                                                "fontSize": "11px",
                                                                "marginLeft": "6px",
                                                            },
                                                        ),
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
                                            "🏆 Best Deals (Model-relative underpricing)",
                                            style={"fontWeight": 800, "marginBottom": "10px", "color": "#10B981"},
                                        ),
                                        html.Ol(
                                            [
                                                html.Li(
                                                    [
                                                        html.Strong("Per-model baseline: "),
                                                        "For each model with at least 5 listings, compute mean and standard deviation of price.",
                                                    ],
                                                    style={"marginBottom": "8px", "fontSize": "13px"},
                                                ),
                                                html.Li(
                                                    [
                                                        html.Strong("Z-score per listing: "),
                                                        "Compute:",
                                                        html.Code(
                                                            "Z = (Price - ModelMean) / ModelStd",
                                                            style={
                                                                "background": "rgba(16, 185, 129, 0.15)",
                                                                "padding": "2px 6px",
                                                                "borderRadius": "4px",
                                                                "fontSize": "11px",
                                                                "marginLeft": "6px",
                                                            },
                                                        ),
                                                    ],
                                                    style={"marginBottom": "8px", "fontSize": "13px"},
                                                ),
                                                html.Li(
                                                    [
                                                        html.Strong("Deal threshold: "),
                                                        "Keep listings where Z < -0.5 (at least half a standard deviation below the model average).",
                                                    ],
                                                    style={"marginBottom": "8px", "fontSize": "13px"},
                                                ),
                                                html.Li(
                                                    [
                                                        html.Strong("Ranking: "),
                                                        "Sort by most negative Z (best relative deal) and display the top results.",
                                                    ],
                                                    style={"marginBottom": "8px", "fontSize": "13px"},
                                                ),
                                            ],
                                            style={"paddingLeft": "20px"},
                                        ),
                                        html.Div(
                                            [
                                                html.Strong("Note: ", style={"color": "#F59E0B"}),
                                                "These are statistical signals based on listing prices. They do not account for trims, accidents, ownership history, or condition unless those fields are included and modeled.",
                                            ],
                                            style={
                                                "background": "rgba(245, 158, 11, 0.08)",
                                                "padding": "12px",
                                                "borderRadius": "8px",
                                                "border": "1px solid rgba(245, 158, 11, 0.25)",
                                                "fontSize": "13px",
                                                "marginTop": "14px",
                                            },
                                        ),
                                    ]
                                ),
                                style={
                                    "background": "rgba(17, 24, 39, 0.8)",
                                    "border": "1px solid rgba(75, 85, 99, 0.3)",
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
                                            options=[{"label": t, "value": t} for t in sorted(df["transmission"].unique())],
                                            value=[],
                                            multi=True,
                                            placeholder="Select transmission types",
                                        ),
                                        html.Div(style={"height": "12px"}),
                                        html.Label("Fuel Type"),
                                        dcc.Dropdown(
                                            id="buyer-fuel",
                                            options=[{"label": f, "value": f} for f in sorted(df["fuel_type"].unique())],
                                            value=[],
                                            multi=True,
                                            placeholder="Select fuel types",
                                        ),
                                        html.Div(style={"height": "12px"}),
                                        html.Label("מספר ידיים (Owner Count)"),
                                        dcc.Dropdown(
                                            id="buyer-owner-count",
                                            options=[{"label": str(int(o)), "value": int(o)} for o in sorted(df["owner_count"].dropna().unique()) if pd.notna(o)],
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
                                                int(df["price"].quantile(0.25)): f"₪{int(df['price'].quantile(0.25) / 1000)}K",
                                                int(df["price"].quantile(0.5)): f"₪{int(df['price'].quantile(0.5) / 1000)}K",
                                                int(df["price"].quantile(0.75)): f"₪{int(df['price'].quantile(0.75) / 1000)}K",
                                                int(df["price"].quantile(0.95)): f"₪{int(df['price'].quantile(0.95) / 1000)}K",
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
                                                int(df["mileage"].quantile(0.5)): f"{int(df['mileage'].quantile(0.5) / 1000)}K",
                                                int(df["mileage"].quantile(0.95)): f"{int(df['mileage'].quantile(0.95) / 1000)}K",
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
                                                html.H6("💡 Quick Tips", style={"fontWeight": 700, "marginBottom": "12px"}),
                                                html.Ul(
                                                    [
                                                        html.Li("🟢 Green = Best value", className="small-muted", style={"marginBottom": "4px", "fontSize": "12px"}),
                                                        html.Li("⭕ Larger = More available", className="small-muted", style={"marginBottom": "4px", "fontSize": "12px"}),
                                                        html.Li("📊 Hover for details", className="small-muted", style={"marginBottom": "4px", "fontSize": "12px"}),
                                                        html.Li(
                                                            "🎯 Target: Large green bubbles!",
                                                            className="small-muted",
                                                            style={"marginBottom": "4px", "fontWeight": 700, "color": COLORS["green"], "fontSize": "12px"},
                                                        ),
                                                    ],
                                                    style={"paddingLeft": "20px"},
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
                                        )
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
                            },
                        ),
                        html.Label("Model"),
                        dcc.Dropdown(
                            id="ga-model",
                            options=[{"label": t, "value": t} for t in sorted(df["vehicle"].unique())],
                            value=sorted(df["vehicle"].unique())[0] if len(df) else None,
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
                                if any(keyword in str(t) for keyword in ["אוטומטי", "ידני", "automatic", "manual", "אוטומט"])
                            ],
                            value=next((t for t in sorted(df["transmission"].unique()) if any(keyword in str(t) for keyword in ["אוטומטי", "automatic", "אוטומט"])), sorted(df["transmission"].unique())[0] if len(df) > 0 else None),
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
                            },
                        ),
                        html.Label("Model"),
                        dcc.Dropdown(
                            id="gb-model",
                            options=[{"label": t, "value": t} for t in sorted(df["vehicle"].unique())],
                            value=sorted(df["vehicle"].unique())[0] if len(df) else None,
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
                                if any(keyword in str(t) for keyword in ["אוטומטי", "ידני", "automatic", "manual", "אוטומט"])
                            ],
                            value=next((t for t in sorted(df["transmission"].unique()) if any(keyword in str(t) for keyword in ["אוטומטי", "automatic", "אוטומט"])), sorted(df["transmission"].unique())[0] if len(df) > 0 else None),
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
            html.Div([html.Span("🛣️ Avg Mileage: ", style={"fontWeight": 600}), html.B(f"{dff['mileage'].mean():,.0f} km")]),
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
def update_buyer_guide(vehicles, price_range, max_mileage, country, transmission, fuel, owner_count, year_range, max_vehicles):
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
            (dff_deals["on_road_year"] >= year_range_filtered[0]) & (dff_deals["on_road_year"] <= year_range_filtered[1])
        ]

    # Only show best deals from vehicles displayed in the matrix
    deals_cards = create_best_deals_cards(dff_deals, displayed_vehicles=displayed_vehicles)
    
    # Store best deals data for modal
    dff_deals_filtered = dff_deals.copy()
    if displayed_vehicles and len(displayed_vehicles) > 0:
        dff_deals_filtered = dff_deals_filtered[dff_deals_filtered["vehicle"].isin(displayed_vehicles)]
    
    # Calculate z-scores for best deals
    dff_deals_filtered["price_zscore"] = np.nan
    for model in dff_deals_filtered["vehicle"].unique():
        model_data = dff_deals_filtered[dff_deals_filtered["vehicle"] == model]
        if len(model_data) >= 5:
            mean_price = model_data["price"].mean()
            std_price = model_data["price"].std()
            if std_price > 0:
                dff_deals_filtered.loc[dff_deals_filtered["vehicle"] == model, "price_zscore"] = (
                    dff_deals_filtered.loc[dff_deals_filtered["vehicle"] == model, "price"] - mean_price
                ) / std_price
    
    best_deals_data = dff_deals_filtered[dff_deals_filtered["price_zscore"] < -0.5].nsmallest(10, "price_zscore")
    best_deals_data = best_deals_data.sort_values("price_zscore")
    
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
    
    # Find which card was clicked
    if "deal-card" not in triggered_id or not n_clicks_list:
        return is_open, "Vehicle Details", html.Div()
    
    # Find the index of the clicked card
    card_index = None
    for i, n_clicks in enumerate(n_clicks_list):
        if n_clicks and n_clicks > 0:
            card_index = i
            break
    
    if card_index is None or card_index >= len(deals_data) or not deals_data:
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
                    html.Span("🚗", style={"fontSize": "20px", "marginRight": "8px"}),
                    html.Span(
                        vehicle_name,
                        style={
                            "fontSize": "24px",
                            "fontWeight": 700,
                            "color": "#F9FAFB",
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
        "country", "url", "vehicle", "price_zscore"
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

    # KPI cards for up to 3 manufacturers
    cards = []
    dff = df[df["manufacturer"].isin(manufacturers)].copy()

    for idx, m in enumerate(manufacturers[:3]):
        md = dff[dff["manufacturer"] == m]
        if md.empty:
            continue

        color = COLOR_SCALE[idx % len(COLOR_SCALE)]
        cards.append(
            html.Div(
                style={
                    "background": "rgba(17, 24, 39, 0.5)",
                    "border": f"1px solid {color}",
                    "borderRadius": "12px",
                    "padding": "16px",
                    "marginBottom": "12px",
                    "backdropFilter": "blur(10px)",
                    "boxShadow": f"0 4px 16px {color}40",
                },
                children=[
                    html.Div(m, style={"fontWeight": 900, "marginBottom": "8px", "color": color, "fontSize": "15px"}),
                    html.Div(f"💰 Avg Price: ₪{md['price'].mean():,.0f}", className="small-muted"),
                    html.Div(f"🛣️ Avg Mileage: {md['mileage'].mean():,.0f} km", className="small-muted"),
                    html.Div(f"📊 Listings: {len(md):,}", className="small-muted"),
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

            if dep_pct < 15:
                icon = "🟢"
                sentiment = "Excellent"
            elif dep_pct < 30:
                icon = "🟡"
                sentiment = "Good"
            elif dep_pct < 45:
                icon = "🟠"
                sentiment = "Moderate"
            else:
                icon = "🔴"
                sentiment = "High"

            trend_items.append(
                html.Div(
                    className="depreciation-item",
                    style={
                        "background": "rgba(17, 24, 39, 0.5)",
                        "border": f"2px solid {color}",
                        "borderRadius": "16px",
                        "padding": "20px 24px",
                        "marginBottom": "12px",
                        "backdropFilter": "blur(10px)",
                        "boxShadow": f"0 8px 24px {color}30",
                        "position": "relative",
                        "overflow": "hidden",
                    },
                    children=[
                        html.Div(
                            style={
                                "position": "absolute",
                                "top": 0,
                                "left": 0,
                                "right": 0,
                                "bottom": 0,
                                "background": f"linear-gradient(90deg, {color}08 0%, transparent 100%)",
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
                                                        html.Span(icon, style={"fontSize": "24px", "marginRight": "12px"}),
                                                        html.Span(
                                                            manufacturer,
                                                            style={"fontWeight": 900, "fontSize": "18px", "color": color},
                                                        ),
                                                    ],
                                                    style={"marginBottom": "12px"},
                                                ),
                                                html.Div(
                                                    [
                                                        html.Span(
                                                            "Depreciation Trend: ",
                                                            style={"color": "#9CA3AF", "fontSize": "13px", "fontWeight": 600},
                                                        ),
                                                        html.Span(
                                                            sentiment,
                                                            style={"color": color, "fontSize": "13px", "fontWeight": 700},
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
                                                            style={
                                                                "fontSize": "28px",
                                                                "fontWeight": 900,
                                                                "color": color,
                                                                "textAlign": "right",
                                                                "marginBottom": "8px",
                                                            },
                                                        ),
                                                        html.Div(
                                                            style={
                                                                "height": "12px",
                                                                "background": "rgba(75, 85, 99, 0.3)",
                                                                "borderRadius": "6px",
                                                                "overflow": "hidden",
                                                            },
                                                            children=[
                                                                html.Div(
                                                                    style={
                                                                        "width": f"{min(dep_pct, 100)}%",
                                                                        "height": "100%",
                                                                        "background": f"linear-gradient(90deg, {color}, {color}AA)",
                                                                        "borderRadius": "6px",
                                                                        "transition": "width 0.6s ease",
                                                                        "boxShadow": f"0 0 12px {color}80",
                                                                    }
                                                                )
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    "Value Loss Rate",
                                                    style={
                                                        "fontSize": "11px",
                                                        "color": "#6B7280",
                                                        "textAlign": "right",
                                                        "textTransform": "uppercase",
                                                        "letterSpacing": "0.5px",
                                                        "fontWeight": 600,
                                                    },
                                                ),
                                            ],
                                            md=6,
                                        ),
                                    ]
                                ),
                                html.Hr(style={"margin": "16px 0", "opacity": 0.2}),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            html.Div(
                                                [
                                                    html.Div(
                                                        "Initial Price",
                                                        style={
                                                            "fontSize": "11px",
                                                            "color": "#9CA3AF",
                                                            "textTransform": "uppercase",
                                                            "marginBottom": "4px",
                                                            "fontWeight": 600,
                                                        },
                                                    ),
                                                    html.Div(
                                                        f"₪{data_['first_price']:,.0f}",
                                                        style={"fontSize": "16px", "fontWeight": 800, "color": "#F9FAFB"},
                                                    ),
                                                ]
                                            ),
                                            md=4,
                                        ),
                                        dbc.Col(
                                            html.Div(
                                                [
                                                    html.Div(
                                                        "Final Price",
                                                        style={
                                                            "fontSize": "11px",
                                                            "color": "#9CA3AF",
                                                            "textTransform": "uppercase",
                                                            "marginBottom": "4px",
                                                            "fontWeight": 600,
                                                        },
                                                    ),
                                                    html.Div(
                                                        f"₪{data_['last_price']:,.0f}",
                                                        style={"fontSize": "16px", "fontWeight": 800, "color": "#F9FAFB"},
                                                    ),
                                                ]
                                            ),
                                            md=4,
                                        ),
                                        dbc.Col(
                                            html.Div(
                                                [
                                                    html.Div(
                                                        "Value Lost",
                                                        style={
                                                            "fontSize": "11px",
                                                            "color": "#9CA3AF",
                                                            "textTransform": "uppercase",
                                                            "marginBottom": "4px",
                                                            "fontWeight": 600,
                                                        },
                                                    ),
                                                    html.Div(
                                                        f"₪{data_['first_price'] - data_['last_price']:,.0f}",
                                                        style={"fontSize": "16px", "fontWeight": 800, "color": color},
                                                    ),
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
            style={"padding": "24px"},
            children=[
                html.Div(
                    [
                        html.Div(
                            style={"display": "flex", "alignItems": "center", "gap": "12px", "marginBottom": "8px"},
                            children=[
                                html.Div(
                                    "Depreciation Analysis",
                                    style={
                                        "fontSize": "20px",
                                        "fontWeight": 900,
                                        "background": "linear-gradient(135deg, #60A5FA, #8B5CF6)",
                                        "WebkitBackgroundClip": "text",
                                        "WebkitTextFillColor": "transparent",
                                    },
                                ),
                                html.Div(
                                    id="calc-info-button",
                                    children="ℹ️",
                                    style={
                                        "width": "56px",
                                        "height": "56px",
                                        "borderRadius": "50%",
                                        "background": "rgba(59, 130, 246, 0.2)",
                                        "border": "1px solid rgba(59, 130, 246, 0.5)",
                                        "display": "flex",
                                        "alignItems": "center",
                                        "justifyContent": "center",
                                        "cursor": "pointer",
                                        "fontSize": "20px",
                                        "transition": "all 0.3s ease",
                                    },
                                    title="Click to see calculation methodology",
                                ),
                            ],
                        ),
                        html.Div(
                            "Value retention trends across selected models",
                            style={"fontSize": "14px", "color": "#9CA3AF", "marginBottom": "20px"},
                        ),
                        dbc.Collapse(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H6(
                                            "📊 Calculation Methodology",
                                            style={"fontWeight": 800, "marginBottom": "12px", "color": "#60A5FA"},
                                        ),
                                        html.P(
                                            ["The depreciation percentage is calculated using a robust statistical approach:"],
                                            style={"marginBottom": "12px", "fontSize": "13px"},
                                        ),
                                        html.Ol(
                                            [
                                                html.Li(
                                                    [
                                                        html.Strong("Data Sorting: "),
                                                        "All vehicles of the selected model are sorted by mileage (low to high).",
                                                    ],
                                                    style={"marginBottom": "8px", "fontSize": "13px"},
                                                ),
                                                html.Li(
                                                    [
                                                        html.Strong("Group Selection: "),
                                                        "The bottom 20% (lowest mileage, minimum 3 vehicles) and top 20% (highest mileage, minimum 3 vehicles) are selected.",
                                                    ],
                                                    style={"marginBottom": "8px", "fontSize": "13px"},
                                                ),
                                                html.Li(
                                                    [
                                                        html.Strong("Average Calculation: "),
                                                        "The average price is computed for each group separately.",
                                                    ],
                                                    style={"marginBottom": "8px", "fontSize": "13px"},
                                                ),
                                                html.Li(
                                                    [
                                                        html.Strong("Depreciation Formula: "),
                                                        html.Code(
                                                            "Depreciation % = ((Low Mileage Avg Price - High Mileage Avg Price) / Low Mileage Avg Price) × 100",
                                                            style={
                                                                "background": "rgba(139, 92, 246, 0.2)",
                                                                "padding": "2px 6px",
                                                                "borderRadius": "4px",
                                                                "fontSize": "11px",
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
                                                html.Strong("💡 Why this method? ", style={"color": "#10B981"}),
                                                "Using percentiles (20%) instead of absolute min/max values reduces the impact of outliers and provides a more reliable depreciation estimate.",
                                            ],
                                            style={
                                                "background": "rgba(16, 185, 129, 0.1)",
                                                "padding": "12px",
                                                "borderRadius": "8px",
                                                "border": "1px solid rgba(16, 185, 129, 0.3)",
                                                "fontSize": "13px",
                                                "marginTop": "8px",
                                            },
                                        ),
                                    ]
                                ),
                                style={
                                    "background": "rgba(17, 24, 39, 0.8)",
                                    "border": "1px solid rgba(75, 85, 99, 0.3)",
                                    "marginBottom": "20px",
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
        if "Price per KM" in metric_name or "Price Stability" in metric_name:
            # Lower is better
            return "A" if val_a < val_b else "B" if val_b < val_a else "Tie"
        else:
            # For Avg Mileage and Avg Price, don't highlight advantage (context-dependent)
            return "Tie"
    
    # Generate comparison insight
    price_per_km_a = metrics_data['Price per KM'][0]
    price_per_km_b = metrics_data['Price per KM'][1]
    stability_a = metrics_data['Price Stability'][0]
    stability_b = metrics_data['Price Stability'][1]
    
    insights = []
    if price_per_km_a < price_per_km_b:
        insights.append("better value per km")
    elif price_per_km_b < price_per_km_a:
        insights.append("better value per km")
    
    if stability_a < stability_b:
        insights.append("higher price stability")
    elif stability_b < stability_a:
        insights.append("higher price stability")
    
    if insights:
        insight_text = f"Group {'A' if price_per_km_a < price_per_km_b or stability_a < stability_b else 'B'} shows {', '.join(insights[:2])}."
    else:
        insight_text = "Both groups show similar value characteristics."
    
    # Compact KPI strip for Group A/B
    group_summary = dbc.Row(
        className="g-3 mb-4",
        children=[
            dbc.Col(
                html.Div(
                    [
                        html.Div("A", className="group-label-badge", style={"background": "rgba(91, 155, 213, 0.2)", "color": COLORS["blue"]}),
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
                        html.Div("B", className="group-label-badge", style={"background": "rgba(139, 92, 246, 0.2)", "color": COLORS["purple"]}),
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
    
    # Comparison insight line - Executive note style
    insight_section = html.Div(
        [
            html.Div(
                insight_text,
                className="comparison-insight",
            ),
        ],
        className="comparison-insight-container",
    )
    
    # Comparison table
    metric_rows = []
    metric_labels = {
        "Price per KM": "Price per KM",
        "Price Stability": "Price Stability (σ)",
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
        
        # Apply subtle emphasis to better value (typographic only)
        value_a_class = "comparison-metric-value"
        value_b_class = "comparison-metric-value"
        if advantage == "A":
            value_a_class = "comparison-metric-value comparison-metric-emphasized"
        elif advantage == "B":
            value_b_class = "comparison-metric-value comparison-metric-emphasized"
        
        metric_rows.append(
            html.Div(
                [
                    html.Div(metric_label, className="comparison-metric-label"),
                    html.Div(
                        formatted_a,
                        className=value_a_class,
                    ),
                    html.Div(
                        formatted_b,
                        className=value_b_class,
                    ),
                ],
                className="comparison-table-row",
            )
        )
    
    comparison_table = html.Div(
        [
            html.Div(
                [
                    html.Div("📊 Metric", className="comparison-table-header"),
                    html.Div("🔵 Group A", className="comparison-table-header comparison-table-header-right"),
                    html.Div("🟣 Group B", className="comparison-table-header comparison-table-header-right"),
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
    app.run(debug=True)
