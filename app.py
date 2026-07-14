import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime, timedelta
import os

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Supply Chain Optimax",
    page_icon="\u2693",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
    /* Main container */
    .stApp {
        font-family: 'Segoe UI', sans-serif;
    }
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1A1F2E 0%, #0E1117 100%);
        border: 1px solid #00B4D8;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 15px rgba(0, 180, 216, 0.15);
    }
    div[data-testid="stMetric"] label {
        color: #90E0EF !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #FAFAFA !important;
        font-weight: 700 !important;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0A1628 0%, #0E1117 100%);
        border-right: 1px solid #1E3A5F;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #CAF0F8 !important;
        font-size: 0.9rem;
    }
    /* Headers */
    h1, h2, h3 {
        color: #CAF0F8 !important;
    }
    h1 {
        border-bottom: 2px solid #00B4D8;
        padding-bottom: 8px;
    }
    /* Buttons */
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #0077B6 0%, #00B4D8 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    div[data-testid="stButton"] > button:hover {
        box-shadow: 0 4px 20px rgba(0, 180, 216, 0.4);
        transform: translateY(-1px);
    }
    /* Forms */
    div[data-testid="stForm"] {
        border: 1px solid #1E3A5F;
        border-radius: 12px;
        padding: 20px;
        background: rgba(26, 31, 46, 0.5);
    }
    /* Dataframe styling */
    div[data-testid="stDataFrame"] {
        border: 1px solid #1E3A5F;
        border-radius: 8px;
        overflow: hidden;
    }
    /* Success/Warning/Error boxes */
    div[data-testid="stAlert"] {
        border-radius: 8px;
        border-left-width: 4px;
    }
    /* Tabs */
    button[data-baseweb="tab"] {
        color: #90E0EF !important;
        font-weight: 500;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #00B4D8 !important;
        border-bottom-color: #00B4D8 !important;
    }
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: #5A6A7A;
        font-size: 0.8rem;
        border-top: 1px solid #1E3A5F;
        margin-top: 40px;
    }
    .footer a {
        color: #00B4D8;
        text-decoration: none;
    }
</style>
""", unsafe_allow_html=True)


# ================= DATABASE =================
@st.cache_resource
def get_db_connection():
    conn = sqlite3.connect("optimax.db", check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS container_movement(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE,
            teus INTEGER
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS transporters(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            cost INTEGER,
            reliability INTEGER,
            speed INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS routes(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_location TEXT,
            warehouse_location TEXT,
            distance_km REAL,
            cost_per_km REAL,
            total_cost REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS activity_log(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            user TEXT,
            action TEXT,
            details TEXT
        )
    """)
    conn.commit()
    return conn


def log_activity(conn, user, action, details=""):
    try:
        conn.execute(
            "INSERT INTO activity_log(timestamp, user, action, details) VALUES (?,?,?,?)",
            (datetime.now().isoformat(), user, action, details),
        )
        conn.commit()
    except Exception:
        pass


# ================= SIDEBAR =================
conn = get_db_connection()

with st.sidebar:
    st.markdown("### \u2693 Supply Chain Optimax")
    st.markdown("---")

    menu = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Upload Data",
            "Demand Forecast",
            "Inventory & Dwell KPIs",
            "Route Costing",
            "Transporter Rating",
            "Data Management",
            "System Health",
        ],
    )


# ================= HELPER: CSV DOWNLOAD =================
def download_csv(df, filename):
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


# ================= DASHBOARD =================
if menu == "Dashboard":
    st.title("\ud83d\udcca Port Operations Dashboard")

    df = pd.read_sql("SELECT * FROM container_movement ORDER BY date", conn)

    if df.empty:
        st.info("\ud83d\udce5 No container movement data yet. Upload data to get started.")
    else:
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.to_period("M").astype(str)
        df["weekday"] = df["date"].dt.day_name()

        # --- KPI Row ---
        c1, c2, c3, c4, c5 = st.columns(5)
        total_teus = int(df["teus"].sum())
        avg_daily = df["teus"].mean()
        std_daily = df["teus"].std()
        max_day = df.loc[df["teus"].idxmax()]
        min_day = df.loc[df["teus"].idxmin()]

        c1.metric("\ud83d\udce6 Total TEUs", f"{total_teus:,}")
        c2.metric("\ud83d\udcc8 Avg Daily TEUs", f"{avg_daily:,.0f}")
        c3.metric("\ud83d\udcc9 Volatility (\u03c3)", f"{std_daily:,.1f}")
        c4.metric("\ud83d\udcc5 Peak Day", f"{max_day['teus']:,}", max_day["date"].strftime("%d %b"))
        c5.metric("\ud83d\udcc1 Lowest Day", f"{min_day['teus']:,}", min_day["date"].strftime("%d %b"))

        st.markdown("---")

        # --- Charts ---
        tab1, tab2, tab3 = st.tabs(["Trend", "Monthly", "Weekday"])

        with tab1:
            fig = px.area(
                df, x="date", y="teus",
                title="Container Throughput Over Time",
                color_discrete_sequence=["#00B4D8"],
            )
            fig.update_layout(
                template="plotly_dark",
                xaxis_title="Date", yaxis_title="TEUs",
                hovermode="x unified",
                margin=dict(l=0, r=0, t=40, b=0),
            )
            fig.update_traces(fillcolor="rgba(0,180,216,0.2)", line=dict(width=2))
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            monthly = df.groupby("month")["teus"].sum().reset_index()
            fig2 = px.bar(
                monthly, x="month", y="teus",
                title="Monthly TEU Volume",
                color="teus",
                color_continuous_scale="Teal",
            )
            fig2.update_layout(
                template="plotly_dark",
                xaxis_title="Month", yaxis_title="Total TEUs",
                margin=dict(l=0, r=0, t=40, b=0),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            weekday = df.groupby("weekday")["teus"].mean().reindex(day_order).reset_index()
            fig3 = px.bar(
                weekday, x="weekday", y="teus",
                title="Average TEUs by Weekday",
                color="teus",
                color_continuous_scale="Viridis",
            )
            fig3.update_layout(
                template="plotly_dark",
                xaxis_title="", yaxis_title="Avg TEUs",
                margin=dict(l=0, r=0, t=40, b=0),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig3, use_container_width=True)

        # --- Summary Table ---
        with st.expander("\ud83d\udcca Data Summary"):
            st.dataframe(df[["date", "teus"]].describe().round(1), use_container_width=True)
            download_csv(df, "container_data.csv")

    # --- Quick Stats from other tables ---
    st.markdown("---")
    st.subheader("\ud83d\udcca Quick Overview")
    qc1, qc2, qc3 = st.columns(3)

    route_count = pd.read_sql("SELECT COUNT(*) as cnt FROM routes", conn).iloc[0, 0]
    trans_count = pd.read_sql("SELECT COUNT(*) as cnt FROM transporters", conn).iloc[0, 0]
    data_count = len(df) if not df.empty else 0

    qc1.metric("\ud83d\ude9a Saved Routes", route_count)
    qc2.metric("\ud83c\udfed Transporters", trans_count)
    qc3.metric("\ud83d\udcc5 Data Points", data_count)


# ================= UPLOAD =================
elif menu == "Upload Data":
    st.title("\ud83d\udce5 Upload Container Throughput Data")

    tab_upload, tab_manual = st.tabs(["CSV Upload", "Manual Entry"])

    with tab_upload:
        with st.form("upload_form"):
            file = st.file_uploader("Upload CSV file with columns: `date`, `teus`", type=["csv"])
            upload_submitted = st.form_submit_button("Upload & Save", use_container_width=True)

            if upload_submitted and file:
                try:
                    df = pd.read_csv(file)
                    if "date" not in df.columns or "teus" not in df.columns:
                        st.error("CSV must have 'date' and 'teus' columns.")
                    else:
                        saved = 0
                        for _, row in df.iterrows():
                            try:
                                conn.execute(
                                    "INSERT OR IGNORE INTO container_movement(date, teus) VALUES (?, ?)",
                                    (str(row["date"]), int(row["teus"])),
                                )
                                saved += 1
                            except Exception:
                                continue
                        conn.commit()
                        st.toast(f"Saved {saved} records successfully!", icon="\u2705")
                        log_activity(conn, "user", "upload_csv", f"records={saved}")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")

        # Preview uploaded data
        existing = pd.read_sql("SELECT * FROM container_movement ORDER BY date DESC LIMIT 10", conn)
        if not existing.empty:
            st.subheader("\ud83d\udcc4 Recent Records")
            st.dataframe(existing, use_container_width=True)

    with tab_manual:
        with st.form("manual_entry"):
            col_a, col_b = st.columns(2)
            with col_a:
                entry_date = st.date_input("Date", value=datetime.now().date())
            with col_b:
                entry_teus = st.number_input("TEUs", min_value=0, value=0)
            manual_submitted = st.form_submit_button("Save Entry", use_container_width=True)

            if manual_submitted:
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO container_movement(date, teus) VALUES (?, ?)",
                        (entry_date.isoformat(), int(entry_teus)),
                    )
                    conn.commit()
                    st.toast(f"Entry saved for {entry_date}", icon="\u2705")
                    log_activity(conn, "user", "manual_entry", f"date={entry_date}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error saving: {e}")


# ================= DEMAND FORECAST =================
elif menu == "Demand Forecast":
    st.title("\ud83d\udcc8 Container Demand Forecast")

    df = pd.read_sql("SELECT * FROM container_movement ORDER BY date", conn)

    if df.empty or len(df) < 5:
        st.warning("\ud83d\udce5 Upload at least 5 data points to enable forecasting.")
    else:
        df["date"] = pd.to_datetime(df["date"])
        df["day"] = df["date"].dt.day
        df["month"] = df["date"].dt.month
        df["dayofweek"] = df["date"].dt.dayofweek
        df["dayofyear"] = df["date"].dt.dayofyear

        with st.spinner("\ud83e\uddee Training models..."):
            features = ["day", "month", "dayofweek", "dayofyear"]
            X = df[features]
            y = df["teus"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            rf = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10)
            lr = LinearRegression()

            rf.fit(X_train, y_train)
            lr.fit(X_train, y_train)

            rf_pred = rf.predict(X_test)
            lr_pred = lr.predict(X_test)

            rf_mae = mean_absolute_error(y_test, rf_pred)
            lr_mae = mean_absolute_error(y_test, lr_pred)
            rf_r2 = r2_score(y_test, rf_pred)
            lr_r2 = r2_score(y_test, lr_pred)

            best_model = rf if rf_mae < lr_mae else lr
            best_name = "Random Forest" if best_model == rf else "Linear Regression"

        # --- Model Comparison ---
        st.subheader("\ud83e\uddee Model Performance")
        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown("**Random Forest**")
            st.metric("MAE", f"{rf_mae:.1f}")
            st.metric("R\u00b2 Score", f"{rf_r2:.3f}")
        with mc2:
            st.markdown("**Linear Regression**")
            st.metric("MAE", f"{lr_mae:.1f}")
            st.metric("R\u00b2 Score", f"{lr_r2:.3f}")

        st.success(f"\ud83c\udfc6 Best Model: **{best_name}** (MAE: {min(rf_mae, lr_mae):.1f})")

        st.markdown("---")

        # --- Forecast ---
        st.subheader("\ud83d\udd2e Forecast")
        days = st.slider("Forecast Horizon (days)", 7, 90, 30)
        last_date = df["date"].max()

        future = []
        for i in range(1, days + 1):
            d = last_date + timedelta(days=i)
            pred = best_model.predict([[d.day, d.month, d.dayofweek(), d.timetuple().tm_yday]])[0]
            future.append({"Date": d, "Forecast TEUs": max(0, int(pred))})

        fdf = pd.DataFrame(future)

        # Confidence band (simple std-based)
        residual_std = y_test - best_model.predict(X_test)
        std_val = residual_std.std()

        fdf["Lower Bound"] = (fdf["Forecast TEUs"] - 1.96 * std_val).clip(lower=0).astype(int)
        fdf["Upper Bound"] = (fdf["Forecast TEUs"] + 1.96 * std_val).astype(int)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["teus"],
            mode="lines+markers", name="Historical",
            line=dict(color="#00B4D8", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=fdf["Date"], y=fdf["Upper Bound"],
            mode="lines", name="Upper Bound",
            line=dict(width=0), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=fdf["Date"], y=fdf["Lower Bound"],
            fill="tonexty", mode="lines", name="95% CI",
            line=dict(width=0), fillcolor="rgba(0,180,216,0.15)",
        ))
        fig.add_trace(go.Scatter(
            x=fdf["Date"], y=fdf["Forecast TEUs"],
            mode="lines+markers", name="Forecast",
            line=dict(color="#F4A261", width=2, dash="dash"),
        ))
        fig.update_layout(
            template="plotly_dark", title="Historical + Forecast",
            xaxis_title="Date", yaxis_title="TEUs",
            hovermode="x unified", legend=dict(orientation="h", y=1.1),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("\ud83d\udcca Forecast Data"):
            st.dataframe(fdf, use_container_width=True)
            download_csv(fdf, "forecast_data.csv")


# ================= INVENTORY & DWELL KPIs =================
elif menu == "Inventory & Dwell KPIs":
    st.title("\ud83d\udce6 Inventory & Dwell Time KPIs")

    tab_calc, tab_sim = st.tabs(["KPI Calculator", "What-If Simulation"])

    with tab_calc:
        with st.form("kpi_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_inventory = st.number_input("Avg Containers in Yard", min_value=1, value=500)
            with col2:
                annual_teus = st.number_input("Annual TEUs", min_value=1, value=50000)
            with col3:
                dwell_time = st.number_input("Avg Dwell Time (days)", min_value=1, value=5)

            kpi_submitted = st.form_submit_button("Calculate KPIs", use_container_width=True)

        if kpi_submitted:
            doi = avg_inventory / (annual_teus / 365)
            daily_throughput = annual_teus / 365
            yard_utilization = avg_inventory / (daily_throughput * dwell_time) * 100 if daily_throughput * dwell_time > 0 else 0

            k1, k2, k3 = st.columns(3)
            k1.metric("\ud83d\udcc5 Days of Inventory", f"{doi:.1f} days")
            k2.metric("\ud83d\udcc8 Daily Throughput", f"{daily_throughput:,.0f}")
            k3.metric("\ud83c\udfed Yard Utilization", f"{yard_utilization:.1f}%")

            if dwell_time > 6:
                st.error("\u26a0\ufe0f High dwell time detected \u2014 congestion risk! Consider expediting clearance.")
            elif dwell_time > 4:
                st.warning("\ud83d\udcc1 Moderate dwell time \u2014 monitor closely for trends.")
            else:
                st.success("\u2705 Dwell time is under control. Operations running smoothly.")

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=dwell_time,
                title={"text": "Dwell Time (days)"},
                delta={"reference": 5, "increasing": {"color": "#E63946"}, "decreasing": {"color": "#2A9D8F"}},
                gauge={
                    "axis": {"range": [0, 10]},
                    "bar": {"color": "#00B4D8"},
                    "steps": [
                        {"range": [0, 4], "color": "rgba(42,157,143,0.3)"},
                        {"range": [4, 6], "color": "rgba(244,162,97,0.3)"},
                        {"range": [6, 10], "color": "rgba(230,57,70,0.3)"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.8,
                        "value": 5,
                    },
                },
            ))
            fig.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

    with tab_sim:
        st.markdown("### \ud83e\uddee What-If Scenario Analysis")
        with st.form("sim_form"):
            sim_inv = st.slider("Simulated Avg Inventory", 100, 5000, 500, step=50)
            sim_teus = st.slider("Simulated Annual TEUs", 10000, 200000, 50000, step=5000)
            sim_dwell = st.slider("Simulated Dwell Time (days)", 1, 15, 5)
            sim_submitted = st.form_submit_button("Run Simulation", use_container_width=True)

        if sim_submitted:
            scenarios = []
            for inv in range(100, sim_inv + 1, 100):
                doi = inv / (sim_teus / 365)
                scenarios.append({"Inventory": inv, "DOI": doi, "Status": "High" if doi > 6 else "Moderate" if doi > 4 else "Optimal"})

            sdf = pd.DataFrame(scenarios)
            fig = px.line(
                sdf, x="Inventory", y="DOI", color="Status",
                title="DOI vs Inventory Level",
                color_discrete_map={"Optimal": "#2A9D8F", "Moderate": "#F4A261", "High": "#E63946"},
            )
            fig.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)


# ================= ROUTE COSTING =================
elif menu == "Route Costing":
    st.title("\ud83d\ude9a Customer to Warehouse Route Costing")

    tab_new, tab_analytics, tab_compare = st.tabs(["New Route", "Analytics", "Compare Routes"])

    with tab_new:
        with st.form("route_form"):
            col1, col2 = st.columns(2)
            with col1:
                customer_location = st.text_input("Customer Location", placeholder="e.g., Mumbai Port")
            with col2:
                warehouse_location = st.text_input("Warehouse Location", placeholder="e.g., Pune DC")

            col3, col4 = st.columns(2)
            with col3:
                distance_km = st.number_input("Distance (km)", min_value=0.1, value=100.0, step=1.0)
            with col4:
                cost_per_km = st.number_input("Cost per km", min_value=0.1, value=60.0, step=1.0)

            route_submitted = st.form_submit_button("Calculate & Save Route", use_container_width=True)

        if route_submitted:
            total_cost = distance_km * cost_per_km
            c1, c2 = st.columns(2)
            c1.metric("\ud83d\udcb0 Total Transport Cost", f"\u20b9 {total_cost:,.2f}")
            c2.metric("\ud83d\udee1\ufe0f Cost per km", f"\u20b9 {cost_per_km:,.2f}")

            conn.execute(
                "INSERT INTO routes(customer_location, warehouse_location, distance_km, cost_per_km, total_cost) VALUES (?,?,?,?,?)",
                (customer_location, warehouse_location, distance_km, cost_per_km, total_cost),
            )
            conn.commit()
            st.toast("Route saved successfully!", icon="\u2705")
            log_activity(conn, "user", "add_route", f"{customer_location} -> {warehouse_location}")
            st.rerun()

    with tab_analytics:
        rdf = pd.read_sql("SELECT * FROM routes ORDER BY created_at DESC", conn)
        if rdf.empty:
            st.info("No routes saved yet.")
        else:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("\ud83d\ude9a Total Routes", len(rdf))
            k2.metric("\ud83d\udcb0 Avg Cost", f"\u20b9 {rdf['total_cost'].mean():,.0f}")
            k3.metric("\ud83c\udfcf Max Distance", f"{rdf['distance_km'].max():.0f} km")
            k4.metric("\ud83d\udcc1 Min Cost", f"\u20b9 {rdf['total_cost'].min():,.0f}")

            fig = px.bar(
                rdf, x=rdf.index, y="total_cost",
                hover_data=["customer_location", "warehouse_location", "distance_km"],
                title="Route Cost Comparison",
                color="total_cost",
                color_continuous_scale="Teal",
            )
            fig.update_layout(template="plotly_dark", xaxis_title="Route", yaxis_title="Total Cost", margin=dict(l=0, r=0, t=40, b=0), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("\ud83d\udcca All Saved Routes"):
                st.dataframe(rdf, use_container_width=True)
                download_csv(rdf, "routes_data.csv")

    with tab_compare:
        rdf = pd.read_sql("SELECT * FROM routes", conn)
        if len(rdf) < 2:
            st.info("Add at least 2 routes to compare.")
        else:
            st.markdown("### \ud83d\udd0d Route Comparison")
            options = st.multiselect(
                "Select routes to compare",
                options=rdf.index.tolist(),
                default=rdf.index[:3].tolist() if len(rdf) >= 3 else rdf.index.tolist(),
                format_func=lambda x: f"{rdf.loc[x, 'customer_location']} -> {rdf.loc[x, 'warehouse_location']}",
            )
            if options:
                cmp = rdf.loc[options]
                fig = make_subplots(
                    rows=1, cols=3,
                    subplot_titles=("Distance (km)", "Cost/km", "Total Cost"),
                )
                fig.add_trace(go.Bar(x=[f"R{i}" for i in cmp.index], y=cmp["distance_km"], name="Distance", marker_color="#00B4D8"), row=1, col=1)
                fig.add_trace(go.Bar(x=[f"R{i}" for i in cmp.index], y=cmp["cost_per_km"], name="Cost/km", marker_color="#F4A261"), row=1, col=2)
                fig.add_trace(go.Bar(x=[f"R{i}" for i in cmp.index], y=cmp["total_cost"], name="Total Cost", marker_color="#E63946"), row=1, col=3)
                fig.update_layout(template="plotly_dark", height=350, showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig, use_container_width=True)


# ================= TRANSPORTER RATING =================
elif menu == "Transporter Rating":
    st.title("\ud83c\udfed Transporter Evaluation")

    tab_add, tab_rank, tab_radar = st.tabs(["Add Transporter", "Leaderboard", "Comparison"])

    with tab_add:
        with st.form("transporter_form"):
            name = st.text_input("Transporter Name", placeholder="e.g., ABC Logistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                cost = st.slider("Cost Efficiency", 1, 10, 5)
            with col2:
                reliability = st.slider("Reliability", 1, 10, 5)
            with col3:
                speed = st.slider("Delivery Speed", 1, 10, 5)

            t_submitted = st.form_submit_button("Save Transporter", use_container_width=True)

        if t_submitted:
            if not name.strip():
                st.error("Please enter a transporter name.")
            else:
                score = (cost + reliability + speed) / 3
                conn.execute(
                    "INSERT INTO transporters(name, cost, reliability, speed) VALUES (?,?,?,?)",
                    (name.strip(), cost, reliability, speed),
                )
                conn.commit()
                st.toast(f"Transporter '{name}' saved! Score: {score:.1f}/10", icon="\u2705")
                log_activity(conn, "user", "add_transporter", name)
                st.rerun()

    with tab_rank:
        tdf = pd.read_sql("SELECT * FROM transporters", conn)
        if tdf.empty:
            st.info("No transporters added yet.")
        else:
            tdf["Score"] = tdf[["cost", "reliability", "speed"]].mean(axis=1).round(1)
            tdf = tdf.sort_values("Score", ascending=False).reset_index(drop=True)

            # Leaderboard
            st.subheader("\ud83c\udfc6 Transporter Leaderboard")
            for i, row in tdf.iterrows():
                medal = "\ud83e\udd47" if i == 0 else "\ud83e\udd48" if i == 1 else "\ud83e\udd49" if i == 2 else f"#{i+1}"
                cols = st.columns([1, 3, 1, 1, 1, 1])
                cols[0].markdown(f"### {medal}")
                cols[1].markdown(f"**{row['name']}**")
                cols[2].metric("Cost", row["cost"])
                cols[3].metric("Reliability", row["reliability"])
                cols[4].metric("Speed", row["speed"])
                cols[5].metric("Score", row["Score"])

            if len(tdf) > 0:
                st.success(f"\u2705 Recommended Transporter: **{tdf.iloc[0]['name']}** (Score: {tdf.iloc[0]['Score']})")

            with st.expander("\ud83d\udcca Full Data"):
                st.dataframe(tdf, use_container_width=True)
                download_csv(tdf, "transporters.csv")

    with tab_radar:
        tdf = pd.read_sql("SELECT * FROM transporters", conn)
        if len(tdf) < 2:
            st.info("Add at least 2 transporters for comparison.")
        else:
            st.subheader("\ud83e\udde0 Radar Comparison")
            selected = st.multiselect(
                "Select transporters",
                options=tdf["name"].tolist(),
                default=tdf["name"].head(3).tolist(),
            )
            if selected:
                fig = go.Figure()
                for name in selected:
                    row = tdf[tdf["name"] == name].iloc[0]
                    fig.add_trace(go.Scatterpolar(
                        r=[row["cost"], row["reliability"], row["speed"], row["cost"]],
                        theta=["Cost Efficiency", "Reliability", "Speed", "Cost Efficiency"],
                        fill="toself",
                        name=name,
                    ))
                fig.update_layout(
                    template="plotly_dark",
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 10]),
                        bgcolor="rgba(0,0,0,0)",
                    ),
                    height=450,
                    margin=dict(l=60, r=60, t=40, b=40),
                )
                st.plotly_chart(fig, use_container_width=True)


# ================= DATA MANAGEMENT =================
elif menu == "Data Management":
    st.title("\ud83d\udd11\ufe0f Data Management")

    tab_cm, tab_tr, tab_rt, tab_log = st.tabs([
        "Container Data",
        "Transporters",
        "Routes",
        "Activity Log",
    ])

    with tab_cm:
        df = pd.read_sql("SELECT * FROM container_movement ORDER BY date DESC", conn)
        if df.empty:
            st.info("No container data.")
        else:
            st.dataframe(df, use_container_width=True)
            del_id = st.number_input("Enter ID to delete", min_value=1, key="del_cm")
            if st.button("Delete Record", key="btn_del_cm"):
                conn.execute("DELETE FROM container_movement WHERE id = ?", (int(del_id),))
                conn.commit()
                st.toast("Record deleted", icon="\ud83d\udeae")
                st.rerun()
            if st.button("Clear All Container Data", key="clear_cm"):
                conn.execute("DELETE FROM container_movement")
                conn.commit()
                st.toast("All container data cleared", icon="\ud83d\udeae")
                st.rerun()

    with tab_tr:
        tdf = pd.read_sql("SELECT * FROM transporters", conn)
        if tdf.empty:
            st.info("No transporters.")
        else:
            st.dataframe(tdf, use_container_width=True)
            del_id = st.number_input("Enter ID to delete", min_value=1, key="del_tr")
            if st.button("Delete Transporter", key="btn_del_tr"):
                conn.execute("DELETE FROM transporters WHERE id = ?", (int(del_id),))
                conn.commit()
                st.toast("Transporter deleted", icon="\ud83d\udeae")
                st.rerun()

    with tab_rt:
        rdf = pd.read_sql("SELECT * FROM routes", conn)
        if rdf.empty:
            st.info("No routes.")
        else:
            st.dataframe(rdf, use_container_width=True)
            del_id = st.number_input("Enter ID to delete", min_value=1, key="del_rt")
            if st.button("Delete Route", key="btn_del_rt"):
                conn.execute("DELETE FROM routes WHERE id = ?", (int(del_id),))
                conn.commit()
                st.toast("Route deleted", icon="\ud83d\udeae")
                st.rerun()

    with tab_log:
        log_df = pd.read_sql("SELECT * FROM activity_log ORDER BY timestamp DESC LIMIT 50", conn)
        if log_df.empty:
            st.info("No activity recorded yet.")
        else:
            st.dataframe(log_df, use_container_width=True)
            download_csv(log_df, "activity_log.csv")


# ================= SYSTEM HEALTH =================
elif menu == "System Health":
    st.title("\ud83d\udccb System Health")

    col1, col2, col3, col4 = st.columns(4)

    cm_count = pd.read_sql("SELECT COUNT(*) as c FROM container_movement", conn).iloc[0, 0]
    tr_count = pd.read_sql("SELECT COUNT(*) as c FROM transporters", conn).iloc[0, 0]
    rt_count = pd.read_sql("SELECT COUNT(*) as c FROM routes", conn).iloc[0, 0]
    log_count = pd.read_sql("SELECT COUNT(*) as c FROM activity_log", conn).iloc[0, 0]

    col1.metric("\ud83d\udce6 Container Records", cm_count)
    col2.metric("\ud83c\udfed Transporters", tr_count)
    col3.metric("\ud83d\ude9a Routes", rt_count)
    col4.metric("\ud83d\udccb Activity Logs", log_count)

    st.markdown("---")

    # DB file size
    db_size = os.path.getsize("optimax.db") / 1024 if os.path.exists("optimax.db") else 0
    st.metric("\ud83d\udcbe Database Size", f"{db_size:.1f} KB")

    # Recent activity
    st.subheader("\ud83d\udd50 Recent Activity")
    recent = pd.read_sql("SELECT * FROM activity_log ORDER BY timestamp DESC LIMIT 10", conn)
    if recent.empty:
        st.info("No activity yet.")
    else:
        st.dataframe(recent, use_container_width=True)

    # Data range
    date_range = pd.read_sql("SELECT MIN(date) as min_d, MAX(date) as max_d FROM container_movement", conn)
    if date_range.iloc[0]["min_d"]:
        st.info(f"\ud83d\udcc5 Data range: {date_range.iloc[0]['min_d']} to {date_range.iloc[0]['max_d']}")


# ================= FOOTER =================
st.markdown("---")
st.markdown(
    '<div class="footer">'
    "Supply Chain Optimax v2.0 | Developed by Nayan | "
    '<a href="https://github.com/mkshariharan07-hub/Supply-Chain-Optimax" target="_blank">GitHub</a>'
    "</div>",
    unsafe_allow_html=True,
)
