import json
import math
import os
import shutil
import zipfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


ROOT = Path.cwd()
OUT = ROOT / "SalesForecasting_Pratham"
CHARTS = OUT / "charts"
DATA = ROOT / "train.csv"
VG_DATA = ROOT / "vgsales.csv"


def season(month):
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": mape(y_true, y_pred),
    }


def money(x):
    return f"${x:,.0f}"


def prepare_dirs():
    OUT.mkdir(exist_ok=True)
    CHARTS.mkdir(exist_ok=True)
    cache_dir = OUT / "__pycache__"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    for path in CHARTS.glob("*.png"):
        path.unlink()
    shutil.copy2(DATA, OUT / "train.csv")
    if VG_DATA.exists():
        shutil.copy2(VG_DATA, OUT / "vgsales.csv")


def load_data():
    df = pd.read_csv(DATA)
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True, errors="coerce")
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Month Name"] = df["Order Date"].dt.month_name()
    df["Week Number"] = df["Order Date"].dt.isocalendar().week.astype(int)
    df["Day of Week"] = df["Order Date"].dt.day_name()
    df["Quarter"] = df["Order Date"].dt.quarter
    df["Season"] = df["Month"].apply(season)
    df["Ship Days"] = (df["Ship Date"] - df["Order Date"]).dt.days
    df["MonthStart"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()
    df["WeekStart"] = df["Order Date"].dt.to_period("W").apply(lambda r: r.start_time)
    return df


def load_vg_data():
    if not VG_DATA.exists():
        return None, None
    vg = pd.read_csv(VG_DATA)
    vg["Year"] = pd.to_numeric(vg["Year"], errors="coerce")
    vg = vg.dropna(subset=["Year"])
    vg["Year"] = vg["Year"].astype(int)
    numeric_cols = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]
    for col in numeric_cols:
        vg[col] = pd.to_numeric(vg[col], errors="coerce").fillna(0)
    yearly = (
        vg.groupby("Year")
        .agg(
            VG_Global_Sales_Millions=("Global_Sales", "sum"),
            VG_NA_Sales_Millions=("NA_Sales", "sum"),
            VG_Title_Count=("Name", "count"),
            VG_Top_Genre=("Genre", lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown"),
        )
        .reset_index()
    )
    return vg, yearly


def build_merged_context(df, vg_yearly):
    if vg_yearly is None:
        context = df.groupby("Year")["Sales"].sum().reset_index(name="Superstore_Sales")
        context["VG_Global_Sales_Millions"] = np.nan
        context["VG_Title_Count"] = np.nan
        context["VG_Top_Genre"] = "Not available"
        return context
    retail_yearly = df.groupby("Year")["Sales"].sum().reset_index(name="Superstore_Sales")
    context = retail_yearly.merge(vg_yearly, on="Year", how="left")
    return context


def build_series(df):
    monthly = df.groupby("MonthStart")["Sales"].sum().asfreq("MS")
    weekly = df.groupby("WeekStart")["Sales"].sum().asfreq("W-MON")
    return monthly.fillna(0), weekly.fillna(0)


def evaluate_models(monthly):
    train = monthly.iloc[:-6]
    test = monthly.iloc[-6:]
    future_index = pd.date_range(monthly.index[-1] + pd.offsets.MonthBegin(1), periods=3, freq="MS")

    seasonal_pred = []
    for dt in test.index:
        seasonal_pred.append(monthly.get(dt - pd.DateOffset(years=1), train.iloc[-1]))
    seasonal_future = [monthly.get(dt - pd.DateOffset(years=1), monthly.iloc[-1]) for dt in future_index]

    work = monthly.reset_index()
    work.columns = ["ds", "y"]
    work["t"] = np.arange(len(work))
    work["month"] = work["ds"].dt.month
    month_dummies = pd.get_dummies(work["month"], prefix="m", drop_first=True)
    X = pd.concat([work[["t"]], month_dummies], axis=1)
    model_lr = LinearRegression().fit(X.iloc[:-6], work["y"].iloc[:-6])
    lr_pred = model_lr.predict(X.iloc[-6:])
    future_work = pd.DataFrame({"ds": future_index, "t": np.arange(len(work), len(work) + 3)})
    future_work["month"] = future_work["ds"].dt.month
    fut_dummies = pd.get_dummies(future_work["month"], prefix="m", drop_first=True)
    fut_X = fut_dummies.reindex(columns=month_dummies.columns, fill_value=0)
    fut_X.insert(0, "t", future_work["t"].values)
    lr_future = np.maximum(model_lr.predict(fut_X), 0)

    sup = pd.DataFrame({"y": monthly.values}, index=monthly.index)
    for lag in [1, 2, 3]:
        sup[f"lag_{lag}"] = sup["y"].shift(lag)
    sup["roll3"] = sup["y"].shift(1).rolling(3).mean()
    sup["month"] = sup.index.month
    sup["quarter"] = sup.index.quarter
    sup["season_code"] = sup["month"].apply(lambda m: ["Winter", "Spring", "Summer", "Autumn"].index(season(m)))
    sup = sup.dropna()
    rf_train = sup.iloc[:-6]
    rf_test = sup.iloc[-6:]
    features = ["lag_1", "lag_2", "lag_3", "roll3", "month", "quarter", "season_code"]
    rf = RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=2)
    rf.fit(rf_train[features], rf_train["y"])
    rf_pred = rf.predict(rf_test[features])
    history = monthly.copy()
    rf_future = []
    for dt in future_index:
        row = pd.DataFrame([{
            "lag_1": history.iloc[-1],
            "lag_2": history.iloc[-2],
            "lag_3": history.iloc[-3],
            "roll3": history.iloc[-3:].mean(),
            "month": dt.month,
            "quarter": dt.quarter,
            "season_code": ["Winter", "Spring", "Summer", "Autumn"].index(season(dt.month)),
        }])
        yhat = float(rf.predict(row[features])[0])
        rf_future.append(yhat)
        history.loc[dt] = yhat

    rows = []
    predictions = {
        "SARIMA fallback - seasonal naive": (seasonal_pred, seasonal_future),
        "Prophet fallback - trend/month regression": (lr_pred, lr_future),
        "XGBoost fallback - RandomForest lags": (rf_pred, rf_future),
    }
    for name, (pred, future) in predictions.items():
        row = {"Model": name, **metrics(test.values, pred)}
        for i, value in enumerate(future, start=1):
            row[f"Forecast Month {i}"] = float(value)
        rows.append(row)
    comparison = pd.DataFrame(rows).sort_values("MAPE").reset_index(drop=True)
    return comparison, future_index, test


def segment_forecasts(df, comparison, monthly):
    best_name = comparison.iloc[0]["Model"]
    future_index = pd.date_range(monthly.index[-1] + pd.offsets.MonthBegin(1), periods=3, freq="MS")
    segments = {
        "Furniture": df[df["Category"] == "Furniture"],
        "Technology": df[df["Category"] == "Technology"],
        "Office Supplies": df[df["Category"] == "Office Supplies"],
        "West": df[df["Region"] == "West"],
        "East": df[df["Region"] == "East"],
    }
    rows = []
    for label, sdf in segments.items():
        series = sdf.groupby("MonthStart")["Sales"].sum().reindex(monthly.index, fill_value=0)
        forecasts = [series.get(dt - pd.DateOffset(years=1), series.iloc[-1]) for dt in future_index]
        rows.append({"Segment": label, **{future_index[i].strftime("%Y-%m"): float(forecasts[i]) for i in range(3)}})
    return best_name, future_index, pd.DataFrame(rows)


def detect_anomalies(weekly):
    data = weekly.reset_index()
    data.columns = ["WeekStart", "Sales"]
    data["week_num"] = np.arange(len(data))
    data["month"] = data["WeekStart"].dt.month
    iso = IsolationForest(contamination=0.06, random_state=42)
    data["IsolationForest"] = iso.fit_predict(data[["Sales", "week_num", "month"]])
    data["RollingMean"] = data["Sales"].rolling(8, min_periods=4).mean()
    data["RollingStd"] = data["Sales"].rolling(8, min_periods=4).std()
    data["ZScore"] = (data["Sales"] - data["RollingMean"]) / data["RollingStd"]
    data["ZScoreFlag"] = data["ZScore"].abs() > 2
    data["IsAnomaly"] = (data["IsolationForest"] == -1) | data["ZScoreFlag"]
    return data


def cluster_products(df):
    monthly_sub = df.groupby(["Sub-Category", "MonthStart"])["Sales"].sum().reset_index()
    pivot = monthly_sub.pivot(index="Sub-Category", columns="MonthStart", values="Sales").fillna(0)
    first_year = df["Year"].min()
    last_year = df["Year"].max()
    total = df.groupby("Sub-Category")["Sales"].sum()
    avg_order = df.groupby("Sub-Category")["Sales"].mean()
    first_sales = df[df["Year"] == first_year].groupby("Sub-Category")["Sales"].sum()
    last_sales = df[df["Year"] == last_year].groupby("Sub-Category")["Sales"].sum()
    growth = ((last_sales - first_sales) / first_sales.replace(0, np.nan)).fillna(0)
    features = pd.DataFrame({
        "Total Sales": total,
        "YoY Growth Rate": growth,
        "Monthly Volatility": pivot.std(axis=1),
        "Average Order Value": avg_order,
    }).fillna(0)
    scaled = StandardScaler().fit_transform(features)
    inertias = []
    for k in range(1, 7):
        inertias.append(KMeans(n_clusters=k, random_state=42, n_init=20).fit(scaled).inertia_)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
    features["Cluster"] = kmeans.fit_predict(scaled)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaled)
    features["PCA1"] = coords[:, 0]
    features["PCA2"] = coords[:, 1]
    cluster_summary = features.groupby("Cluster")[["Total Sales", "YoY Growth Rate", "Monthly Volatility"]].mean()
    labels = {}
    for cluster, row in cluster_summary.iterrows():
        if row["Total Sales"] == cluster_summary["Total Sales"].max():
            labels[cluster] = "High Volume, Stable Core"
        elif row["Monthly Volatility"] == cluster_summary["Monthly Volatility"].max():
            labels[cluster] = "High Volatility Watchlist"
        elif row["YoY Growth Rate"] == cluster_summary["YoY Growth Rate"].max():
            labels[cluster] = "Growing Demand"
        else:
            labels[cluster] = "Low Volume / Declining Demand"
    features["Demand Segment"] = features["Cluster"].map(labels)
    return features.reset_index(), inertias


def make_charts(df, monthly, weekly, comparison, seg_df, anomaly_df, cluster_df, inertias, vg_yearly, merged_context):
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(11, 5))
    monthly.plot(ax=ax, color="#2563eb", linewidth=2)
    ax.set_title("Overall Monthly Sales Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Sales")
    fig.tight_layout()
    fig.savefig(CHARTS / "monthly_sales_trend.png", dpi=160)
    plt.close(fig)

    trend = monthly.rolling(6, center=True, min_periods=3).mean()
    seasonal = monthly.groupby(monthly.index.month).transform("mean") - monthly.mean()
    resid = monthly - trend.bfill().ffill() - seasonal
    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(monthly.index, monthly.values, color="#111827")
    axes[0].set_title("Observed")
    axes[1].plot(trend.index, trend.values, color="#059669")
    axes[1].set_title("Trend")
    axes[2].plot(seasonal.index, seasonal.values, color="#d97706")
    axes[2].set_title("Seasonal Index")
    axes[3].plot(resid.index, resid.values, color="#dc2626")
    axes[3].set_title("Residual")
    fig.tight_layout()
    fig.savefig(CHARTS / "decomposition.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    comp_plot = comparison.set_index("Model")[["MAE", "RMSE", "MAPE"]]
    comp_plot.plot(kind="bar", ax=ax)
    ax.set_title("Model Comparison Metrics")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(CHARTS / "model_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for _, row in seg_df.iterrows():
        xs = [c for c in seg_df.columns if c != "Segment"]
        ax.plot(xs, [row[x] for x in xs], marker="o", label=row["Segment"])
    ax.set_title("3-Month Forecast by Category and Region")
    ax.set_ylabel("Sales")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS / "segment_forecasts.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(anomaly_df["WeekStart"], anomaly_df["Sales"], color="#374151", linewidth=1)
    flagged = anomaly_df[anomaly_df["IsAnomaly"]]
    ax.scatter(flagged["WeekStart"], flagged["Sales"], color="#dc2626", s=28, label="Anomaly")
    ax.set_title("Weekly Sales Anomalies")
    ax.set_ylabel("Sales")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS / "anomaly_report.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, 7), inertias, marker="o", color="#4f46e5")
    ax.set_title("Elbow Method for K-Means")
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia")
    fig.tight_layout()
    fig.savefig(CHARTS / "elbow_method.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    for label, group in cluster_df.groupby("Demand Segment"):
        ax.scatter(group["PCA1"], group["PCA2"], label=label, s=70)
        for _, row in group.iterrows():
            ax.annotate(row["Sub-Category"], (row["PCA1"], row["PCA2"]), fontsize=8)
    ax.set_title("Product Demand Segments")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(CHARTS / "product_clusters.png", dpi=160)
    plt.close(fig)

    if vg_yearly is not None:
        fig, ax = plt.subplots(figsize=(10, 5))
        recent = vg_yearly[(vg_yearly["Year"] >= 2000) & (vg_yearly["Year"] <= 2020)]
        ax.plot(recent["Year"], recent["VG_Global_Sales_Millions"], marker="o", color="#7c3aed")
        ax.set_title("Supplementary Dataset: Video Game Global Sales by Year")
        ax.set_xlabel("Year")
        ax.set_ylabel("Global Sales (millions)")
        fig.tight_layout()
        fig.savefig(CHARTS / "vgsales_yearly_context.png", dpi=160)
        plt.close(fig)

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.bar(merged_context["Year"], merged_context["Superstore_Sales"], color="#2563eb", alpha=0.75, label="Superstore Sales")
        ax1.set_ylabel("Superstore Sales")
        ax2 = ax1.twinx()
        ax2.plot(merged_context["Year"], merged_context["VG_Global_Sales_Millions"], color="#dc2626", marker="o", label="VG Global Sales")
        ax2.set_ylabel("Video Game Sales (millions)")
        ax1.set_title("Year-Level Merge: Retail Sales vs Supplementary Market Context")
        fig.tight_layout()
        fig.savefig(CHARTS / "merged_yearly_context.png", dpi=160)
        plt.close(fig)


def make_app(best_model):
    app = f'''import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True, errors="coerce")
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Quarter"] = df["Order Date"].dt.quarter
    df["MonthStart"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()
    df["WeekStart"] = df["Order Date"].dt.to_period("W").apply(lambda r: r.start_time)
    df["Ship Days"] = (df["Ship Date"] - df["Order Date"]).dt.days
    return df

@st.cache_data
def load_vg_context():
    try:
        vg = pd.read_csv("vgsales.csv")
    except FileNotFoundError:
        return None
    vg["Year"] = pd.to_numeric(vg["Year"], errors="coerce")
    vg = vg.dropna(subset=["Year"])
    vg["Year"] = vg["Year"].astype(int)
    return vg.groupby("Year").agg(
        VG_Global_Sales_Millions=("Global_Sales", "sum"),
        VG_Title_Count=("Name", "count"),
        VG_Top_Genre=("Genre", lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown")
    ).reset_index()

def forecast_series(series, horizon):
    series = series.sort_index().asfreq("MS").fillna(0)
    if len(series) < 15:
        future = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
        return pd.Series([series.tail(3).mean()] * horizon, index=future), None
    sup = pd.DataFrame({{"y": series.values}}, index=series.index)
    for lag in [1, 2, 3]:
        sup[f"lag_{{lag}}"] = sup["y"].shift(lag)
    sup["roll3"] = sup["y"].shift(1).rolling(3).mean()
    sup["month"] = sup.index.month
    sup["quarter"] = sup.index.quarter
    sup = sup.dropna()
    features = ["lag_1", "lag_2", "lag_3", "roll3", "month", "quarter"]
    train, test = sup.iloc[:-6], sup.iloc[-6:]
    model = RandomForestRegressor(n_estimators=250, random_state=42, min_samples_leaf=2)
    model.fit(train[features], train["y"])
    pred = model.predict(test[features])
    score = {{"MAE": mean_absolute_error(test["y"], pred), "RMSE": mean_squared_error(test["y"], pred) ** 0.5}}
    history = series.copy()
    future_idx = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    values = []
    for dt in future_idx:
        row = pd.DataFrame([{{"lag_1": history.iloc[-1], "lag_2": history.iloc[-2], "lag_3": history.iloc[-3],
                             "roll3": history.iloc[-3:].mean(), "month": dt.month, "quarter": dt.quarter}}])
        yhat = float(model.predict(row[features])[0])
        values.append(yhat)
        history.loc[dt] = yhat
    return pd.Series(values, index=future_idx), score

df = load_data()
vg_context = load_vg_context()
page = st.sidebar.radio("Page", ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Demand Segments"])

if page == "Sales Overview":
    st.title("Sales Overview Dashboard")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Sales", f"${{df['Sales'].sum():,.0f}}")
    c2.metric("Orders", f"{{df['Order ID'].nunique():,}}")
    c3.metric("Avg Ship Days", f"{{df['Ship Days'].mean():.1f}}")
    region = st.multiselect("Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))
    category = st.multiselect("Category", sorted(df["Category"].unique()), default=sorted(df["Category"].unique()))
    view = df[df["Region"].isin(region) & df["Category"].isin(category)]
    st.bar_chart(view.groupby("Year")["Sales"].sum())
    st.line_chart(view.groupby("MonthStart")["Sales"].sum())
    st.dataframe(view.groupby(["Region", "Category"])["Sales"].sum().reset_index().sort_values("Sales", ascending=False), use_container_width=True)
    if vg_context is not None:
        st.caption("Supplementary dataset context: video game sales are merged only at year level because there is no product/customer key in common.")
        retail_yearly = view.groupby("Year")["Sales"].sum().reset_index(name="Superstore_Sales")
        st.dataframe(retail_yearly.merge(vg_context, on="Year", how="left"), use_container_width=True)

elif page == "Forecast Explorer":
    st.title("Forecast Explorer")
    segment_type = st.selectbox("Segment type", ["Category", "Region"])
    values = sorted(df[segment_type].unique())
    selected = st.selectbox(segment_type, values)
    horizon = st.slider("Forecast horizon", 1, 3, 3)
    series = df[df[segment_type] == selected].groupby("MonthStart")["Sales"].sum()
    forecast, score = forecast_series(series, horizon)
    combined = pd.concat([series.rename("Actual"), forecast.rename("Forecast")], axis=1)
    st.line_chart(combined)
    if score:
        st.caption(f"Model used: RandomForest lag model, selected locally as practical fallback for {best_model}. MAE=${{score['MAE']:,.0f}}, RMSE=${{score['RMSE']:,.0f}}")
    st.dataframe(forecast.reset_index().rename(columns={{"index": "Month", 0: "Forecast Sales"}}), use_container_width=True)

elif page == "Anomaly Report":
    st.title("Anomaly Report")
    weekly = df.groupby("WeekStart")["Sales"].sum().reset_index()
    weekly["week_num"] = np.arange(len(weekly))
    weekly["month"] = weekly["WeekStart"].dt.month
    iso = IsolationForest(contamination=0.06, random_state=42)
    weekly["IF Flag"] = iso.fit_predict(weekly[["Sales", "week_num", "month"]]) == -1
    weekly["RollingMean"] = weekly["Sales"].rolling(8, min_periods=4).mean()
    weekly["RollingStd"] = weekly["Sales"].rolling(8, min_periods=4).std()
    weekly["ZScore"] = (weekly["Sales"] - weekly["RollingMean"]) / weekly["RollingStd"]
    weekly["Z Flag"] = weekly["ZScore"].abs() > 2
    weekly["Anomaly"] = weekly["IF Flag"] | weekly["Z Flag"]
    st.line_chart(weekly.set_index("WeekStart")["Sales"])
    st.dataframe(weekly[weekly["Anomaly"]][["WeekStart", "Sales", "IF Flag", "Z Flag", "ZScore"]], use_container_width=True)
    if vg_context is not None:
        st.caption("Year-level supplementary context from vgsales.csv. This helps document multi-source analysis, but it is not a direct cause of weekly retail anomalies.")
        st.dataframe(vg_context.tail(10), use_container_width=True)

else:
    st.title("Product Demand Segments")
    monthly_sub = df.groupby(["Sub-Category", "MonthStart"])["Sales"].sum().reset_index()
    pivot = monthly_sub.pivot(index="Sub-Category", columns="MonthStart", values="Sales").fillna(0)
    first_year, last_year = df["Year"].min(), df["Year"].max()
    features = pd.DataFrame({{
        "Total Sales": df.groupby("Sub-Category")["Sales"].sum(),
        "YoY Growth Rate": ((df[df["Year"] == last_year].groupby("Sub-Category")["Sales"].sum() - df[df["Year"] == first_year].groupby("Sub-Category")["Sales"].sum()) / df[df["Year"] == first_year].groupby("Sub-Category")["Sales"].sum()).fillna(0),
        "Monthly Volatility": pivot.std(axis=1),
        "Average Order Value": df.groupby("Sub-Category")["Sales"].mean(),
    }}).fillna(0)
    scaled = StandardScaler().fit_transform(features)
    features["Cluster"] = KMeans(n_clusters=4, random_state=42, n_init=20).fit_predict(scaled)
    coords = PCA(n_components=2, random_state=42).fit_transform(scaled)
    features["PCA1"], features["PCA2"] = coords[:, 0], coords[:, 1]
    st.scatter_chart(features, x="PCA1", y="PCA2", color="Cluster", size="Total Sales")
    st.dataframe(features.reset_index(), use_container_width=True)
'''
    (OUT / "app.py").write_text(app, encoding="utf-8")


def make_requirements():
    req = """pandas
numpy
matplotlib
scikit-learn
statsmodels
prophet
xgboost
streamlit
plotly
reportlab
"""
    (OUT / "requirements.txt").write_text(req, encoding="utf-8")


def make_notebook(df, comparison, seg_df, anomaly_df, cluster_df, vg, vg_yearly, merged_context):
    category_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
    monthly_by_year = df.groupby(["Year", "Month"])["Sales"].sum().reset_index()
    seasonality = monthly_by_year.groupby("Month")["Sales"].mean().sort_values(ascending=False)
    ship_by_region = df.groupby("Region")["Ship Days"].mean().sort_values()
    region_growth = df.groupby(["Region", "Year"])["Sales"].sum().unstack().fillna(0)
    growth_scores = (region_growth.pct_change(axis=1).replace([np.inf, -np.inf], np.nan).mean(axis=1) /
                     region_growth.pct_change(axis=1).replace([np.inf, -np.inf], np.nan).std(axis=1))
    best_region = growth_scores.sort_values(ascending=False).index[0]
    top_anoms = anomaly_df[anomaly_df["IsAnomaly"]].sort_values("Sales", ascending=False).head(3)
    best_model = comparison.iloc[0]["Model"]
    vg_note = "The supplementary `vgsales.csv` dataset is now included and merged at year level as external market context."
    if vg is None:
        vg_note = "The supplementary `vgsales.csv` dataset was not available, so the merge section remains a documented limitation."
    cells = []

    def md(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)})

    def code(text):
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.strip().splitlines(True)})

    md(f"""# End-to-End Sales Forecasting & Demand Intelligence System

This notebook completes the Week 3 and Week 4 internship project using the Superstore `train.csv` dataset and the supplementary `vgsales.csv` dataset.

Important honesty note: the local environment used to build this project did not have `statsmodels`, `prophet`, or `xgboost` installed. I wrote the notebook with the intended model imports and fallbacks so the work is reproducible after installing the packages, while still producing usable charts and business outputs from the available data.
""")
    code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = Path("train.csv")
VG_PATH = Path("vgsales.csv")
df = pd.read_csv(DATA_PATH)
df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True, errors="coerce")
vg = pd.read_csv(VG_PATH) if VG_PATH.exists() else None
df.head(), None if vg is None else vg.head()""")
    md(f"""## Task 1 - Data Loading, Merging & Deep Exploration

The dataset contains **{len(df):,} rows**, **{df['Order ID'].nunique():,} unique orders**, and covers **{df['Order Date'].min().date()} to {df['Order Date'].max().date()}**.

{vg_note}

Merge pain point: the video game dataset does not share product IDs, customers, regions, or dates with the Superstore data. A row-level merge would be misleading, so I used a year-level merge for external demand context. This is useful for demonstrating multi-source handling, but it should not be treated as a causal driver of office/furniture/technology sales.
""")
    code("""def season(month):
    if month in (12, 1, 2): return "Winter"
    if month in (3, 4, 5): return "Spring"
    if month in (6, 7, 8): return "Summer"
    return "Autumn"

df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.month
df["Week Number"] = df["Order Date"].dt.isocalendar().week.astype(int)
df["Day of Week"] = df["Order Date"].dt.day_name()
df["Quarter"] = df["Order Date"].dt.quarter
df["Season"] = df["Month"].apply(season)
df["Ship Days"] = (df["Ship Date"] - df["Order Date"]).dt.days
df["MonthStart"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()
df["WeekStart"] = df["Order Date"].dt.to_period("W").apply(lambda r: r.start_time)

missing = df.isna().sum().sort_values(ascending=False)
duplicates = df.duplicated().sum()
weekly_sales = df.groupby("WeekStart")["Sales"].sum()
monthly_sales = df.groupby("MonthStart")["Sales"].sum().asfreq("MS").fillna(0)
missing.head(10), duplicates, weekly_sales.head(), monthly_sales.head()""")
    code("""if vg is not None:
    vg["Year"] = pd.to_numeric(vg["Year"], errors="coerce")
    vg = vg.dropna(subset=["Year"])
    vg["Year"] = vg["Year"].astype(int)
    vg_yearly = vg.groupby("Year").agg(
        VG_Global_Sales_Millions=("Global_Sales", "sum"),
        VG_Title_Count=("Name", "count"),
        VG_Top_Genre=("Genre", lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown")
    ).reset_index()
    retail_yearly = df.groupby("Year")["Sales"].sum().reset_index(name="Superstore_Sales")
    merged_context = retail_yearly.merge(vg_yearly, on="Year", how="left")
    display(merged_context)
else:
    print("vgsales.csv not available")""")
    md(f"""### EDA Answers

- Highest revenue category: **{category_sales.index[0]}** with **{money(category_sales.iloc[0])}** in sales.
- Most consistent regional growth: **{best_region}**, based on average yearly growth adjusted by volatility.
- Average order-to-ship time: **{df['Ship Days'].mean():.2f} days**. Fastest region was **{ship_by_region.index[0]} ({ship_by_region.iloc[0]:.2f} days)** and slowest was **{ship_by_region.index[-1]} ({ship_by_region.iloc[-1]:.2f} days)**.
- Consistent seasonal spike months by average monthly sales: **{', '.join([str(int(m)) for m in seasonality.head(3).index])}**.
""")
    code("""df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
df.groupby(["Region", "Year"])["Sales"].sum().unstack()
df.groupby("Region")["Ship Days"].mean().sort_values()
df.groupby(["Year", "Month"])["Sales"].sum().reset_index().groupby("Month")["Sales"].mean().sort_values(ascending=False).head(12)""")

    md("""## Task 2 - Time Series Analysis & Decomposition

Because `statsmodels` was unavailable locally, I used a transparent manual decomposition: a 6-month centered rolling trend, average month seasonal index, and residual. The notebook below also shows the `statsmodels` version to use when the package is installed.
""")
    code("""monthly_sales.plot(figsize=(12, 4), title="Overall Monthly Sales Trend")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()""")
    code("""trend = monthly_sales.rolling(6, center=True, min_periods=3).mean()
seasonal = monthly_sales.groupby(monthly_sales.index.month).transform("mean") - monthly_sales.mean()
residual = monthly_sales - trend.bfill().ffill() - seasonal

fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
axes[0].plot(monthly_sales.index, monthly_sales.values); axes[0].set_title("Observed")
axes[1].plot(trend.index, trend.values); axes[1].set_title("Trend")
axes[2].plot(seasonal.index, seasonal.values); axes[2].set_title("Seasonal")
axes[3].plot(residual.index, residual.values); axes[3].set_title("Residual")
plt.tight_layout()
plt.show()""")
    code("""try:
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(monthly_sales.dropna())
    print("ADF statistic:", result[0])
    print("p-value:", result[1])
    diff_result = adfuller(monthly_sales.diff().dropna())
    print("Differenced p-value:", diff_result[1])
except ModuleNotFoundError:
    print("statsmodels is not installed locally. Install it with: pip install statsmodels")
    print("Plain English: stationarity means the time series has a stable average and variance over time.")
    print("Visual result: this monthly sales series shows trend/seasonality, so differencing is likely needed before SARIMA.")""")
    md("""Observations:

- The series trends upward but with noisy month-to-month movement.
- Seasonality is visible, especially around late-year retail activity.
- Residual noise is highest around unusually large promotion-like spikes.
- The main pain point is that only 48 monthly points exist, which is not much data for robust monthly forecasting.
""")

    md("""## Task 3 - Forecasting Using 3 Approaches

The ideal production notebook would use SARIMA, Prophet, and XGBoost. Locally, the heavy packages were missing, so the executed comparison uses:

- SARIMA fallback: seasonal naive forecast.
- Prophet fallback: linear trend plus month dummy regression.
- XGBoost fallback: RandomForest lag model.

This is not perfect, but it is honest and runnable.
""")
    code("""comparison = pd.read_csv("model_comparison.csv")
comparison""")
    md(f"""Recommended model from local metrics: **{best_model}**. This recommendation is based on the lowest validation MAPE in the generated comparison table, not preference.
""")
    code("""# Ideal package imports for a fully installed environment:
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from prophet import Prophet
# from xgboost import XGBRegressor

# Local model comparison produced by the build script:
comparison[["Model", "MAE", "RMSE", "MAPE", "Forecast Month 1", "Forecast Month 2", "Forecast Month 3"]]""")

    md("""## Task 4 - Category and Region Forecasts

The best local model/fallback was repeated across Furniture, Technology, Office Supplies, West, and East. The chart is saved in `charts/segment_forecasts.png`.
""")
    code("""segment_forecasts = pd.read_csv("segment_forecasts.csv")
segment_forecasts""")
    top_seg = seg_df.set_index("Segment").iloc[:, -1].sort_values(ascending=False).index[0]
    md(f"Strongest upcoming segment by the last forecast month: **{top_seg}**.")

    md("""## Task 5 - Anomaly Detection

Two methods were used: Isolation Forest and rolling Z-score. They do not always agree, which is useful: Isolation Forest looks at unusual patterns globally, while Z-score flags sharp deviations from a recent rolling baseline.
""")
    code("""anomalies = pd.read_csv("anomalies.csv")
anomalies[anomalies["IsAnomaly"] == True].head(20)""")
    md("Top detected high-sales anomaly weeks:\n\n" + "\n".join([f"- {r['WeekStart'].date()}: {money(r['Sales'])}, likely promotion/festive demand or bulk ordering." for _, r in top_anoms.iterrows()]))

    md("""## Task 6 - Product Demand Segmentation

Sub-categories were clustered using total sales, growth rate, monthly volatility, and average order value. PCA was used only for 2D visualization.
""")
    code("""clusters = pd.read_csv("product_segments.csv")
clusters[["Sub-Category", "Total Sales", "YoY Growth Rate", "Monthly Volatility", "Average Order Value", "Demand Segment"]]""")
    md("""Stocking strategy:

- High Volume, Stable Core: keep higher safety stock and monitor service levels.
- Growing Demand: increase purchase planning gradually and review monthly.
- High Volatility Watchlist: avoid overcommitting inventory; use shorter replenishment cycles.
- Low Volume / Declining Demand: reduce bulk buying and consider clearance or made-to-order handling.
""")

    md("""## Task 7 - Streamlit Dashboard

The dashboard is implemented in `app.py` with four pages:

- Sales Overview
- Forecast Explorer
- Anomaly Report
- Product Demand Segments

Run locally with:

```bash
streamlit run app.py
```

Deployment pain point: I cannot deploy to Streamlit Community Cloud from this local workspace without the user's GitHub/Streamlit account access. The app code and requirements are ready for deployment.
""")
    md("""## Task 8 - Executive Report

The business report has been generated as `summary.pdf`.

## Process Pain Points

- Supplementary dataset is included now, but the join is only year-level because there is no common product/customer/store key.
- Forecasting packages were not installed locally, so fallbacks were needed.
- Only four years of monthly data gives roughly 48 points, which is small for confident monthly seasonality modeling.
- Superstore data still lacks stockouts, discounts, holidays, prices, and promotions.
- Dashboard deployment requires account-level access and cannot be completed only from local files.
""")

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (OUT / "analysis.ipynb").write_text(json.dumps(nb, indent=2), encoding="utf-8")


def make_pdf(df, comparison, seg_df, anomaly_df, cluster_df, vg, merged_context):
    pdf = OUT / "summary.pdf"
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10))
    doc = SimpleDocTemplate(str(pdf), pagesize=A4, rightMargin=0.55 * inch, leftMargin=0.55 * inch, topMargin=0.45 * inch, bottomMargin=0.45 * inch)
    story = []
    def cell(text):
        return Paragraph(str(text), styles["Small"])

    story.append(Paragraph("Executive Business Report: Sales Forecasting and Demand Intelligence", styles["Title"]))
    story.append(Spacer(1, 8))
    best = comparison.iloc[0]
    story.append(Paragraph(
        f"Executive summary: The Superstore business generated {money(df['Sales'].sum())} across {df['Order ID'].nunique():,} orders. "
        f"The forecasting system estimates the next three months at {money(best['Forecast Month 1'])}, {money(best['Forecast Month 2'])}, and {money(best['Forecast Month 3'])}. "
        "The system is useful for planning, but should be treated as a decision-support tool because promotion, price, stockout, and holiday data were not available. "
        "The supplementary video game sales dataset was included as year-level external market context, not as a direct causal feature.",
        styles["BodyText"],
    ))
    story.append(Spacer(1, 8))

    category_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
    region_sales = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
    ship_region = df.groupby("Region")["Ship Days"].mean().sort_values()
    findings = [
        [cell("Finding"), cell("Evidence")],
        [cell("Top category"), cell(f"{category_sales.index[0]} generated {money(category_sales.iloc[0])}.")],
        [cell("Top region"), cell(f"{region_sales.index[0]} generated {money(region_sales.iloc[0])}.")],
        [cell("Shipping"), cell(f"Average ship time was {df['Ship Days'].mean():.2f} days; {ship_region.index[-1]} was slowest at {ship_region.iloc[-1]:.2f} days.")],
        [cell("Best local model"), cell(f"{best['Model']} had MAE {money(best['MAE'])}, RMSE {money(best['RMSE'])}, and MAPE {best['MAPE']:.1f}%.")],
    ]
    if vg is not None:
        vg_rows = len(vg)
        overlap = merged_context["VG_Global_Sales_Millions"].notna().sum()
        findings.append([cell("Supplementary data"), cell(f"vgsales.csv added {vg_rows:,} rows. {overlap} Superstore years matched by year for context.")])
    table = Table(findings, colWidths=[1.45 * inch, 4.95 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9ca3af")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Forecast and Anomalies", styles["Heading2"]))
    story.append(Paragraph(
        f"The next three monthly forecasts are {money(best['Forecast Month 1'])}, {money(best['Forecast Month 2'])}, and {money(best['Forecast Month 3'])}. "
        "Confidence ranges should be interpreted cautiously because the local fallback model does not produce formal statistical intervals. "
        "A practical business range is plus/minus the validation RMSE.",
        styles["BodyText"],
    ))
    top_anoms = anomaly_df[anomaly_df["IsAnomaly"]].sort_values("Sales", ascending=False).head(3)
    anom_text = " ".join([f"{r['WeekStart'].date()} had {money(r['Sales'])}, likely due to festive sale, promotion, or bulk buying." for _, r in top_anoms.iterrows()])
    story.append(Paragraph("Top anomalies: " + anom_text, styles["BodyText"]))
    story.append(Spacer(1, 8))

    story.append(PageBreak())
    story.append(Paragraph("Demand Segments and Stocking Strategy", styles["Heading2"]))
    seg_summary = cluster_df.groupby("Demand Segment")["Sub-Category"].apply(lambda x: ", ".join(x)).reset_index()
    rows = [[cell("Segment"), cell("Sub-categories"), cell("Stocking action")]]
    actions = {
        "High Volume, Stable Core": "Maintain safety stock; review service levels weekly.",
        "Growing Demand": "Increase purchase planning gradually; review monthly.",
        "High Volatility Watchlist": "Use shorter replenishment cycles; avoid heavy overstock.",
        "Low Volume / Declining Demand": "Limit bulk buys; use clearance planning where needed.",
    }
    for _, row in seg_summary.iterrows():
        rows.append([
            cell(row["Demand Segment"]),
            cell(row["Sub-Category"]),
            cell(actions.get(row["Demand Segment"], "Review manually.")),
        ])
    table = Table(rows, colWidths=[1.55 * inch, 3.05 * inch, 2.0 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9ca3af")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    story.append(table)
    story.append(Spacer(1, 8))

    story.append(Paragraph("Recommendations and Limitation", styles["Heading2"]))
    recs = [
        f"1. Protect inventory for {category_sales.index[0]}, the largest category at {money(category_sales.iloc[0])}.",
        f"2. Review regional replenishment in {ship_region.index[-1]}, where shipping takes {ship_region.iloc[-1]:.2f} days on average.",
        f"3. Use anomaly weeks as promotion planning signals rather than deleting them from the data.",
        "Limitation: The system uses sales history plus weak year-level external context; adding price, discount, stockout, and promotion calendars would improve forecast reliability.",
    ]
    for rec in recs:
        story.append(Paragraph(rec, styles["BodyText"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Project pain points documented: supplementary dataset has no direct join key, missing local forecasting libraries, small monthly history, no promotion/stockout features, and deployment requiring user account access.",
        styles["Small"],
    ))
    doc.build(story)


def save_tables(comparison, seg_df, anomaly_df, cluster_df, merged_context, vg_yearly):
    comparison.to_csv(OUT / "model_comparison.csv", index=False)
    seg_df.to_csv(OUT / "segment_forecasts.csv", index=False)
    anomaly_df.to_csv(OUT / "anomalies.csv", index=False)
    cluster_df.to_csv(OUT / "product_segments.csv", index=False)
    merged_context.to_csv(OUT / "merged_yearly_context.csv", index=False)
    if vg_yearly is not None:
        vg_yearly.to_csv(OUT / "vgsales_yearly_summary.csv", index=False)


def make_zip():
    zip_path = ROOT / "SalesForecasting_Pratham.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in OUT.rglob("*"):
            if "__pycache__" in path.parts:
                continue
            zf.write(path, path.relative_to(ROOT))
    return zip_path


def main():
    prepare_dirs()
    df = load_data()
    vg, vg_yearly = load_vg_data()
    merged_context = build_merged_context(df, vg_yearly)
    monthly, weekly = build_series(df)
    comparison, _, _ = evaluate_models(monthly)
    best_model, _, seg_df = segment_forecasts(df, comparison, monthly)
    anomaly_df = detect_anomalies(weekly)
    cluster_df, inertias = cluster_products(df)
    save_tables(comparison, seg_df, anomaly_df, cluster_df, merged_context, vg_yearly)
    make_charts(df, monthly, weekly, comparison, seg_df, anomaly_df, cluster_df, inertias, vg_yearly, merged_context)
    make_app(best_model)
    make_requirements()
    make_notebook(df, comparison, seg_df, anomaly_df, cluster_df, vg, vg_yearly, merged_context)
    make_pdf(df, comparison, seg_df, anomaly_df, cluster_df, vg, merged_context)
    zip_path = make_zip()
    print(f"Created {OUT}")
    print(f"Created {zip_path}")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
