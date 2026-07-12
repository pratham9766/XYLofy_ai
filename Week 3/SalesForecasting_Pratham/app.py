import pandas as pd
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
    sup = pd.DataFrame({"y": series.values}, index=series.index)
    for lag in [1, 2, 3]:
        sup[f"lag_{lag}"] = sup["y"].shift(lag)
    sup["roll3"] = sup["y"].shift(1).rolling(3).mean()
    sup["month"] = sup.index.month
    sup["quarter"] = sup.index.quarter
    sup = sup.dropna()
    features = ["lag_1", "lag_2", "lag_3", "roll3", "month", "quarter"]
    train, test = sup.iloc[:-6], sup.iloc[-6:]
    model = RandomForestRegressor(n_estimators=250, random_state=42, min_samples_leaf=2)
    model.fit(train[features], train["y"])
    pred = model.predict(test[features])
    score = {"MAE": mean_absolute_error(test["y"], pred), "RMSE": mean_squared_error(test["y"], pred) ** 0.5}
    history = series.copy()
    future_idx = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    values = []
    for dt in future_idx:
        row = pd.DataFrame([{"lag_1": history.iloc[-1], "lag_2": history.iloc[-2], "lag_3": history.iloc[-3],
                             "roll3": history.iloc[-3:].mean(), "month": dt.month, "quarter": dt.quarter}])
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
    c1.metric("Total Sales", f"${df['Sales'].sum():,.0f}")
    c2.metric("Orders", f"{df['Order ID'].nunique():,}")
    c3.metric("Avg Ship Days", f"{df['Ship Days'].mean():.1f}")
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
        st.caption(f"Model used: RandomForest lag model, selected locally as practical fallback for Prophet fallback - trend/month regression. MAE=${score['MAE']:,.0f}, RMSE=${score['RMSE']:,.0f}")
    st.dataframe(forecast.reset_index().rename(columns={"index": "Month", 0: "Forecast Sales"}), use_container_width=True)

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
    features = pd.DataFrame({
        "Total Sales": df.groupby("Sub-Category")["Sales"].sum(),
        "YoY Growth Rate": ((df[df["Year"] == last_year].groupby("Sub-Category")["Sales"].sum() - df[df["Year"] == first_year].groupby("Sub-Category")["Sales"].sum()) / df[df["Year"] == first_year].groupby("Sub-Category")["Sales"].sum()).fillna(0),
        "Monthly Volatility": pivot.std(axis=1),
        "Average Order Value": df.groupby("Sub-Category")["Sales"].mean(),
    }).fillna(0)
    scaled = StandardScaler().fit_transform(features)
    features["Cluster"] = KMeans(n_clusters=4, random_state=42, n_init=20).fit_predict(scaled)
    coords = PCA(n_components=2, random_state=42).fit_transform(scaled)
    features["PCA1"], features["PCA2"] = coords[:, 0], coords[:, 1]
    st.scatter_chart(features, x="PCA1", y="PCA2", color="Cluster", size="Total Sales")
    st.dataframe(features.reset_index(), use_container_width=True)
