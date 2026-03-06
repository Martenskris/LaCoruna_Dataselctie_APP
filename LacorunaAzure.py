import numpy as np
import pandas as pd
import streamlit as st
import pyarrow.dataset as ds
import pyarrow.fs as fs
import plotly.graph_objects as go
from datetime import timedelta

# =========================
# CONFIG
# =========================

LAT_COL = "GPS_x"
LON_COL = "GPS_y"

EXCLUDE = {"Time","Seconds","Minutes","Hours","Year","Month","Day"}

MAX_POINTS = 8000
TIME_STEP = timedelta(minutes=1)

PARQUET_URL = st.secrets["AZURE_BLOB_SAS_URL"]

# =========================
# DATASET CONNECTIE
# =========================

@st.cache_resource
def get_dataset():

    http_fs = fs.HttpFileSystem()

    dataset = ds.dataset(
        PARQUET_URL,
        filesystem=http_fs,
        format="parquet"
    )

    return dataset


dataset = get_dataset()

# =========================
# SCHEMA
# =========================

@st.cache_data
def read_schema():

    schema = dataset.schema

    col_names = schema.names

    col_types = {f.name: f.type for f in schema}

    return col_names, col_types


col_names, col_types = read_schema()

required = ["Timestamp", LAT_COL, LON_COL]

missing = [c for c in required if c not in col_names]

if missing:

    st.error(f"Ontbrekende kolommen: {missing}")
    st.stop()


def is_numeric(pa_type):

    s = str(pa_type).lower()

    return ("int" in s) or ("float" in s) or ("double" in s) or ("bool" in s)


signals = [

    c for c in col_names
    if c not in required
    and c not in EXCLUDE
    and is_numeric(col_types[c])

]

# =========================
# APP
# =========================

st.set_page_config(layout="wide")

st.title("Geo + signalen")

# =========================
# PREVIEW SIGNAAL
# =========================

preview_signal = st.selectbox(
    "Preview signaal",
    signals,
    index=0
)

@st.cache_data
def preview_sample(signal):

    table = dataset.to_table(columns=["Timestamp", signal])

    df = table.to_pandas()

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    df = df.sort_values("Timestamp")

    if len(df) > MAX_POINTS:

        idx = np.linspace(0, len(df)-1, MAX_POINTS).astype(int)

        df = df.iloc[idx]

    return df


preview_df = preview_sample(preview_signal)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=preview_df["Timestamp"],
        y=preview_df[preview_signal],
        mode="lines"
    )
)

fig.update_layout(height=300)

st.plotly_chart(fig, use_container_width=True)

# =========================
# TIJDSELECTIE
# =========================

min_time = preview_df["Timestamp"].min().to_pydatetime()
max_time = preview_df["Timestamp"].max().to_pydatetime()

start_dt, end_dt = st.slider(
    "Tijdslot",
    min_value=min_time,
    max_value=max_time,
    value=(min_time, min_time + timedelta(hours=1)),
    step=TIME_STEP
)

# =========================
# AANTAL SIGNALEN
# =========================

st.subheader("Aantal grafieken")

n_signals = st.slider(
    "Aantal signalen",
    1,
    min(12, len(signals)),
    3
)

# =========================
# SIGNAAL SELECTIE
# =========================

defaults = ["EEC1_Speed","Verbruik_g_per_km","GPS_speed"]

selected = []

for i in range(n_signals):

    default = defaults[i] if i < len(defaults) and defaults[i] in signals else signals[i]

    s = st.selectbox(
        f"Signaal {i+1}",
        signals,
        index=signals.index(default) if default in signals else 0,
        key=f"sig{i}"
    )

    selected.append(s)

# =========================
# DATA FILTEREN
# =========================

cols = ["Timestamp", LAT_COL, LON_COL] + selected

filter_expr = (
    (ds.field("Timestamp") >= pd.Timestamp(start_dt)) &
    (ds.field("Timestamp") <= pd.Timestamp(end_dt))
)

@st.cache_data
def load_filtered(columns, start, end):

    table = dataset.to_table(
        columns=columns,
        filter=filter_expr
    )

    df = table.to_pandas()

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    df = df.sort_values("Timestamp")

    return df


df = load_filtered(cols, start_dt, end_dt)

if df.empty:

    st.warning("Geen data in dit tijdslot")

    st.stop()

# =========================
# GRAFIEKEN
# =========================

st.subheader("Grafieken")

for s in selected:

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Timestamp"],
            y=df[s],
            mode="lines"
        )
    )

    fig.update_layout(
        title=s,
        height=300
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================
# DOWNLOAD
# =========================

st.subheader("Download")

csv = df.to_csv(index=False).encode()

st.download_button(
    "Download CSV",
    csv,
    file_name="export.csv",
    mime="text/csv"
)
