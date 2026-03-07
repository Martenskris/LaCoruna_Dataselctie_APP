import numpy as np
import pandas as pd
import streamlit as st
import pyarrow.dataset as ds
import plotly.graph_objects as go
from datetime import timedelta
import adlfs

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

LAT_COL = "GPS_x"
LON_COL = "GPS_y"

MAX_POINTS_PREVIEW = 8000
MAX_POINTS_GRAPH = 50000

TIME_STEP = timedelta(minutes=1)

DEFAULT_SIGNALS = ["EEC1_Speed","Verbruik_g_per_km","GPS_speed"]

EXCLUDE = {"Time","Seconds","Minutes","Hours","Year","Month","Day"}

PARQUET_URL = st.secrets["AZURE_BLOB_SAS_URL"]

# -------------------------------------------------
# DATASET
# -------------------------------------------------

@st.cache_resource
def get_dataset():

    account = PARQUET_URL.split(".")[0].split("//")[1]
    sas_token = PARQUET_URL.split("?")[1]
    path = PARQUET_URL.split(".net/")[1].split("?")[0]

    fs = adlfs.AzureBlobFileSystem(
        account_name=account,
        sas_token=sas_token
    )

    return ds.dataset(path, filesystem=fs, format="parquet")

dataset = get_dataset()

# -------------------------------------------------
# SCHEMA
# -------------------------------------------------

@st.cache_data
def read_schema():

    schema = dataset.schema

    names = schema.names
    types = {f.name: f.type for f in schema}

    return names, types

col_names, col_types = read_schema()

required = ["Timestamp",LAT_COL,LON_COL]

for r in required:
    if r not in col_names:
        st.error(f"Kolom ontbreekt: {r}")
        st.stop()

def is_numeric(t):

    s = str(t).lower()

    return "int" in s or "float" in s or "double" in s

signals = [
    c for c in col_names
    if c not in required
    and c not in EXCLUDE
    and is_numeric(col_types[c])
]

# -------------------------------------------------
# STREAMLIT
# -------------------------------------------------

st.set_page_config(layout="wide")
st.title("Geo + Signalen")

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------

if "selected_signals" not in st.session_state:
    st.session_state.selected_signals = DEFAULT_SIGNALS.copy()

# -------------------------------------------------
# LAYOUT
# -------------------------------------------------

left, right = st.columns([1,2])

# -------------------------------------------------
# LINKS: SIGNAL SELECTIE
# -------------------------------------------------

with left:

    st.subheader("Signalen")

    preview_signal = st.selectbox(
        "Preview signaal",
        signals,
        index=0
    )

    n_signals = st.number_input(
        "Aantal signalen",
        min_value=1,
        max_value=min(12,len(signals)),
        value=3,
        step=1
    )

    selected=[]

    for i in range(n_signals):

        if i < len(st.session_state.selected_signals):
            default = st.session_state.selected_signals[i]
        else:
            default = signals[0]

        s = st.selectbox(
            f"Signaal {i+1}",
            signals,
            index=signals.index(default)
        )

        selected.append(s)

    st.session_state.selected_signals = selected

# -------------------------------------------------
# PREVIEW DATA
# -------------------------------------------------

@st.cache_data
def load_preview(signal):

    scanner = dataset.scanner(
        columns=["Timestamp",signal],
        batch_size=200000,
        use_threads=True
    )

    table = scanner.to_table()

    df = table.to_pandas()

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    df = df.sort_values("Timestamp")

    if len(df) > MAX_POINTS_PREVIEW:

        idx = np.linspace(0,len(df)-1,MAX_POINTS_PREVIEW).astype(int)

        df = df.iloc[idx]

    return df

preview_df = load_preview(preview_signal)

min_time = preview_df["Timestamp"].min().to_pydatetime()
max_time = preview_df["Timestamp"].max().to_pydatetime()

# -------------------------------------------------
# RECHTS: PREVIEW
# -------------------------------------------------

with right:

    start_dt,end_dt = st.slider(
        "Tijdslot",
        min_value=min_time,
        max_value=max_time,
        value=(min_time,min_time+timedelta(hours=1)),
        step=TIME_STEP
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=preview_df["Timestamp"],
            y=preview_df[preview_signal],
            mode="lines"
        )
    )

    fig.add_vrect(
        x0=start_dt,
        x1=end_dt,
        fillcolor="rgba(0,0,0,0.2)",
        line_width=0
    )

    fig.update_layout(height=250)

    st.plotly_chart(fig,use_container_width=True)

# -------------------------------------------------
# LOAD BUTTON
# -------------------------------------------------

load_button = st.button("Laad geselecteerd tijdslot")

# -------------------------------------------------
# DATA LADEN
# -------------------------------------------------

if load_button:

    cols = list(dict.fromkeys(["Timestamp",LAT_COL,LON_COL]+selected))

    filter_expr = (
        (ds.field("Timestamp")>=pd.Timestamp(start_dt)) &
        (ds.field("Timestamp")<=pd.Timestamp(end_dt))
    )

    scanner = dataset.scanner(
        columns=cols,
        filter=filter_expr,
        batch_size=200000,
        use_threads=True
    )

    table = scanner.to_table()

    df = table.to_pandas()

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    if len(df) > MAX_POINTS_GRAPH:

        idx = np.linspace(0,len(df)-1,MAX_POINTS_GRAPH).astype(int)

        df = df.iloc[idx]

    # -------------------------------------------------
    # GEO PLOT
    # -------------------------------------------------

    st.subheader("Geo")

    fig = go.Figure()

    fig.add_trace(
        go.Scattermapbox(
            lat=df[LAT_COL],
            lon=df[LON_COL],
            mode="markers",
            marker=dict(size=6)
        )
    )

    fig.update_layout(
        mapbox_style="open-street-map",
        height=600
    )

    st.plotly_chart(fig,use_container_width=True)

    # -------------------------------------------------
    # SIGNAL GRAFIEKEN
    # -------------------------------------------------

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

        fig.update_layout(title=s,height=250)

        st.plotly_chart(fig,use_container_width=True)

    # -------------------------------------------------
    # DOWNLOAD
    # -------------------------------------------------

    csv = df.to_csv(index=False).encode()

    st.download_button(
        "Download CSV",
        csv,
        file_name="export.csv",
        mime="text/csv"
    )
