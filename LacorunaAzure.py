import numpy as np
import pandas as pd
import streamlit as st
import pyarrow.dataset as ds
import plotly.graph_objects as go
from datetime import timedelta, datetime
import adlfs

# =========================================================
# CONFIG
# =========================================================

LAT_COL = "GPS_x"
LON_COL = "GPS_y"

EXCLUDE = {"Time","Seconds","Minutes","Hours","Year","Month","Day"}

MAX_POINTS_PREVIEW = 5000
MAX_POINTS_GRAPH = 50000

TIME_STEP = timedelta(minutes=1)

DEFAULT_SIGNALS = ["EEC1_Speed","Verbruik_g_per_km","GPS_speed"]

PARQUET_URL = st.secrets["AZURE_BLOB_SAS_URL"]

# =========================================================
# DATASET
# =========================================================

@st.cache_resource
def get_dataset():

    account = PARQUET_URL.split(".")[0].split("//")[1]
    sas_token = PARQUET_URL.split("?")[1]
    path = PARQUET_URL.split(".net/")[1].split("?")[0]

    fs = adlfs.AzureBlobFileSystem(
        account_name=account,
        sas_token=sas_token
    )

    dataset = ds.dataset(
        path,
        filesystem=fs,
        format="parquet"
    )

    return dataset


dataset = get_dataset()

# =========================================================
# SCHEMA
# =========================================================

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

    return ("int" in s) or ("float" in s) or ("double" in s)


signals = [
    c for c in col_names
    if c not in required
    and c not in EXCLUDE
    and is_numeric(col_types[c])
]

# =========================================================
# APP
# =========================================================

st.set_page_config(layout="wide")

st.title("Geo + signalen")

# =========================================================
# PREVIEW
# =========================================================

preview_signal = st.selectbox(
    "Preview signaal",
    signals,
    index=0,
)

@st.cache_data
def preview_sample(signal):

    scanner = dataset.scanner(
        columns=["Timestamp", signal],
        batch_size=200000,
        use_threads=True
    )

    table = scanner.head(MAX_POINTS_PREVIEW)

    df = table.to_pandas()

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    return df


preview_df = preview_sample(preview_signal)

min_time = preview_df["Timestamp"].min().to_pydatetime()
max_time = preview_df["Timestamp"].max().to_pydatetime()

# =========================================================
# SESSION STATE
# =========================================================

if "start_dt" not in st.session_state:

    st.session_state.start_dt = min_time
    st.session_state.end_dt = min_time + timedelta(hours=1)

if "data_loaded" not in st.session_state:

    st.session_state.data_loaded = False

if "selected_signals" not in st.session_state:

    st.session_state.selected_signals = DEFAULT_SIGNALS.copy()

# =========================================================
# TIJDSELECTIE
# =========================================================

st.subheader("Tijdselectie")

start_dt, end_dt = st.slider(
    "Tijdslot",
    min_value=min_time,
    max_value=max_time,
    value=(st.session_state.start_dt, st.session_state.end_dt),
    step=TIME_STEP
)

st.session_state.start_dt = start_dt
st.session_state.end_dt = end_dt

# =========================================================
# PREVIEW GRAFIEK
# =========================================================

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

fig.update_layout(height=300)

st.plotly_chart(fig,use_container_width=True)

# =========================================================
# AANTAL SIGNALEN
# =========================================================

st.subheader("Aantal grafieken")

n_signals = st.number_input(
    "Aantal signalen",
    min_value=1,
    max_value=min(12,len(signals)),
    value=3,
    step=1
)

# =========================================================
# SIGNAAL SELECTIE
# =========================================================

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

# =========================================================
# LOAD BUTTON
# =========================================================

if st.button("Laad geselecteerd tijdslot"):

    st.session_state.data_loaded = True

# =========================================================
# DATA LADEN (ROW GROUP PRUNING)
# =========================================================

if st.session_state.data_loaded:

    cols = list(dict.fromkeys(["Timestamp",LAT_COL,LON_COL]+selected))

    filter_expr = (
        (ds.field("Timestamp") >= pd.Timestamp(start_dt)) &
        (ds.field("Timestamp") <= pd.Timestamp(end_dt))
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

    # sampling voor snelle grafieken
    if len(df) > MAX_POINTS_GRAPH:

        idx = np.linspace(0,len(df)-1,MAX_POINTS_GRAPH).astype(int)

        df = df.iloc[idx]

    for s in selected:

        df[s] = pd.to_numeric(df[s], errors="coerce")

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

        fig.update_layout(title=s,height=300)

        st.plotly_chart(fig,use_container_width=True)

    st.subheader("Download")

    csv = df.to_csv(index=False).encode()

    st.download_button(
        "Download CSV",
        csv,
        file_name="export.csv",
        mime="text/csv"
    )
