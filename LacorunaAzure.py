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

MAX_POINTS = 8000
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

    return ds.dataset(path, filesystem=fs, format="parquet")


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

    return ("int" in s) or ("float" in s) or ("double" in s) or ("bool" in s)

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
    key="preview_signal"
)

@st.cache_data
def preview_sample(signal):

    table = dataset.to_table(columns=["Timestamp", signal])

    df = table.to_pandas()

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    df = df.sort_values("Timestamp")

    if len(df) > MAX_POINTS:

        idx = np.linspace(0,len(df)-1,MAX_POINTS).astype(int)

        df = df.iloc[idx]

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
# CALLBACKS
# =========================================================

def update_from_inputs():

    start = datetime.combine(
        st.session_state.start_date,
        st.session_state.start_time
    )

    end = datetime.combine(
        st.session_state.end_date,
        st.session_state.end_time
    )

    st.session_state.start_dt = start
    st.session_state.end_dt = end
    st.session_state.time_slider = (start,end)


def update_from_slider():

    start,end = st.session_state.time_slider

    st.session_state.start_dt = start
    st.session_state.end_dt = end

    st.session_state.start_date = start.date()
    st.session_state.start_time = start.time()

    st.session_state.end_date = end.date()
    st.session_state.end_time = end.time()

# =========================================================
# TIJDSELECTIE
# =========================================================

st.subheader("Tijdselectie")

c1,c2 = st.columns(2)

with c1:

    st.date_input(
        "Start datum",
        value=st.session_state.start_dt.date(),
        key="start_date",
        on_change=update_from_inputs
    )

    st.time_input(
        "Start tijd",
        value=st.session_state.start_dt.time(),
        step=60,
        key="start_time",
        on_change=update_from_inputs
    )

with c2:

    st.date_input(
        "Eind datum",
        value=st.session_state.end_dt.date(),
        key="end_date",
        on_change=update_from_inputs
    )

    st.time_input(
        "Eind tijd",
        value=st.session_state.end_dt.time(),
        step=60,
        key="end_time",
        on_change=update_from_inputs
    )

# =========================================================
# SLIDER
# =========================================================

st.slider(
    "Tijdslot",
    min_value=min_time,
    max_value=max_time,
    value=(st.session_state.start_dt,st.session_state.end_dt),
    step=TIME_STEP,
    key="time_slider",
    on_change=update_from_slider
)

start_dt = st.session_state.start_dt
end_dt = st.session_state.end_dt

# =========================================================
# PREVIEW FIGUUR
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
    fillcolor="rgba(0,0,0,0.15)",
    line_width=0
)

fig.add_vline(x=start_dt)
fig.add_vline(x=end_dt)

fig.update_layout(height=320)

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
    step=1,
    key="num_signals"
)

# =========================================================
# SIGNAAL SELECTIE (keuzes blijven behouden)
# =========================================================

selected=[]

for i in range(n_signals):

    if i < len(st.session_state.selected_signals):
        default = st.session_state.selected_signals[i]
    elif i < len(DEFAULT_SIGNALS) and DEFAULT_SIGNALS[i] in signals:
        default = DEFAULT_SIGNALS[i]
    else:
        default = signals[0]

    s = st.selectbox(
        f"Signaal {i+1}",
        signals,
        index=signals.index(default) if default in signals else 0,
        key=f"signal_select_{i}"
    )

    selected.append(s)

st.session_state.selected_signals = selected

# =========================================================
# LOAD BUTTON
# =========================================================

if st.button("Laad geselecteerd tijdslot", key="load_button"):
    st.session_state.data_loaded = True

# =========================================================
# DATA LADEN NA BUTTON
# =========================================================

if st.session_state.data_loaded:

    cols=["Timestamp",LAT_COL,LON_COL]+selected

    filter_expr=(
        (ds.field("Timestamp")>=pd.Timestamp(start_dt))&
        (ds.field("Timestamp")<=pd.Timestamp(end_dt))
    )

    table=dataset.to_table(columns=cols,filter=filter_expr)

    df=table.to_pandas()

    df["Timestamp"]=pd.to_datetime(df["Timestamp"])

    df=df.sort_values("Timestamp")

    if df.empty:
        st.warning("Geen data in geselecteerd tijdslot")
        st.stop()

    st.subheader("Grafieken")

    for s in selected:

        fig=go.Figure()

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

    csv=df.to_csv(index=False).encode()

    st.download_button(
        "Download CSV",
        csv,
        file_name="export.csv",
        mime="text/csv"
    )
