import numpy as np
import pandas as pd
import streamlit as st
import pyarrow.dataset as ds
import plotly.graph_objects as go
from datetime import timedelta, datetime
import adlfs

st.set_page_config(layout="wide")

# =========================================================
# CONFIG
# =========================================================

LAT_COL = "GPS_x"
LON_COL = "GPS_y"

EXCLUDE = {"Time", "Seconds", "Minutes", "Hours", "Year", "Month", "Day"}

MAX_POINTS = 8000
TIME_STEP = timedelta(minutes=1)

DEFAULT_SIGNALS = ["EEC1_Speed", "Verbruik_g_per_km", "GPS_speed"]

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
        sas_token=sas_token,
    )

    dataset = ds.dataset(
        path,
        filesystem=fs,
        format="parquet",
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

if not signals:
    st.error("Geen numerieke signalen gevonden om te tonen.")
    st.stop()

# =========================================================
# APP
# =========================================================

st.title("Geo + signalen")

# =========================================================
# PREVIEW
# =========================================================

st.subheader("Preview")

preview_signal = st.selectbox(
    "Preview signaal",
    signals,
    index=0,
    key="preview_signal",
)


@st.cache_data
def preview_sample(signal):
    scanner = dataset.scanner(
        columns=["Timestamp", signal],
        batch_size=200000,
        use_threads=True,
    )

    table = scanner.head(MAX_POINTS)
    df = table.to_pandas()

    if df.empty:
        return df

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df[signal] = pd.to_numeric(df[signal], errors="coerce")
    df = df.dropna(subset=["Timestamp", signal]).sort_values("Timestamp")

    return df


preview_df = preview_sample(preview_signal)

if preview_df.empty:
    st.warning("Geen previewdata gevonden voor dit signaal.")
    st.stop()

min_time = preview_df["Timestamp"].min().to_pydatetime()
max_time = preview_df["Timestamp"].max().to_pydatetime()

if min_time >= max_time:
    max_time = min_time + timedelta(minutes=1)

# =========================================================
# SESSION STATE
# =========================================================

if "start_dt" not in st.session_state:
    st.session_state.start_dt = min_time
    default_end = min_time + timedelta(hours=1)
    st.session_state.end_dt = min(default_end, max_time)

if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

if "selected_signals" not in st.session_state:
    st.session_state.selected_signals = [s for s in DEFAULT_SIGNALS if s in signals]
    if not st.session_state.selected_signals:
        st.session_state.selected_signals = [signals[0]]

# =========================================================
# CALLBACKS
# =========================================================

def update_from_inputs():
    start = datetime.combine(
        st.session_state.start_date,
        st.session_state.start_time,
    )

    end = datetime.combine(
        st.session_state.end_date,
        st.session_state.end_time,
    )

    if end < start:
        end = start

    st.session_state.start_dt = start
    st.session_state.end_dt = end
    st.session_state.time_slider = (start, end)


def update_from_slider():
    start, end = st.session_state.time_slider

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

c1, c2 = st.columns(2)

with c1:
    st.date_input(
        "Start datum",
        value=st.session_state.start_dt.date(),
        key="start_date",
        on_change=update_from_inputs,
    )

    st.time_input(
        "Start tijd",
        value=st.session_state.start_dt.time(),
        step=60,
        key="start_time",
        on_change=update_from_inputs,
    )

with c2:
    st.date_input(
        "Eind datum",
        value=st.session_state.end_dt.date(),
        key="end_date",
        on_change=update_from_inputs,
    )

    st.time_input(
        "Eind tijd",
        value=st.session_state.end_dt.time(),
        step=60,
        key="end_time",
        on_change=update_from_inputs,
    )

# =========================================================
# SLIDER
# =========================================================

st.slider(
    "Tijdslot",
    min_value=min_time,
    max_value=max_time,
    value=(st.session_state.start_dt, st.session_state.end_dt),
    step=TIME_STEP,
    key="time_slider",
    on_change=update_from_slider,
)

start_dt = st.session_state.start_dt
end_dt = st.session_state.end_dt

# =========================================================
# PREVIEW FIGUUR
# =========================================================

st.write(f"Aantal previewpunten: {len(preview_df)}")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=preview_df["Timestamp"],
        y=preview_df[preview_signal],
        mode="lines",
        name=preview_signal,
    )
)

fig.add_vrect(
    x0=start_dt,
    x1=end_dt,
    fillcolor="rgba(0,0,0,0.15)",
    line_width=0,
)

fig.update_layout(
    title=f"Preview van {preview_signal}",
    xaxis_title="Tijd",
    yaxis_title=preview_signal,
    height=320,
)

st.plotly_chart(fig, use_container_width=True)

# =========================================================
# AANTAL SIGNALEN
# =========================================================

st.subheader("Aantal grafieken")

n_signals = st.number_input(
    "Aantal signalen",
    min_value=1,
    max_value=min(12, len(signals)),
    value=min(3, len(signals)),
    step=1,
    key="num_signals",
)

# =========================================================
# SIGNAAL SELECTIE
# =========================================================

selected = []

for i in range(n_signals):
    if i < len(st.session_state.selected_signals) and st.session_state.selected_signals[i] in signals:
        default = st.session_state.selected_signals[i]
    elif i < len(DEFAULT_SIGNALS) and DEFAULT_SIGNALS[i] in signals:
        default = DEFAULT_SIGNALS[i]
    else:
        default = signals[0]

    s = st.selectbox(
        f"Signaal {i + 1}",
        signals,
        index=signals.index(default),
        key=f"signal_select_{i}",
    )

    selected.append(s)

st.session_state.selected_signals = selected

# =========================================================
# LOAD BUTTON
# =========================================================

if st.button("Laad geselecteerd tijdslot"):
    st.session_state.data_loaded = True

# =========================================================
# DATA LADEN (SNEL)
# =========================================================

if st.session_state.data_loaded:
    cols = list(dict.fromkeys(["Timestamp", LAT_COL, LON_COL] + selected))

    filter_expr = (
        (ds.field("Timestamp") >= pd.Timestamp(start_dt))
        & (ds.field("Timestamp") <= pd.Timestamp(end_dt))
    )

    scanner = dataset.scanner(
        columns=cols,
        filter=filter_expr,
        batch_size=200000,
        use_threads=True,
    )

    table = scanner.to_table()
    df = table.to_pandas()

    if df.empty:
        st.warning("Geen data gevonden in het geselecteerde tijdslot.")
    else:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp")

        for s in selected:
            df[s] = pd.to_numeric(df[s], errors="coerce")

        st.subheader("Grafieken")

        for s in selected:
            sub_df = df.dropna(subset=[s])

            if sub_df.empty:
                st.info(f"Geen bruikbare waarden voor signaal: {s}")
                continue

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=sub_df["Timestamp"],
                    y=sub_df[s],
                    mode="lines",
                    name=s,
                )
            )

            fig.update_layout(title=s, height=300)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Download")

        csv = df.to_csv(index=False).encode()

        st.download_button(
            "Download CSV",
            csv,
            file_name="export.csv",
            mime="text/csv",
        )
