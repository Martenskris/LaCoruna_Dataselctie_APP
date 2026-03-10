import numpy as np
import pandas as pd
import streamlit as st
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import plotly.graph_objects as go
from datetime import timedelta, datetime
import adlfs

# =========================================================
# CONFIG
# =========================================================

LAT_COL = "GPS_x"
LON_COL = "GPS_y"

EXCLUDE = {"Time","Seconds","Minutes","Hours","Year","Month","Day"}

DEFAULT_SIGNALS = ["EEC1_Speed","Verbruik_g_per_km","GPS_speed"]

MAX_POINTS_PREVIEW = 5000
MAX_POINTS_GRAPH = 50000

TIME_STEP = timedelta(minutes=1)

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

    dataset = ds.dataset(path, filesystem=fs, format="parquet")

    return dataset, fs, path


dataset, fs, dataset_path = get_dataset()

# =========================================================
# TIJDRANGE VIA METADATA
# =========================================================

@st.cache_data
def get_time_range():

    parquet = pq.ParquetFile(dataset_path, filesystem=fs)

    schema_names = parquet.schema.names
    ts_index = schema_names.index("Timestamp")

    min_ts = None
    max_ts = None

    for i in range(parquet.metadata.num_row_groups):

        rg = parquet.metadata.row_group(i)
        col = rg.column(ts_index)

        stats = col.statistics

        if stats is None:
            continue

        if min_ts is None or stats.min < min_ts:
            min_ts = stats.min

        if max_ts is None or stats.max > max_ts:
            max_ts = stats.max

    return pd.to_datetime(min_ts), pd.to_datetime(max_ts)


min_time, max_time = get_time_range()

# =========================================================
# SCHEMA
# =========================================================

schema = dataset.schema

col_names = schema.names
col_types = {f.name: f.type for f in schema}

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
# PREVIEW SIGNAAL
# =========================================================

preview_signal = st.selectbox(
    "Preview signaal",
    signals,
    index=0
)

# =========================================================
# PREVIEW DATA
# =========================================================

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

# =========================================================
# TIJDSELECTIE
# =========================================================

st.subheader("Tijdselectie")

c1,c2 = st.columns(2)

start_date = c1.date_input(
    "Start datum",
    min_time.date()
)

start_time = c1.time_input(
    "Start tijd",
    min_time.time(),
    step=60
)

end_date = c2.date_input(
    "Eind datum",
    min_time.date()
)

end_time = c2.time_input(
    "Eind tijd",
    (min_time + timedelta(hours=1)).time(),
    step=60
)

start_dt = datetime.combine(start_date,start_time)
end_dt = datetime.combine(end_date,end_time)

# =========================================================
# SLIDER
# =========================================================

start_dt, end_dt = st.slider(
    "Tijdslot",
    min_value=min_time.to_pydatetime(),
    max_value=max_time.to_pydatetime(),
    value=(start_dt,end_dt),
    step=TIME_STEP
)

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

    default = DEFAULT_SIGNALS[i] if i < len(DEFAULT_SIGNALS) else signals[0]

    s = st.selectbox(
        f"Signaal {i+1}",
        signals,
        index=signals.index(default) if default in signals else 0
    )

    selected.append(s)

# =========================================================
# LOAD BUTTON
# =========================================================

load_data = st.button("Laad geselecteerd tijdslot")

# =========================================================
# DATA LADEN
# =========================================================

if load_data:

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

    if len(df) > MAX_POINTS_GRAPH:

        idx = np.linspace(0,len(df)-1,MAX_POINTS_GRAPH).astype(int)

        df = df.iloc[idx]

    for s in selected:

        df[s] = pd.to_numeric(df[s],errors="coerce")

    # =========================================================
    # GRAFIEKEN
    # =========================================================

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

    # =========================================================
    # DOWNLOADS
    # =========================================================

    st.subheader("Downloads")

    csv_signals = df.to_csv(index=False).encode()

    st.download_button(
        "Download CSV geselecteerde signalen",
        csv_signals,
        file_name="signals_export.csv",
        mime="text/csv"
    )

    scanner_full = dataset.scanner(
        filter=filter_expr,
        batch_size=200000,
        use_threads=True
    )

    df_full = scanner_full.to_table().to_pandas()

    csv_full = df_full.to_csv(index=False).encode()

    st.download_button(
        "Download volledige dataset voor dit tijdslot",
        csv_full,
        file_name="full_timeslot_export.csv",
        mime="text/csv"
    )
