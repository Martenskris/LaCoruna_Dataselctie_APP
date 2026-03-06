import streamlit as st
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from azure.storage.blob import BlobClient
from io import BytesIO

st.set_page_config(page_title="Geo Dashboard", layout="wide")

# =========================================================
# Config
# =========================================================

def get_blob_client() -> BlobClient:
    """
    Verwacht in st.secrets:
    AZURE_BLOB_SAS_URL = "https://<account>.blob.core.windows.net/<container>/<blob>?<sas>"
    """
    sas_url = st.secrets["AZURE_BLOB_SAS_URL"]
    return BlobClient.from_blob_url(sas_url)


# =========================================================
# Blob helpers
# =========================================================

@st.cache_data(show_spinner=False)
def download_blob_bytes() -> bytes:
    """
    Download de blob volledig in geheugen.
    Geschikt voor kleine tot middelgrote parquet-bestanden.
    """
    blob_client = get_blob_client()
    blob_data = blob_client.download_blob().readall()
    if not blob_data:
        raise ValueError("De blob is leeg.")
    return blob_data


def validate_parquet_bytes(blob_data: bytes) -> None:
    """
    Basiscontrole om sneller te zien of de blob echt parquet is.
    Een parquet-bestand start en eindigt normaal met b'PAR1'.
    """
    if len(blob_data) < 8:
        raise ValueError("Blob is te klein om een geldig parquet-bestand te zijn.")

    if blob_data[:4] != b"PAR1" or blob_data[-4:] != b"PAR1":
        preview = blob_data[:120]
        raise ValueError(
            f"Blob lijkt geen geldig parquet-bestand. Eerste bytes: {preview!r}"
        )


# =========================================================
# Data readers
# =========================================================

@st.cache_data(show_spinner=False)
def read_schema():
    """
    Lees enkel het schema uit parquet.
    """
    blob_data = download_blob_bytes()
    validate_parquet_bytes(blob_data)

    reader = pa.BufferReader(blob_data)
    pf = pq.ParquetFile(reader)

    schema = pf.schema_arrow
    col_names = schema.names
    col_types = [str(t) for t in schema.types]

    return col_names, col_types


@st.cache_data(show_spinner=True)
def load_dataframe(columns=None, nrows=None) -> pd.DataFrame:
    """
    Laad parquet vanuit blob naar pandas DataFrame.
    columns: optionele lijst kolommen
    nrows: optioneel aantal rijen om te tonen
    """
    blob_data = download_blob_bytes()
    validate_parquet_bytes(blob_data)

    table = pq.read_table(pa.BufferReader(blob_data), columns=columns)
    df = table.to_pandas()

    if nrows is not None:
        df = df.head(nrows)

    return df


# =========================================================
# UI
# =========================================================

st.title("Geo Dashboard")
st.caption("Leest data rechtstreeks uit Azure Blob Storage via SAS URL.")

with st.sidebar:
    st.header("Instellingen")

    show_debug = st.toggle("Debug info tonen", value=False)

    nrows = st.number_input(
        "Aantal rijen tonen",
        min_value=5,
        max_value=5000,
        value=100,
        step=5,
    )

# =========================================================
# Main
# =========================================================

try:
    col_names, col_types = read_schema()

    st.subheader("Schema")
    schema_df = pd.DataFrame(
        {
            "kolom": col_names,
            "type": col_types,
        }
    )
    st.dataframe(schema_df, use_container_width=True)

    selected_columns = st.multiselect(
        "Kies kolommen om te laden",
        options=col_names,
        default=col_names[: min(10, len(col_names))]
    )

    if selected_columns:
        df = load_dataframe(columns=selected_columns, nrows=nrows)

        st.subheader("Data preview")
        st.dataframe(df, use_container_width=True)

        st.subheader("Samenvatting")
        st.write(f"Aantal getoonde rijen: {len(df)}")
        st.write(f"Aantal kolommen: {len(df.columns)}")

    if show_debug:
        st.subheader("Debug")
        blob_data = download_blob_bytes()
        st.write("Blob size (bytes):", len(blob_data))
        st.write("Eerste 32 bytes:", blob_data[:32])
        st.write("Laatste 32 bytes:", blob_data[-32:])

except Exception as e:
    st.error("Fout bij het lezen van de blob/parquet-data.")
    st.exception(e)
