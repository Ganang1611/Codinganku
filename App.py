import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Hybrid ARIMA-SVR", layout="wide")

st.title("📊 Sistem Prediksi Harga Bawang Merah")
st.markdown("### Model Hybrid ARIMA–SVR (Horizon hingga 24 Bulan)")
st.markdown("---")

# =========================================================
# LOAD MODEL (AMAN PATH)
# =========================================================
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "hybrid_model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model tidak ditemukan di: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Gagal load model: {e}")
    st.stop()

# =========================================================
# VALIDASI MODEL
# =========================================================
required_keys = ["arima_model", "svr_model", "scaler", "lag_order"]

for key in required_keys:
    if key not in model:
        st.error(f"❌ Key '{key}' tidak ditemukan dalam model")
        st.stop()

arima_model = model["arima_model"]
svr_model = model["svr_model"]
scaler = model["scaler"]
lag = model["lag_order"]

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("⚙️ Pengaturan")

steps = st.sidebar.slider("Jumlah Bulan Prediksi", 1, 24, 6)

# =========================================================
# LOAD DATA
# =========================================================
st.subheader("📥 Upload Dataset")

uploaded_file = st.file_uploader("Upload file CSV / XLSX", type=["csv", "xlsx"])

def load_data(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    df.columns = df.columns.str.strip().str.lower()

    # Auto detect kolom
    date_col = None
    price_col = None

    for col in df.columns:
        if "tanggal" in col or "bulan" in col or "date" in col:
            date_col = col
        if "harga" in col or "price" in col:
            price_col = col

    if date_col is None or price_col is None:
        raise ValueError("Dataset harus memiliki kolom tanggal dan harga")

    df = df.rename(columns={date_col: "tanggal", price_col: "harga"})

    df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
    df["harga"] = pd.to_numeric(df["harga"], errors="coerce")

    df = df.dropna().sort_values("tanggal").reset_index(drop=True)

    return df

# =========================================================
# HYBRID FORECAST FUNCTION (STABIL 24 BULAN)
# =========================================================
def hybrid_forecast(steps):
    arima_pred = arima_model.forecast(steps=steps)

    residuals = arima_model.resid.values
    input_res = residuals[-lag:].copy()

    preds = []

    for i in range(steps):
        x = input_res[-lag:].reshape(1, -1)

        x_scaled = scaler.transform(x)
        svr_pred = svr_model.predict(x_scaled)[0]

        # 🔥 STABILISASI (PENTING UNTUK >12 BULAN)
        svr_pred = np.clip(svr_pred, -5000, 5000)

        final = arima_pred.iloc[i] + svr_pred
        preds.append(final)

        input_res = np.append(input_res, svr_pred)

    return np.array(preds)

# =========================================================
# MAIN FLOW
# =========================================================
if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)

        st.subheader("📋 Preview Data")
        st.dataframe(df.tail(), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Jumlah Data", len(df))
        col2.metric("Harga Terakhir", f"{df['harga'].iloc[-1]:,.0f}")
        col3.metric("Tanggal Terakhir", str(df["tanggal"].iloc[-1].date()))

        st.markdown("---")

        # =========================
        # BUTTON
        # =========================
        if st.button("🚀 Jalankan Prediksi"):
            preds = hybrid_forecast(steps)

            # Tanggal masa depan (AKURAT)
            future_dates = pd.date_range(
                start=df["tanggal"].iloc[-1] + pd.offsets.MonthBegin(1),
                periods=steps,
                freq="MS"
            )

            result = pd.DataFrame({
                "Tanggal": future_dates,
                "Prediksi Harga": preds
            })

            st.success("✅ Prediksi berhasil dilakukan")

            # KPI
            c1, c2, c3 = st.columns(3)
            c1.metric("Prediksi Awal", f"{preds[0]:,.0f}")
            c2.metric("Prediksi Akhir", f"{preds[-1]:,.0f}")
            c3.metric("Rata-rata", f"{preds.mean():,.0f}")

            # Tabel
            st.subheader("📊 Hasil Prediksi")
            st.dataframe(result, use_container_width=True)

            # Grafik
            st.subheader("📈 Grafik")
            hist = df.set_index("tanggal")[["harga"]]
            hist.columns = ["Historis"]

            pred = result.set_index("Tanggal")
            pred.columns = ["Prediksi"]

            combined = hist.join(pred, how="outer")

            st.line_chart(combined)

            # Warning validasi
            if any(preds < 0):
                st.warning("⚠️ Terdapat nilai prediksi negatif, perlu validasi model")

            # Download
            csv = result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Hasil",
                csv,
                "prediksi_24_bulan.csv",
                "text/csv"
            )

    except Exception as e:
        st.error(f"❌ Error: {e}")

else:
    st.info("Silakan upload dataset terlebih dahulu")