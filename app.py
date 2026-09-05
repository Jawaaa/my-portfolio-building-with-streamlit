import os
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Setup Halaman Streamlit
st.set_page_config(
    page_title="Zahwa Rizzi Ani | Data Portfolio",
    page_icon="🏠",
    layout="wide"
)

st.sidebar.title("📌 Navigasi Portofolio")

# UPDATE: variabel perantara buat nampung "tujuan" pindah halaman dari tombol lain
if "nav_target" not in st.session_state:
    st.session_state.nav_target = None

# UPDATE: proses perpindahan ini dilakukan SEBELUM widget radio dibuat
if st.session_state.nav_target is not None:
    st.session_state.page = st.session_state.nav_target
    st.session_state.nav_target = None

if "page" not in st.session_state:
    st.session_state.page = "Tentang Saya"

page = st.sidebar.radio(
    "Pilih Halaman:",
    ["Tentang Saya", "Proyek Saya", "Cek & Visualisasi Data (EDA)", "Prediksi Model (.pkl)"],
    key="page"
)

# Loader (dengan cache) untuk memuat model dan data
@st.cache_resource
def load_models():
    with open("models/models.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_metrics():
    with open("models/metrics.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_raw_data():
    return pd.read_csv("data/house_prices_sample.csv")

# About me page
if page == "Tentang Saya":
    st.title("👋 My Portfolio with Streamlit")
    st.markdown(
        "Selamat datang di aplikasi portofolio interaktif saya! Aplikasi ini menampilkan "
        "profil, proyek analitik, visualisasi data, serta modul prediksi machine learning "
        "berbasis dataset **House Prices (Kaggle)**."
    )
    st.divider()

    col_photo, col_bio, col_stats = st.columns([1.2, 3, 2])

    with col_photo:
        photo_path = "assets/foto_zahwa.jpg"
        if os.path.exists(photo_path):
            st.image(photo_path, width=180, caption="Zahwa Rizzi Ani")
        else:
            st.info("💡 Letakkan file foto_zahwa.jpg di dalam folder assets/")

    with col_bio:
        st.markdown("""
        Hi! Saya **Zahwa Rizzi Ani**, lulusan **Teknik Fisika (Engineering Physics)** yang
        berdomisili di **Bogor**. Saya memiliki passion mendalam di bidang
        **Data Science, Machine Learning, dan Artificial Intelligence**.

        Dengan kombinasi latar belakang pemikiran sistemik fisika dan analitik data, saya
        terbiasa membangun *pipeline machine learning*, analisis statistik, dan aplikasi
        data interaktif untuk mendukung pengambilan keputusan bisnis.
        """)
        st.markdown("""
        **Tech Stack & Skills:**
        `Python` · `SQL` · `Machine Learning` · `Streamlit` · `Exploratory Data Analysis` · `Predictive Modeling`
        """)

    with col_stats:
        with st.container(border=True):
            st.markdown("#### Quick Summary")
            st.write("🎓 **Background:** Engineering Physics")
            st.write("📍 **Location:** Bogor, Indonesia")
            st.write("📜 **Focus:** Data Analytics & ML Engineering")

# Project page
elif page == "Proyek Saya":
    st.title("📂 Proyek Portofolio Saya")
    st.write("Berikut beberapa proyek berbasis data dan machine learning yang telah saya kerjakan:")
    st.divider()

    projects = [
        {
            "title": "1. House Price Prediction",
            "desc": "Memprediksi harga properti menggunakan Random Forest & Gradient Boosting berdasarkan fitur fisik bangunan (dataset Kaggle House Prices).",
            "image": "assets/project_house_price.png",
            "metric": "R² = 0.897 | Full ML Pipeline",
            "github": None  # menyusul
        },
        {
            "title": "2. Customer Churn Data Storytelling",
            "desc": "Menganalisis perilaku pelanggan untuk memahami faktor-faktor yang menyebabkan churn (berhenti berlangganan). Proyek ini menggunakan pendekatan data storytelling berbasis Python untuk mengubah data transaksi menjadi insight yang mudah dipahami stakeholder bisnis.",
            "image": "assets/project_churn.png",
            "metric": "Tech: Python (Pandas, NumPy, Matplotlib, Seaborn)",
            "github": "https://github.com/Jawaaa/customer-churn-data-storytelling"
        },
        {
            "title": "3. E-commerce Product Lifecycle & Sales Trends",
            "desc": "Proyek ini menggunakan Exploratory Data Analysis (EDA) berbasis Python untuk menganalisis siklus hidup produk dan pola penjualan e-commerce, mengidentifikasi performa produk, dan memberikan insight untuk pengambilan keputusan bisnis.",
            "image": "assets/project_ecommerce.png",
            "metric": "Tech: Python (Pandas, NumPy, Matplotlib, Seaborn)",
            "github": "https://github.com/Jawaaa/e-commerce-product-lifecycle-and-sales-trends"
        },
    ]

    cols = st.columns(3)
    for col, proj in zip(cols, projects):
        with col:
            with st.container(border=True):
                st.subheader(proj["title"])
                if os.path.exists(proj["image"]):
                    st.image(proj["image"], use_container_width=True)
                else:
                    st.caption("📷 (gambar proyek belum ditambahkan)")
                st.write(proj["desc"])
                st.info(proj["metric"])

                if proj["github"]:
                    st.link_button("🔗 Lihat di GitHub", proj["github"])
                else:
                    if st.button("📊 Lihat EDA & Prediksi", key=f"goto_{proj['title']}"):
                        st.session_state.nav_target = "Cek & Visualisasi Data (EDA)"   # UPDATE: bukan .page langsung
                        st.rerun()
                        
# EDA + Visualization Performa Model page
elif page == "Cek & Visualisasi Data (EDA)":
    st.title("📊 Eksplorasi & Visualisasi Data")
    df = load_raw_data()

    tab1, tab2, tab3 = st.tabs(["Preview Data", "Distribusi Fitur", "Korelasi Antar Fitur"])

    with tab1:
        st.write(f"Dataset memiliki **{df.shape[0]} baris** dan **{df.shape[1]} kolom**.")
        st.dataframe(df.head(10), use_container_width=True)

    with tab2:
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        default_idx = numeric_cols.index("SalePrice") if "SalePrice" in numeric_cols else 0
        selected_col = st.selectbox("Pilih fitur numerik:", numeric_cols, index=default_idx)
        fig = px.histogram(df, x=selected_col, nbins=40, marginal="box",
                            title=f"Distribusi {selected_col}")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        numeric_df = df.select_dtypes(include=["int64", "float64"])
        corr = numeric_df.corr()
        top_corr = corr["SalePrice"].abs().sort_values(ascending=False).head(15).index
        fig2 = px.imshow(numeric_df[top_corr].corr(), text_auto=".2f", aspect="auto",
                          title="Korelasi Top 15 Fitur terhadap SalePrice",
                          color_continuous_scale="RdBu_r")
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("🤖 Visualisasi Performa Model")

    metrics = load_metrics()
    model_choice = st.selectbox("Pilih model untuk divisualisasikan:", list(metrics.keys()))
    m = metrics[model_choice]

    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"{m['rmse']:,.2f}")
    c2.metric("MAE", f"{m['mae']:,.2f}")
    c3.metric("R² Score", f"{m['r2']:.3f}")

    colA, colB = st.columns(2)
    with colA:
        fig_pred = px.scatter(
            x=m["y_test"], y=m["y_pred"],
            labels={"x": "Actual SalePrice", "y": "Predicted SalePrice"},
            title=f"Actual vs Predicted — {model_choice}"
        )
        fig_pred.add_shape(type="line", x0=m["y_test"].min(), y0=m["y_test"].min(),
                            x1=m["y_test"].max(), y1=m["y_test"].max(),
                            line=dict(color="red", dash="dash"))
        st.plotly_chart(fig_pred, use_container_width=True)

    with colB:
        fig_lc = go.Figure()
        fig_lc.add_trace(go.Scatter(x=m["train_sizes"], y=m["train_scores_mean"], name="Train Score"))
        fig_lc.add_trace(go.Scatter(x=m["train_sizes"], y=m["test_scores_mean"], name="Validation Score"))
        fig_lc.update_layout(title=f"Learning Curve — {model_choice}",
                              xaxis_title="Training Size", yaxis_title="R² Score")
        st.plotly_chart(fig_lc, use_container_width=True)

# Prediksi Model page (UPLOAD CSV -> BUTTON -> HASIL)
elif page == "Prediksi Model (.pkl)":
    st.title("🔮 Prediksi Harga Rumah")
    st.write("Upload file CSV berisi data rumah (kolom harus sama seperti dataset training, tanpa kolom `SalePrice`).")

    models = load_models()

    # UPDATE: pilihan mode — 1 model atau bandingkan semua sekaligus
    mode = st.radio("Mode Prediksi:", ["Pilih 1 Model", "Bandingkan Semua Model"], horizontal=True)

    if mode == "Pilih 1 Model":
        model_choice = st.selectbox("Pilih model:", list(models.keys()))

    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)

        # Auto-drop kolom yang nggak dipakai pipeline (biar cocok sama saat training)
        if "Id" in input_df.columns:
            input_df = input_df.drop(columns=["Id"])
            st.caption("ℹ️ Kolom `Id` terdeteksi dan otomatis di-drop sebelum prediksi.")

        if "SalePrice" in input_df.columns:
            input_df = input_df.drop(columns=["SalePrice"])
            st.caption("ℹ️ Kolom `SalePrice` terdeteksi dan otomatis di-drop (bukan input, tapi target prediksi).")

        st.write("**Preview data yang diupload:**")
        st.dataframe(input_df.head(), use_container_width=True)

        if st.button("🚀 Jalankan Prediksi"):
            try:
                if mode == "Pilih 1 Model":
                    # ----- MODE SATU MODEL -----
                    pipe = models[model_choice]
                    predictions = pipe.predict(input_df)

                    result_df = input_df.copy()
                    result_df["Predicted_SalePrice"] = predictions

                    st.success(f"Prediksi berhasil menggunakan model **{model_choice}**!")
                    st.dataframe(result_df, use_container_width=True)

                    csv_result = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Download Hasil Prediksi", data=csv_result,
                                        file_name="hasil_prediksi.csv", mime="text/csv")

                else:
                    # UPDATE: ----- MODE BANDINGKAN SEMUA MODEL -----
                    result_df = input_df.copy()

                    for name, pipe in models.items():
                        preds = pipe.predict(input_df)
                        col_name = f"Predicted_{name.replace(' ', '_')}"
                        result_df[col_name] = preds

                    st.success(f"Prediksi berhasil dijalankan untuk **{len(models)} model**: {', '.join(models.keys())}")
                    st.dataframe(result_df, use_container_width=True)

                    # UPDATE: ringkasan statistik perbandingan antar model
                    st.subheader("📊 Ringkasan Perbandingan Prediksi")
                    pred_cols = [c for c in result_df.columns if c.startswith("Predicted_")]
                    summary = result_df[pred_cols].describe().T[["mean", "min", "max", "std"]]
                    summary.columns = ["Rata-rata", "Minimum", "Maksimum", "Std Dev"]
                    st.dataframe(summary.style.format("{:,.0f}"), use_container_width=True)

                    # UPDATE: chart perbandingan visual (untuk beberapa baris pertama)
                    st.subheader("📈 Grafik Perbandingan (10 Baris Pertama)")
                    chart_df = result_df[pred_cols].head(10).reset_index(drop=True)
                    chart_df.index.name = "Baris ke-"
                    fig_compare = px.bar(
                        chart_df, barmode="group",
                        title="Perbandingan Prediksi Harga Antar Model (10 Baris Pertama)",
                        labels={"value": "Predicted SalePrice", "index": "Baris ke-", "variable": "Model"}
                    )
                    st.plotly_chart(fig_compare, use_container_width=True)

                    csv_result = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Download Semua Hasil Prediksi", data=csv_result,
                                        file_name="hasil_prediksi_semua_model.csv", mime="text/csv")

            except Exception as e:
                st.error(f"Terjadi error saat prediksi: {e}")
    else:
        st.info("Silakan upload file CSV terlebih dahulu.")