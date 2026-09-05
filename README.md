# 🏠 My Portfolio with Streamlit

Aplikasi portofolio interaktif berbasis **Streamlit** yang menampilkan profil, proyek data science, eksplorasi data, dan modul prediksi machine learning menggunakan dataset **House Prices - Advanced Regression Techniques (Kaggle)**.

🔗 **Live App:** [my-portfolio-building-with-streamlit.streamlit.app](https://my-portfolio-building-zahwa-rizzi-ani.streamlit.app/)

---

## 📋 Fitur Utama

- **Tentang Saya** — profil singkat, latar belakang, dan tech stack.
- **Proyek Saya** — 3 proyek data science dengan gambar, deskripsi, dan tautan ke repository GitHub.
- **Eksplorasi & Visualisasi Data (EDA)** — distribusi fitur, korelasi antar fitur, dan visualisasi performa model (RMSE, MAE, R², Actual vs Predicted, Learning Curve).
- **Prediksi Model** — upload file CSV, pilih 1 model atau bandingkan 3 model sekaligus (Linear Regression, Random Forest, Gradient Boosting), lihat & unduh hasil prediksi.

---

## 🧠 Machine Learning Pipeline

| Tahap | Deskripsi |
|---|---|
| Data | [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) (1460 baris, 80 kolom) |
| Preprocessing | Imputasi missing value (median/mode, serta `"None"`/`0` untuk fitur yang memang tidak ada), scaling numerik, one-hot encoding kategorikal |
| Model | Linear Regression, Random Forest, Gradient Boosting |
| Evaluasi | RMSE, MAE, R², Learning Curve |
| Best Model | **Gradient Boosting** (R² = 0.897) |

Pipeline training ada di [`train_model.py`](train_model.py), hasilnya disimpan sebagai `models/models.pkl` dan `models/metrics.pkl`, lalu dimuat langsung oleh aplikasi Streamlit tanpa perlu training ulang.

---

## 🛠️ Tech Stack

`Python` · `Streamlit` · `Pandas` · `NumPy` · `Scikit-learn` · `Plotly`

---

## 📁 Struktur Project
my-portfolio-with-streamlit/
├── app.py # Aplikasi Streamlit utama
├── train_model.py # Script training pipeline ML
├── requirements.txt # Daftar dependency
├── data/ # Dataset (train.csv, test.csv, house_prices_sample.csv)
├── models/ # Model & metrics hasil training (models.pkl, metrics.pkl)
└── assets/ # Gambar profil & proyek


---

## 🚀 Cara Menjalankan Secara Lokal

```bash
# 1. Clone repository
git clone https://github.com/Jawaaa/my-portfolio-building-with-streamlit.git
cd my-portfolio-building-with-streamlit

# 2. Buat virtual environment & install dependency
python -m venv env
env\Scripts\activate        # Windows
pip install -r requirements.txt

# 3. (Opsional) Jalankan ulang training pipeline
python train_model.py

# 4. Jalankan aplikasi
streamlit run app.py
```

---

## 👩‍💻 Tentang Saya

**Zahwa Rizzi Ani** — lulusan Teknik Fisika (Engineering Physics), berdomisili di Bogor, dengan fokus di bidang Data Science, Machine Learning, dan Artificial Intelligence.

---

## 📄 Proyek Lain

- [Customer Churn Data Storytelling](https://github.com/Jawaaa/customer-churn-data-storytelling)
- [E-commerce Product Lifecycle & Sales Trends](https://github.com/Jawaaa/e-commerce-product-lifecycle-and-sales-trends)