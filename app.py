import streamlit as st
import pandas as pd

# Konfigurasi Halaman
st.set_page_config(
    page_title="Gas Turbine Emission Prediction - Kelompok 04",
    page_icon="⚡",
    layout="wide"
)

# Load Dataset (dengan cache agar ringan)
@st.cache_data
def load_data():
    return pd.read_csv("data_turbin.csv")

df = load_data()

# --- SIDEBAR NAVIGASI (8 MENU SESUAI PANDUAN) ---
st.sidebar.title("Navigasi Dashboard")
st.sidebar.caption("Tugas Besar Kecerdasan Buatan — Kelompok 04")

menu = st.sidebar.radio(
    "Pilih Halaman:",
    [
        "1. Overview",
        "2. Dataset Explorer",
        "3. GA Optimization",
        "4. PSO Optimization",
        "5. Neural Network",
        "6. Fuzzy Inference",
        "7. Prediction / Simulation",
        "8. Model Comparison"
    ]
)

st.sidebar.divider()
st.sidebar.info("Status Progres: **Minggu 13 (GA Selesai)**\n\nMenu 4, 6, dan 7 akan diisi bertahap pada Minggu 14–16.")

# ==========================================
# 1. OVERVIEW
# ==========================================
if menu == "1. Overview":
    st.title("Prediksi Emisi Gas Turbin (CO & NOx)")
    st.subheader("Pendekatan Hybrid: Genetic Algorithm, PSO, Neural Network & Fuzzy Logic")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Informasi Proyek")
        st.write(
            "**Topik:** Prediksi emisi karbon monoksida (CO) dan nitrogen oksida (NOx) "
            "pada pembangkit listrik turbin gas berdasarkan 9 parameter sensor operasional."
        )
        st.write(
            "**Sumber Dataset:** UCI Machine Learning Repository — *Gas Turbine CO and NOx Emission Data Set* "
            "(digabungkan dari periode 2011–2015)."
        )
        st.write(
            "**Tujuan:** Mengoptimalkan seleksi fitur sensor menggunakan GA, melakukan *tuning hyperparameter* "
            "Neural Network menggunakan PSO, serta memberikan klasifikasi/keputusan tingkat emisi menggunakan Fuzzy Inference."
        )

    with col2:
        st.markdown("### Anggota Kelompok 04")
        st.markdown(
            """
            * **Anggota 1:** [Isi Nama] (NIM ...15)
            * **Anggota 2:** [Isi Nama] (NIM ...01)
            * **Anggota 3:** [Isi Nama] (NIM ...16)
            * **Anggota 4:** [Isi Nama] (NIM ...91)
            """
        )

    st.divider()
    st.markdown("### Diagram Alur Sistem")
    st.code(
        "Dataset UCI (9 Sensor) ──► EDA & Preprocessing ──► GA (Seleksi Fitur) "
        "──► PSO (Tuning NN) ──► Neural Network (Prediksi CO & NOx) "
        "──► Fuzzy Inference (Keputusan Tingkat Emisi)",
        language="text"
    )

# ==========================================
# 2. DATASET EXPLORER (W12)
# ==========================================
elif menu == "2. Dataset Explorer":
    st.title("Dataset Explorer (Progres Minggu 12)")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah Baris (Sampel)", f"{df.shape[0]:,}")
    c2.metric("Jumlah Kolom", df.shape[1])
    c3.metric("Total Missing Values", int(df.isnull().sum().sum()))

    st.subheader("1. Pratinjau Dataset (10 Baris Pertama)")
    st.dataframe(df.head(10), use_container_width=True)

    col_stat, col_null = st.columns([2, 1])
    with col_stat:
        st.subheader("2. Statistik Deskriptif")
        st.dataframe(df.describe(), use_container_width=True)
    with col_null:
        st.subheader("3. Cek Missing Values")
        null_df = pd.DataFrame({
            "Kolom": df.columns,
            "Missing Values": df.isnull().sum().values
        })
        st.dataframe(null_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("4. Distribusi Target & Korelasi Fitur")
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.image(
            "W12_EDA_BASELINE_NN_KEL4/distribusi_emisi_(NOX&CO).png",
            caption="Distribusi Emisi Gas Turbin (CO & NOx)",
            use_container_width=True
        )
    with col_img2:
        st.image(
            "W12_EDA_BASELINE_NN_KEL4/matriks_korelasi sensor dan terget emisi.png",
            caption="Matriks Korelasi Sensor dan Target Emisi",
            use_container_width=True
        )

# ==========================================
# 3. GA OPTIMIZATION (W13)
# ==========================================
elif menu == "3. GA Optimization":
    st.title("Genetic Algorithm (GA) Feature Selection (Progres Minggu 13)")
    
    st.markdown("### 1. Representasi Kromosom & Fungsi Fitness")
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.info(
            "**Representasi Kromosom:** Biner 9-bit (mewakili 9 sensor: `AT, AP, AH, AFDP, GTEP, TIT, TAT, TEY, CDP`). "
            "Nilai `1` berarti fitur dipilih, `0` berarti dieliminasi."
        )
    with col_info2:
        st.info(
            "**Fungsi Fitness:** Meminimalkan *Validation MSE* sekaligus memberi penalti jumlah fitur "
            "agar menghasilkan subset sensor paling ringkas dan akurat (dievaluasi di Validation Set, bebas kebocoran Test Set)."
        )

    st.subheader("2. Ringkasan Fitur Terpilih (Seed Terbaik: 91)")
    col_ga1, col_ga2, col_ga3 = st.columns(3)
    col_ga1.metric("Fitur Awal (Baseline)", "9 Sensor")
    col_ga2.metric("Fitur Terpilih (GA)", "7 Sensor", delta="-2 Sensor")
    col_ga3.metric("Sensor Dieliminasi", "TEY, CDP")

    st.divider()
    st.subheader("3. Kurva Konvergensi Fitness GA")
    st.image(
        "W13_GA_FEATURE_SELECTION_KEL4/ga_convergence.png",
        caption="Kurva Konvergensi Fitness GA (Best Fitness vs Average Fitness)",
        use_container_width=True
    )

    st.divider()
    st.subheader("4. Hasil Eksperimen Multi-Seed (4 Seed NIM Kelompok)")
    df_seeds = pd.DataFrame({
        "Seed": [15, 1, 16, 91],
        "Jumlah Fitur": [6, 7, 6, 7],
        "Fitur Terpilih": [
            "AT, AP, AFDP, TIT, TAT, CDP",
            "AT, AH, AFDP, TIT, TAT, TEY, CDP",
            "AT, AH, AFDP, TIT, TAT, TEY",
            "AT, AP, AH, AFDP, GTEP, TIT, TAT"
        ],
        "Best Fitness": [0.79717, 0.80914, 0.80782, 0.80971],
        "Val MSE": [0.24778, 0.22810, 0.23123, 0.22724]
    })
    st.dataframe(df_seeds, use_container_width=True, hide_index=True)

# ==========================================
# 4. PSO OPTIMIZATION (W14 - PLACEHOLDER)
# ==========================================
elif menu == "4. PSO Optimization":
    st.title("Particle Swarm Optimization (PSO) — Tuning Hyperparameter")
    st.warning("🚧 **Tahap Pengembangan (Minggu 14):** Halaman ini akan menampilkan *search space*, kurva *gbest*, dan *hyperparameter* Neural Network terbaik hasil optimasi PSO.")

# ==========================================
# 5. NEURAL NETWORK (W12)
# ==========================================
elif menu == "5. Neural Network":
    st.title("Baseline Neural Network (Progres Minggu 12)")

    st.subheader("1. Arsitektur Model Baseline")
    st.code(
        "Input Layer (9 Fitur Sensor)\n"
        "  └── Dense(64, activation='relu')\n"
        "  └── Dense(32, activation='relu')\n"
        "  └── Dense(16, activation='relu')\n"
        "  └── Output Layer: Dense(2, activation='linear') -> [CO, NOx]",
        language="text"
    )

    st.subheader("2. Metrik Pengujian Baseline NN")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("CO RMSE", "1.1451")
    m2.metric("CO R²", "0.7588")
    m3.metric("NOx RMSE", "4.1992")
    m4.metric("NOx R²", "0.8669")

    st.divider()
    st.subheader("3. Kurva Training & Validation Loss")
    st.image(
        "W12_EDA_BASELINE_NN_KEL4/basline_NN.png",
        caption="Kurva Loss Pelatihan Baseline NN (9 Fitur)",
        use_container_width=True
    )

# ==========================================
# 6. FUZZY INFERENCE (W15 - PLACEHOLDER)
# ==========================================
elif menu == "6. Fuzzy Inference":
    st.title("Fuzzy Inference System (FIS)")
    st.warning("🚧 **Tahap Pengembangan (Minggu 15):** Halaman ini akan menampilkan *membership function plots*, *rule base*, output defuzzifikasi, dan penjelasan keputusan tingkat emisi.")

# ==========================================
# 7. PREDICTION / SIMULATION (W16 - PLACEHOLDER)
# ==========================================
elif menu == "7. Prediction / Simulation":
    st.title("Simulasi & Prediksi Emisi Real-Time")
    st.warning("🚧 **Tahap Pengembangan (Minggu 15–16):** Halaman ini akan menyediakan input manual sensor / unggah CSV untuk menjalankan *inference* Neural Network dan menghasilkan keputusan Fuzzy secara *live*.")

# ==========================================
# 8. MODEL COMPARISON (W13 -> W16)
# ==========================================
elif menu == "8. Model Comparison":
    st.title("Perbandingan Performa Model")
    st.write("Perbandingan bertahap seluruh skenario model (**Baseline NN vs GA+NN** aktif; **PSO+NN** dan **GA+PSO+NN** akan ditambahkan pada Minggu 14).")

    df_comp = pd.DataFrame({
        "Model": [
            "Baseline NN (9 Fitur)",
            "GA + NN (7 Fitur)",
            "PSO + NN (Minggu 14)",
            "GA + PSO + NN (Minggu 14)"
        ],
        "Jumlah Fitur": ["9", "7", "-", "-"],
        "Waktu Latih (detik)": ["90.62", "85.55", "-", "-"],
        "Val Loss (MSE)": ["0.1697", "0.1740", "-", "-"],
        "CO RMSE": ["1.1451", "1.1369", "-", "-"],
        "CO R²": ["0.7588", "0.7623", "-", "-"],
        "NOx RMSE": ["4.1992", "4.5756", "-", "-"],
        "NOx R²": ["0.8669", "0.8420", "-", "-"]
    })
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Visualisasi Perbandingan: Baseline NN vs GA + NN")
    st.image(
        "W13_GA_FEATURE_SELECTION_KEL4/ga_vs_baseline.png",
        caption="Perbandingan Akurasi (R²) & Waktu Pelatihan: Baseline NN vs GA+NN",
        use_container_width=True
    )
