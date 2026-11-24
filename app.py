import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Gas Turbine Emission Dashboard", layout="wide")
st.title("Gas Turbine Emission Intelligence System")
st.write("Sistem prediksi emisi CO dan NOx terintegrasi. (Versi Beta)")

# Memisahkan tab menjadi 6 bagian berurutan sesuai target mingguan
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Dataset Explorer", 
    "Baseline NN (Minggu 12)",
    "Genetic Algorithm (Minggu 13)", 
    "PSO Optimization (Minggu 14)", 
    "Fuzzy Logic (Minggu 15)", 
    "Prediction/Simulation"
])

with tab1:
    st.header("Dataset Gas Turbine Emission")
    df = pd.read_csv("data_turbin.csv")
    
    col_info1, col_info2 = st.columns(2)
    col_info1.metric("Jumlah Baris (Sampel)", f"{df.shape[0]:,}")
    col_info2.metric("Jumlah Kolom (Sensor & Target)", df.shape[1])
    
    st.subheader("Pratinjau Data (10 Baris Pertama)")
    st.dataframe(df.head(10), use_container_width=True)
    
    with st.expander("Lihat Statistik Deskriptif Dataset"):
        st.dataframe(df.describe(), use_container_width=True)

with tab2:
    st.header("Progres Minggu 12: EDA & Baseline Neural Network")
    
    st.subheader("1. Visualisasi Distribusi & Korelasi")
    col1, col2 = st.columns(2)
    with col1:
        st.image(
            "W12_EDA_BASELINE_NN_KEL4/distribusi_emisi_(NOX&CO).png",
            caption="Distribusi Emisi Gas Turbin (CO & NOx)",
            use_container_width=True
        )
    with col2:
        st.image(
            "W12_EDA_BASELINE_NN_KEL4/matriks_korelasi sensor dan terget emisi.png",
            caption="Matriks Korelasi Sensor dan Target Emisi",
            use_container_width=True
        )
        
    st.divider()
    st.subheader("2. Evaluasi & Kurva Pelatihan Baseline NN (9 Fitur)")
    st.image(
        "W12_EDA_BASELINE_NN_KEL4/basline_NN.png",
        caption="Kurva Loss Pelatihan Baseline NN (9 Fitur)",
        use_container_width=True
    )

with tab3:
    st.header("Progres Minggu 13: Seleksi Fitur (Genetic Algorithm)")
    st.write(
        "Implementasi Algoritma Genetika (GA) untuk mereduksi jumlah fitur sensor gas turbin "
        "menggunakan evaluasi Validation Set (bebas kebocoran Test Set) dan pengujian multi-seed."
    )

    # 1. Konfigurasi & Ringkasan Fitur Terpilih
    st.subheader("1. Konfigurasi GA & Fitur Terpilih")
    col_ga1, col_ga2, col_ga3 = st.columns(3)
    col_ga1.metric("Fitur Awal (Baseline)", "9 Sensor")
    col_ga2.metric("Fitur Terpilih (GA)", "7 Sensor", delta="-2 Sensor")
    col_ga3.metric("Sensor Dieliminasi", "TEY, CDP")

    # 2. Visualisasi Grafik GA
    st.divider()
    st.subheader("2. Kurva Konvergensi & Perbandingan Performa")
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.image(
            "W13_GA_FEATURE_SELECTION_KEL4/ga_convergence.png",
            caption="Kurva Konvergensi Fitness GA (Best vs Average)",
            use_container_width=True
        )
    with col_img2:
        st.image(
            "W13_GA_FEATURE_SELECTION_KEL4/ga_vs_baseline.png",
            caption="Perbandingan Akurasi (R²) & Waktu Pelatihan: Baseline NN vs GA+NN",
            use_container_width=True
        )

    # 3. Tabel Bukti Multi-Seed
    st.divider()
    st.subheader("3. Hasil Eksperimen Multi-Seed")
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

    # 4. Tabel Perbandingan Baseline NN vs GA+NN
    st.subheader("4. Tabel Perbandingan: Baseline NN vs GA + NN")
    df_comp = pd.DataFrame({
        "Model": ["Baseline NN (9 Fitur)", "GA + NN (7 Fitur)"],
        "Jumlah Fitur": [9, 7],
        "Waktu Latih (detik)": [90.62, 85.55],
        "Val Loss (MSE)": [0.1697, 0.1740],
        "CO RMSE": [1.1451, 1.1369],
        "CO R²": [0.7588, 0.7623],
        "NOx RMSE": [4.1992, 4.5756],
        "NOx R²": [0.8669, 0.8420]
    })
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

with tab4:
    st.header("Optimasi Hyperparameter: PSO")
    st.write("Hasil pencarian parameter learning rate dan jumlah neuron optimal untuk Neural Network akan ditampilkan di sini.")

with tab5:
    st.header("Sistem Keputusan Fuzzy Logic")
    st.info("Implementasi penentuan status risiko emisi (Safe - Moderate - High - Critical) menggunakan logika Fuzzy Takagi-Sugeno-Kang dijadwalkan pada Minggu 15.")

with tab6:
    st.header("Simulasi Prediksi Emisi (Baseline NN)")
    
    try:
        scaler_X = joblib.load('W12_EDA_BASELINE_NN/scaler_X.pkl')
        scaler_y = joblib.load('W12_EDA_BASELINE_NN/scaler_y.pkl')
        model = load_model('W12_EDA_BASELINE_NN/baseline_model.h5')
        
        st.write("Masukkan nilai operasional turbin gas (9 parameter):")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            input_AT = st.number_input("Ambient Temp (AT)", value=15.0)
            input_AP = st.number_input("Ambient Pressure (AP)", value=1013.0)
            input_AH = st.number_input("Ambient Humidity (AH)", value=60.0)
        with col2:
            input_AFDP = st.number_input("Air Filter Diff Pressure (AFDP)", value=4.0)
            input_GTEP = st.number_input("Gas Turbine Exh Pressure (GTEP)", value=24.0)
            input_TIT = st.number_input("Turbine Inlet Temp (TIT)", value=1080.0)
        with col3:
            input_TAT = st.number_input("Turbine After Temp (TAT)", value=550.0)
            input_TEY = st.number_input("Turbine Energy Yield (TEY)", value=130.0)
            input_CDP = st.number_input("Compressor Discharge Press (CDP)", value=12.0)
        
        if st.button("Prediksi Emisi", type="primary"):
            input_array = np.array([[input_AT, input_AP, input_AH, input_AFDP, input_GTEP, input_TIT, input_TAT, input_TEY, input_CDP]])
            
            input_scaled = scaler_X.transform(input_array)
            pred_scaled = model.predict(input_scaled)
            pred_asli = scaler_y.inverse_transform(pred_scaled)
            
            st.markdown("### Hasil Prediksi:")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.success(f"Karbon Monoksida (CO): {pred_asli[0][0]:.4f} mg/m³")
            with res_col2:
                st.error(f"Nitrogen Oksida (NOx): {pred_asli[0][1]:.4f} mg/m³")
                
    except Exception as e:
        st.warning("Silakan unggah model (baseline_model.h5) dan scaler (scaler_X.pkl, scaler_y.pkl) terlebih dahulu agar simulasi berjalan.")
        st.error(f"Detail error dari sistem: {e}")
