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
    st.header("Eksplorasi Dataset")
    try:
        df = pd.read_csv("data_turbin.csv") 
        st.dataframe(df)
        st.write(df.describe())
    except FileNotFoundError:
        st.warning("File dataset (data_turbin.csv) belum diunggah.")

with tab2:
    st.header("Arsitektur Baseline Neural Network")
    st.write("Model dasar sebelum dilakukan seleksi fitur dan optimasi hyperparameter.")
    st.code('''
    Model: Multi-Layer Perceptron (MLP) Multi-Output
    - Input Layer: 9 Fitur Sensor Operasional
    - Hidden Layers: 64, 32, 16 (Aktivasi: ReLU)
    - Output Layer: 2 (Aktivasi: Linear) untuk Prediksi CO & NOx
    Optimizer: Adam | Loss Function: MSE
    ''')

with tab3:
    st.header("Seleksi Fitur: Genetic Algorithm")
    st.write("Hasil reduksi dimensi dan metrik evaluasi subset fitur terbaik dari proses evolusi GA akan ditampilkan di sini.")

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
