import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import numpy as np

st.set_page_config(page_title="Gas Turbine Emission Dashboard", layout="wide")
st.title("Gas Turbine Emission Intelligence System")
st.write("Sistem prediksi emisi CO dan NOx terintegrasi. (Versi Beta)")

tab1, tab2, tab3, tab4 = st.tabs(["Dataset Explorer", "GA & PSO Optimization", "Neural Network & Fuzzy", "Prediction/Simulation"])

with tab1:
    st.header("Eksplorasi Dataset")
    st.write("Berikut adalah cuplikan data parameter operasional turbin gas:")
    
    try:
        # Membaca file CSV yang sudah diunggah ke GitHub
        df = pd.read_csv("data_turbin.csv")
        
        # Menampilkan tabel data yang bisa di-scroll
        st.dataframe(df)
        
        # Menampilkan statistik deskriptif dasar
        st.subheader("Statistik Dataset")
        st.write(df.describe())
    except FileNotFoundError:
        st.error("File data_turbin.csv belum ditemukan di repository.")

with tab2:
    st.write("Halaman parameter algoritma genetika dan PSO akan muncul di sini.")
with tab3:
    st.write("Halaman arsitektur model dan aturan Fuzzy akan muncul di sini.")
with tab4:
    st.header("Simulasi Prediksi Emisi (Baseline NN)")
    
    try:
        # Memuat scaler dan model
        scaler_X = joblib.load('scaler_X.pkl')
        scaler_y = joblib.load('scaler_y.pkl')
        model = load_model('baseline_model.h5')
        
        # Contoh input manual sederhana (hanya untuk tes)
        st.write("Masukkan nilai sensor simulasi:")
        input_AT = st.number_input("Ambient Temperature (AT)", value=15.0)
        input_AP = st.number_input("Ambient Pressure (AP)", value=1013.0)
        # ... (Tambahkan input lain sesuai jumlah fitur Anda) ...
        
        if st.button("Prediksi Emisi"):
            # Proses input, scaling, dan prediksi
            # (Pastikan array input disesuaikan dengan jumlah kolom fitur asli Anda)
            input_array = np.zeros((1, 9)) # Contoh jika ada 9 fitur
            input_array[0, 0] = input_AT
            input_array[0, 1] = input_AP
            
            input_scaled = scaler_X.transform(input_array)
            pred_scaled = model.predict(input_scaled)
            pred_asli = scaler_y.inverse_transform(pred_scaled)
            
            st.success(f"Prediksi CO: {pred_asli[0][0]:.4f}")
            st.error(f"Prediksi NOx: {pred_asli[0][1]:.4f}")
            
    except Exception as e:
        st.warning("Model belum diunggah atau ada kesalahan format input.")
