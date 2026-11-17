import streamlit as st
import pandas as pd

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
    st.write("Halaman simulasi prediksi emisi akan muncul di sini.")
