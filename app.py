import streamlit as st

st.set_page_config(page_title="Gas Turbine Emission Dashboard", layout="wide")
st.title("Gas Turbine Emission Intelligence System")
st.write("Sistem prediksi emisi CO dan NOx terintegrasi. (Versi Beta)")

tab1, tab2, tab3, tab4 = st.tabs(["Dataset Explorer", "GA & PSO Optimization", "Neural Network & Fuzzy", "Prediction/Simulation"])

with tab1:
    st.write("Halaman eksplorasi dataset dan visualisasi akan muncul di sini.")
with tab2:
    st.write("Halaman parameter algoritma genetika dan PSO akan muncul di sini.")
with tab3:
    st.write("Halaman arsitektur model dan aturan Fuzzy akan muncul di sini.")
with tab4:
    st.write("Halaman simulasi prediksi emisi akan muncul di sini.")
