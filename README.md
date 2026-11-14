# Gas Turbine Emission Intelligence System

**Tugas Besar Kecerdasan Buatan (TE2514031) - Gasal 2026/2027**  
**Institut Teknologi Kalimantan (ITK)**

---

## 👥 Kelompok 4
Sistem ini dikembangkan oleh:
1. 04231091 - Yogi Lambang Pratama
2. 04231016 - Dafa Vieri Rizky Fitrah
3. 04231001 - Abdul Kahfi
4. 04241015 - Azka Bramantyo Fathoriq Algifari

## 📌 Deskripsi Proyek
Proyek ini bertujuan untuk mengembangkan sistem kecerdasan buatan hibrida untuk memprediksi emisi Gas Karbon Monoksida (CO) dan Nitrogen Oksida (NOx) dari turbin gas. Sistem ini mengintegrasikan beberapa komponen AI:
* **Neural Network (MLP Multi-Output Regression):** Sebagai mesin prediksi utama untuk mengestimasi tingkat emisi CO dan NOx.
* **Genetic Algorithm (GA):** Digunakan untuk melakukan seleksi sensor/fitur paling optimal.
* **Particle Swarm Optimization (PSO):** Digunakan untuk optimasi *hyperparameter* dari model Neural Network.
* **Fuzzy Logic (Decision-Support Layer):** Mengubah nilai prediksi emisi menjadi keputusan status emisi (*Safe - Moderate - High - Critical*).

## 📊 Dataset
Dataset yang digunakan adalah **Gas Turbine CO and NOx Emission Data Set (UCI ID 551)** dari UCI Machine Learning Repository.  
Link Dataset: (https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set)

## 🚀 Cara Menjalankan Aplikasi (Local Development)
1. *Clone* repositori ini ke komputer lokal Anda.
2. Pastikan Python sudah terinstall.
3. Install semua kebutuhan *library* dengan perintah:
   ```bash
   pip install -r requirements.txt# TUBES_KB_2026_KelasB_Kelompok04
