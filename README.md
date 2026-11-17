# Implementasi CNN From Scratch untuk Klasifikasi Pneumonia pada Citra X-Ray Dada

> Model Convolutional Neural Network sebagai alat bantu skrining awal pneumonia, lengkap dengan aplikasi web interaktif berbasis Streamlit.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![SDG](https://img.shields.io/badge/SDG-3%20Good%20Health%20%26%20Well--Being-green)

## Tentang Proyek

Pneumonia masih menjadi salah satu penyakit yang mematikan sehingga deteksi dini sangat diperlukan. Interpretasi citra X-Ray secara manual terkendala oleh keterbatasan waktu, akses, dan jumlah tenaga ahli radiologi, terutama di daerah dengan sumber daya terbatas.

Banyak penelitian sebelumnya memakai arsitektur yang sangat kompleks atau model pre-trained (transfer learning). Proyek ini merancang, melatih, dan mengevaluasi model CNN **murni dari nol (from scratch)** untuk menunjukkan bahwa arsitektur yang sederhana dan dibangun mandiri tetap mampu mencapai performa yang layak sebagai alat bantu skrining awal, sekaligus memahami keterbatasannya.

## Rumusan Masalah

1. Bagaimana merancang arsitektur CNN from scratch yang optimal untuk mengklasifikasikan citra X-Ray dada ke kelas Normal dan Pneumonia?
2. Seberapa baik performa model, ditinjau dari accuracy, precision, recall, dan F1-score?
3. Bagaimana potensi penerapannya sebagai alat bantu skrining awal pneumonia untuk mendukung SDG 3 (Good Health and Well-Being)?

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dari Kaggle (Paul Mooney), dengan dua kelas: **Pneumonia** dan **Normal**.

## Arsitektur dan Metode

| Tahap | Detail |
|-------|--------|
| Input | Citra X-Ray dada grayscale, di-resize menjadi 150×150 piksel |
| Preprocessing | Normalisasi piksel 0–1 (layer `Rescaling`) |
| Augmentasi | Rotasi dan flip horizontal |
| Feature extraction | 3 blok `Conv2D` (32, 64, 128 filter, ReLU), masing-masing diikuti `MaxPooling2D` |
| Klasifikasi | `Flatten` → `Dense(128, ReLU)` → `Dropout(0.5)` → `Dense(1, Sigmoid)` |
| Training | Optimizer Adam, loss Binary Cross-Entropy, Early Stopping |
| Evaluasi | Confusion matrix dan classification report |
| Deployment | Model disimpan sebagai `.h5` dan diintegrasikan ke Streamlit |

## Hasil Evaluasi

Diuji pada 624 citra data uji.

Pada proses training, akurasi training dan validasi sama-sama mencapai sekitar 0,95 dan loss terus menurun, tanpa tanda overfitting yang jelas.

| Kelas | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| Normal | 0.95 | 0.52 | 0.67 | 234 |
| Pneumonia | 0.77 | 0.98 | 0.87 | 390 |
| **Accuracy** | | | **0.81** | 624 |
| Macro avg | 0.86 | 0.75 | 0.77 | 624 |
| Weighted avg | 0.84 | 0.81 | 0.79 | 624 |

**Confusion matrix**

|  | Prediksi Normal | Prediksi Pneumonia |
|--|:--:|:--:|
| **Aktual Normal** | 122 | 112 |
| **Aktual Pneumonia** | 6 | 384 |

Model sangat sensitif terhadap pneumonia (recall 0,98, hanya 6 kasus terlewat), tetapi masih sering salah menandai citra Normal sebagai Pneumonia (112 kasus). Untuk skrining awal ini lebih aman daripada melewatkan pasien, namun menjadi keterbatasan yang perlu diperbaiki.

## Struktur Repositori

```
├── Proyek_X_ray.ipynb              # Notebook pelatihan dan evaluasi model
├── aplikasi.py                     # Aplikasi web Streamlit
├── model_pneumonia_dari_awal.h5    # Model CNN terlatih
├── requirements.txt                # Daftar dependensi
└── README.md
```

## Cara Menjalankan

```bash
# 1. Clone repository
git clone https://github.com/Syithah15/Project-Skrining-Pneumonia-CNN.git
cd Project-Skrining-Pneumonia-CNN

# 2. Install dependensi
pip install -r requirements.txt

# 3. Jalankan aplikasi
streamlit run aplikasi.py
```

## Tautan Proyek

- Aplikasi Streamlit: https://project-skrining-pneumonia-cnn-fmwhrggtzxiyz7motxhxia.streamlit.app/
- Notebook Colab: https://colab.research.google.com/drive/1u_lyqYiK6DF0e0pPxq-mGl0BNAWUwOUx?usp=sharing
- Dataset Kaggle: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## Referensi

- Nugroho dan Puspaningrum (2021), JTIIK: https://doi.org/10.25126/jtiik.2021834515
- Mosi dkk. (2025): https://hostjournals.com/bulletincsr/article/view/576
- Khairudin dkk. (2025), Balok: https://ejurnal.unisan.ac.id/balok/article/view/1462

## Tim Kami

| Nama | NIM |
|------|-----|
| Masyithah | F1G125061 |
| Muh. Dani Nugrah Ramadhan | F1G125067 |
| Erlin | F1G125057 |

**Kelompok:** Kelompok 3

## Disclaimer

Proyek ini dibuat untuk tujuan pembelajaran dan alat bantu skrining awal. Hasil prediksi **bukan pengganti diagnosis dokter atau tenaga radiologi**.