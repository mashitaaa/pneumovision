---
title: Pneumonia Detection dari Chest X-Ray
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
license: mit
---

# 🫁 PneumoVision+: Platform Klasifikasi Otomatis Pneumonia pada Citra Chest X-Ray Berbasis Kecerdasan Buatan Menggunakan EfficientNetB0 

Aplikasi klasifikasi gambar medis untuk mendeteksi pneumonia dari foto X-Ray dada menggunakan **Transfer Learning (EfficientNetB0)**.

## 📌 Tentang Project

**CPMK 6 — Klasifikasi Gambar Medis menggunakan Transfer Learning**

| Item | Detail |
|------|--------|
| Model | EfficientNetB0 (pretrained ImageNet) |
| Dataset | [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) |
| Jumlah Data | 5.863 gambar X-Ray |
| Kelas | Normal / Pneumonia |
| Framework | TensorFlow / Keras |

## 🚀 Cara Pakai

1. Upload foto X-Ray dada (format JPG/PNG)
2. Klik tombol **Analisis X-Ray**
3. Lihat hasil prediksi dan tingkat keyakinan model

## 🧠 Arsitektur Model

- **Base model:** EfficientNetB0 (frozen, pretrained ImageNet)
- **Custom head:** GlobalAveragePooling → Dense(256) → Dropout(0.4) → Dense(64) → Sigmoid
- **Training:** 2 phase — frozen base → fine-tuning 30 layer terakhir
- **Augmentasi:** rotation, zoom, flip, shift
