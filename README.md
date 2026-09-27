# assignment-model-experiment-Ramadhan-Ihsani-Yulfa

Eksperimen dan evaluasi dua pendekatan AI untuk **sentiment analysis ulasan pelanggan e-commerce**:

1. **Model klasik Machine Learning**: TF-IDF + Logistic Regression (Scikit-learn), dilatih sendiri.
2. **LLM API**: Gemini (`gemini-3.5-flash-lite`) dengan prompting zero-shot, tanpa training.

## Struktur Project

```
├── data/
│   └── customer_reviews_sentiment.csv    # dataset asli, tidak diubah
├── notebook/
│   └── experiment_notebook.ipynb         # seluruh eksperimen + output
├── README.md
└── requirements.txt
```

## 1. Problem Statement

**Objective.** Tim produk e-commerce ingin fitur otomatis yang mengklasifikasikan sentimen ulasan pelanggan di halaman produk. Eksperimen ini membandingkan model klasik vs LLM API untuk menentukan pendekatan mana yang layak dikembangkan ke produksi, berdasarkan data.

**Target/label.** Kolom `sentiment` dengan dua kelas: `positif` dan `negatif` (klasifikasi biner). Input: kolom `review_text` (Bahasa Indonesia, informal).

**Batasan & asumsi.**
- Hanya memakai dataset yang dibagikan (200 ulasan, 110 positif / 90 negatif). Dataset tidak diubah dan tidak ditambah.
- Tidak ada kelas netral; setiap ulasan dianggap jelas positif atau negatif.
- Kedua pendekatan dievaluasi pada **test set yang sama** (split 80/20, `stratify`, `random_state=42` → 160 train, 40 test).
- LLM dipakai apa adanya (tanpa fine-tuning), hanya lewat prompt.
- Deep learning dari nol di luar cakupan.

## Dataset

`data/customer_reviews_sentiment.csv` berisi 200 baris dengan kolom `review_id`, `product_name`, `review_text`, `sentiment`. Tidak ada missing value. Catatan penting: hanya ada **40 kalimat ulasan unik** yang diulang untuk produk berbeda, sehingga 98% kalimat di test set juga muncul persis di train set (dihitung di notebook, Section 3).

## 2. Ringkasan Eksperimen

Detail lengkap dan seluruh output ada di [`notebook/experiment_notebook.ipynb`](notebook/experiment_notebook.ipynb).

| | Pendekatan 1: Model Klasik | Pendekatan 2: LLM API |
|---|---|---|
| Metode | TF-IDF + `LogisticRegression` (Scikit-learn, default params) | Gemini `gemini-3.5-flash-lite`, prompt zero-shot |
| Training | Ya, pada 160 data train | Tidak ada |
| Parameter penting | `TfidfVectorizer()` default | `temperature=0.0` (output deterministik & konsisten untuk klasifikasi), retry 5x, timeout 30 detik |
| Post-processing | - | Normalisasi output (lowercase, strip, map ke `positif`/`negatif`, sisanya `tidak_valid`) |

**Prompt** (zero-shot, output dibatasi satu kata agar mudah di-parse):

```
Klasifikasikan sentimen ulasan berikut sebagai "positif" atau "negatif".
Jawab hanya dengan satu kata: positif atau negatif.

Ulasan: "{review_text}"
Sentimen:
```

Catatan model: `gemini-2.0-flash` (starter) dan `gemini-2.5-flash` sudah tidak tersedia untuk pengguna baru, sedangkan `gemini-3.8-flash` hanya memberi kuota free tier 20 request/hari (< 40 data test), sehingga dipakai `gemini-3.5-flash-lite`.

## 3. Hasil Evaluasi & Perbandingan

Evaluasi pada **test set yang sama** (40 ulasan: 22 positif, 18 negatif). Precision/Recall/F1 = rata-rata macro.

| Metrik | Model Klasik (TF-IDF + LogReg) | LLM API (Gemini zero-shot) |
|---|---|---|
| Accuracy | 1.00 | 1.00 |
| Precision | 1.00 | 1.00 |
| Recall | 1.00 | 1.00 |
| F1-Score | 1.00 | 1.00 |
| Confusion matrix (baris = aktual `negatif`, `positif`) | `[[18, 0], [0, 22]]` | `[[18, 0], [0, 22]]` |
| Waktu training | ±0.004 detik | tidak perlu |
| Waktu inferensi 40 ulasan | ±0.0004 detik | ±124 detik (±3 detik/ulasan, termasuk jeda rate limit) |
| Output tidak valid | - | 0 |

**Interpretasi.** Kedua pendekatan memprediksi seluruh 40 ulasan dengan benar: tidak ada false positive (precision 1.00) maupun ulasan yang terlewat (recall 1.00), di kedua kelas. Dari sisi metrik, hasilnya **seri**. Tetapi 98% kalimat test juga muncul persis di train set, sehingga skor model klasik lebih mencerminkan hafalan daripada generalisasi, sedangkan LLM mencapai skor yang sama tanpa pernah melihat data (zero-shot).

## 4. Analisis Trade-off & Limitation

- **Performa:** seri di semua metrik; bukti generalisasi LLM lebih kuat karena zero-shot, sedangkan model klasik diuntungkan duplikasi data.
- **Effort implementasi:** model klasik butuh data berlabel + training + retraining berkala; LLM tanpa training, effort bergeser ke desain prompt, parsing output, error handling, rate limit, dan pengelolaan API key.
- **Kecepatan:** model klasik ±300.000× lebih cepat saat inferensi (lokal, tanpa jaringan). LLM ±3 detik/ulasan di free tier.
- **Biaya:** model klasik praktis gratis setelah dilatih. LLM dibayar per token per ulasan (±60 token input/request), biaya naik linear dengan volume; free tier dibatasi kuota harian per model.

**Limitation model klasik:** vocabulary hanya dari 40 kalimat unik (slang/typo/kata baru tidak dikenali); TF-IDF bag-of-words tidak memahami negasi ("tidak mengecewakan") dan sarkasme; skor 1.00 kemungkinan terlalu optimis karena data leakage dari duplikasi.

**Limitation LLM API:** bergantung layanan eksternal (latensi, rate limit, kuota, model bisa dihentikan, seperti yang terjadi pada `gemini-2.0-flash`/`gemini-2.5-flash`); output teks bebas tetap butuh normalisasi; biaya per request; data ulasan dikirim ke pihak ketiga; kasus sulit (sarkasme, ulasan campuran) belum teruji karena test set berisi kalimat yang eksplisit.

**Kesalahan prediksi yang diamati:** tidak ada pada test set untuk kedua pendekatan.

## 5. Rekomendasi Technical Approach

**Kembangkan LLM API (Gemini, zero-shot) sebagai solusi awal produksi, lalu siapkan model klasik sebagai tahap optimasi biaya.**

1. **Metrik seri, tetapi bukti LLM lebih kuat:** LLM mencapai 1.00 tanpa melihat data, sedangkan 1.00 model klasik didapat dari test set yang 98% sudah ada di train set. LLM lebih bisa dipercaya untuk ulasan nyata yang jauh lebih beragam.
2. **Tidak butuh data berlabel:** tim belum punya dataset representatif (hanya 40 kalimat unik); model klasik yang andal perlu pengumpulan dan pelabelan data besar terlebih dahulu.
3. **Latensi bukan hambatan:** sentimen ulasan tidak perlu real-time, sehingga bisa diproses asynchronous/batch saat ulasan masuk.

Langkah lanjut: pakai akun berbayar (bebas kuota harian) dengan retry, timeout, normalisasi output, dan fallback antrian ulang; simpan label LLM (plus koreksi manual) sebagai dataset berlabel; setelah data cukup, latih ulang model klasik dan evaluasi pada test set tanpa duplikasi. Jika performanya mendekati LLM, pindah ke model klasik atau pola hybrid (klasik untuk prediksi confidence tinggi, LLM untuk kasus ragu) untuk menekan biaya dan latensi.

## Cara Menjalankan

Prasyarat: Python 3.10+ dan API key Gemini ([Google AI Studio](https://aistudio.google.com/apikey)).

```bash
python -m venv .venv
```

Aktifkan venv (Windows: `.venv\Scripts\activate`, Linux/macOS: `source .venv/bin/activate`), lalu:

```bash
pip install -r requirements.txt
```

Buat file `.env` di root project (sudah ada di `.gitignore`, jangan di-commit):

```
GEMINI_API_KEY=isi_api_key_anda
```

Jalankan notebook (kernel otomatis berjalan di folder `notebook/`, sehingga path dataset `../data/` langsung valid):

```bash
jupyter notebook notebook/experiment_notebook.ipynb
```

Atau eksekusi penuh dari terminal:

```bash
jupyter nbconvert --to notebook --execute --inplace notebook/experiment_notebook.ipynb
```

Catatan: free tier Gemini punya batas request per menit dan per hari per model. Client sudah dikonfigurasi retry otomatis, tetapi jika kuota harian habis, tunggu reset atau ganti `model=` di Section 5.2.
