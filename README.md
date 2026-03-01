# Review Credibility NLP (MVP)

fitur utama sistem **skor kredibilitas ulasan** berbasis NLP + indikator linguistik:
- TF-IDF (bag-of-words)
- Multinomial Naive Bayes
- Indikator linguistik: panjang teks, jumlah digit (detail konkret), TTR (type-token ratio), frasa generik, dan *template similarity* (kemiripan ke ulasan lain).

## Instalasi
Butuh Python 3.9+ dan paket:
- pandas
- numpy
- scipy
- scikit-learn

Contoh:
```bash
pip install -r requirements.txt
```

## Data
CSV minimal:
- `text` : isi ulasan
- `label`: 0 (asli) / 1 (manipulatif)

## Training + evaluasi (5-fold CV)
```bash
python3 train.py --data examples/sample_reviews.csv --out models/model_v1
```
Output:
- metrik CV (precision/recall/F1/accuracy)
- model tersimpan di `models/model_v1/`

## Scoring (inferensi)
```bash
python3 predict.py --model-dir models/model_v1 --input examples/sample_unlabeled.csv --output scored.csv
```

Kolom output:
- `p_fake` : probabilitas ulasan manipulatif
- `credibility_score` : 0-100 (semakin tinggi semakin kredibel)

## Catatan tentang template similarity
Similarity adalah fitur yang sensitif terhadap *data leakage*.
- Saat CV: similarity untuk data validasi dihitung terhadap **data train** pada fold itu.
- Saat inferensi: default dihitung terhadap batch input (MVP). Untuk produk, sebaiknya hitung terhadap **korpus referensi** (mis. ulasan terbaru untuk SKU/seller) dengan opsi `--reference-csv`.
