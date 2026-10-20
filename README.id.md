# Peramalan Permintaan dan Kebijakan Inventory untuk 50 SKU

*Terinspirasi dari tugas kuliah Supply Chain & Operations Management yang awalnya dikerjakan di Excel; dibangun ulang secara mandiri di Python dengan data sintetis.* Dokumentasi lengkap (bahasa Inggris) ada di [README.md](README.md).

## Ringkasan

Pipeline Python yang meramalkan permintaan bulanan 50 SKU, memilih metode terbaik per SKU lewat backtest rolling-origin, lalu menguji apakah ramalan yang lebih baik menurunkan biaya inventory pada tingkat layanan yang sama, memakai simulasi harian pada tahun yang ditahan (held-out).

* **Data**: 100 % sintetis, generator ber-seed (50 SKU x 72 bulan: 60 bulan histori + 1 tahun uji). Tujuh pola: stabil, tren naik/turun, musiman, tren+musiman, intermittent, volatil. Tidak ada data dosen yang disalin.
* **Metode baseline** (sama dengan Excel): naive, MA, WMA, SES, Holt, regresi tren. **Lanjutan**: Holt-Winters, Croston/SBA, LightGBM global.
* **Evaluasi**: MAE, RMSE, sMAPE, MASE, WAPE; pemilihan metode per SKU dibekukan sebelum tahun uji.
* **Inventory**: ABC-XYZ, EOQ, safety stock berbasis error ramalan dan service level, ROP.

## Hasil kunci (seed 42, dari run sungguhan)

* MASE tahun uji: MA(3) 1,344 -> seleksi per SKU 0,757 (turun 43,7 %); baseline terbaik (regresi tren) 1,144. Peningkatan terbesar pada SKU musiman (MASE 1,87 -> 0,48) dan tren+musiman (2,23 -> 0,77). Pada SKU intermittent tidak ada peningkatan berarti.
* Simulasi inventory: biaya total kebijakan klasik (MA3 + SS dari simpangan baku permintaan) 116.203 vs model terbaik + SS dari error ramalan 77.616 (turun 33,2 %); fill rate 98,56 % -> 99,84 %. Dibanding MA3 dengan SS berbasis error: biaya turun 13,3 % dan nilai persediaan 10,9 % lebih rendah.
* Pada 6 dataset sintetis independen: rata-rata penghematan biaya 23,1 % (rentang 15,1-33,2 %) terhadap kebijakan klasik dan 10,9 % terhadap MA3 + SS berbasis error. Seed 42 adalah yang paling menguntungkan dari keenamnya.
* Peringatan sensitivitas: kebijakan terbaik hanya lebih murah jika satu penjualan hilang bernilai lebih dari sekitar 7 % harga pokok; tanpa penalti, kebijakan klasik paling murah.

## Keterbatasan

Data sintetis lebih "bersih" dari data nyata; parameter biaya adalah asumsi; satu tahun simulasi per dataset; SKU intermittent tidak terbantu oleh model yang lebih rumit. Aturan "SKU intermittent hanya boleh memakai metode level" ditambahkan setelah run pertama (dijelaskan di README.md).

## Menjalankan

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
pytest
python scripts/run_pipeline.py --no-cache
jupyter nbconvert --to notebook --execute --inplace notebooks/demand_forecasting.ipynb
```

## Pengungkapan bantuan AI

Repositori ini dibuat dengan bantuan besar dari asisten AI (Claude, Anthropic, lewat Claude Code): kode, tes, notebook, dan dokumentasi ditulis AI berdasarkan arahan pemilik repo, dan seluruh angka berasal dari eksekusi kode di repo ini. Pemilik repo tetap bertanggung jawab atas isinya.

Lisensi: MIT.
