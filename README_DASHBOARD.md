# Groundwater Lab — Rasen

Prototipe dashboard untuk eksplorasi data air tanah dan prediksi rata-rata hydraulic head satu minggu ke depan pada studi kasus Drenthe, Belanda. Dibangun sebagai kelanjutan proyek akhir Machine Learning & AI DQLab.

## Pertanyaan proyek

Apakah riwayat air tanah dan cuaca dapat memperbaiki prediksi satu minggu ke depan dibanding persistence, yaitu memakai head minggu ini sebagai perkiraan minggu depan?

## Alur analisis

1. Identifikasi masalah, pemeriksaan kalender harian, dan agregasi mingguan.
2. EDA pada periode pengembangan.
3. Lag, rolling, fitur musim, serta target satu minggu ke depan.
4. Perbandingan delapan algoritma dan tiga kelompok fitur menggunakan validasi berdasarkan waktu; tuning kandidat pada pengembangan.
5. Evaluasi model terkunci pada holdout berdasarkan tanggal target mulai 1 Januari 2017.
6. Dashboard membaca hasil evaluasi serta memakai pipeline tersimpan untuk prediksi.

Target yang hilang tidak diinterpolasi. Semua metode dalam tabel evaluasi dibandingkan pada tanggal yang sama; seasonal naïve memakai irisan tanggal ketika riwayat 52 minggu tersedia. Dashboard tidak memilih ulang model berdasarkan holdout.

## Hasil

Skor, nama algoritma, dan kesimpulan dibaca dari `outputs/evaluation/evaluation_summary.json`, bukan diketik tetap dalam aplikasi. Gunakan hasil lokal tersebut untuk menulis laporan. R² bukan persentase akurasi; pengurangan MAE terhadap persistence juga bukan akurasi.

## Menjalankan

Ikuti `PANDUAN_DASHBOARD.md`. Pasang dependensi dashboard pada interpreter model, kemudian jalankan `python -m streamlit run app.py` dari folder proyek.

## Berkas yang digunakan

- Notebook 01–05 tetap berada di folder `notebooks`.
- CSV mingguan, skema fitur, dan holdout tetap berada di `data/processed`.
- Pipeline serta metadata tetap berada di `models`.
- Hasil evaluasi tetap berada di `outputs/evaluation`.
- `app.py`, `dashboard_utils.py`, dan `.streamlit/config.toml` berada di akar proyek.

Paket dashboard tidak memuat ulang data dari internet dan tidak melatih model. Tidak ada unggahan model joblib dari pengunjung; hanya pipeline lokal hasil notebook 04 yang dimuat.

## Batasan

Satu sumur, data historis, cuaca arsip, dan evaluasi pada sampel lengkap. Prediksi dilakukan satu minggu ke depan dengan riwayat yang sudah diamati, bukan ramalan multi-tahun sekaligus. Belum ada interval ketidakpastian yang dikalibrasi atau toleransi kesalahan operasional yang disepakati. Penerapan pada lokasi lain memerlukan evaluasi baru. Prototipe ini tidak menentukan keamanan lereng dan tidak mengendalikan dewatering.

## Sumber

- [Groundwater Time Series Modelling Challenge](https://github.com/gwmodeling/challenge)
- [Dokumentasi Netherlands](https://github.com/gwmodeling/challenge/blob/main/data/Netherlands/README.md)
- [Collenteur dkk. (2024)](https://doi.org/10.5194/hess-28-5193-2024)

Pertahankan atribusi sumber. Periksa ketentuan lisensi dataset dan repositori sebelum redistribusi. Proyek memakai rancangan prediksi sendiri, bukan mereplikasi aturan challenge asal.

## Project Presentation

[View the project presentation (PDF)](Groundwater_Forecasting_Raden Satrio Hibatull Rasendriyo_Portfolio.pdf).
