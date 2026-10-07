# Ringkasan Evaluasi Holdout

Kandidat yang diuji adalah MLP (tuning_MLP_3) dengan 19 fitur.

Evaluasi mencakup 204 sampel, dari target 2017-01-01 sampai 2020-11-22.

MAE kandidat ML = 0.0374 m (3.74 cm), RMSE = 0.0493 m, R² = 0.9431.

MAE persistence = 0.0390 m (3.90 cm).

MAE kandidat ML lebih rendah daripada persistence sebesar 0.16 cm.

Pengurangan MAE relatif terhadap persistence: 4.16% (bukan akurasi).

Tahun dengan MAE ML terbesar pada holdout: 2019, sebesar 4.71 cm dari 52 sampel.

Perbandingan tiga metode dengan baseline musiman memakai 165 tanggal bersama.

Rekomendasi yang sudah ditetapkan dari CV: kandidat_ml; tidak diubah melalui tuning holdout.

Hasil ini berlaku untuk prediksi satu minggu ke depan pada lokasi dan sampel yang diuji, dengan riwayat observasi yang tersedia.

Belum ada ambang toleransi operasional yang disepakati, jadi skor ini tidak menjadi sertifikasi layak pakai di lapangan.