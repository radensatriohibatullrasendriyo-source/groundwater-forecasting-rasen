# Menjalankan dashboard air tanah

Dashboard ini melanjutkan notebook 01–05. Dashboard menggunakan hasil proyekmu sendiri; paket ini tidak mengganti data atau model dengan hasil orang lain.

## 1. Tempatkan file

Ekstrak ZIP, lalu salin **isi** folder hasil ekstraksi ke folder proyek `groundwater_forecasting`:

- `app.py`
- `dashboard_utils.py`
- `requirements_dashboard.txt`
- folder `.streamlit` beserta `config.toml`
- `PANDUAN_DASHBOARD.md` dan `README_DASHBOARD.md`

Posisi `app.py` harus langsung sejajar dengan folder `notebooks`, `data`, `models`, dan `outputs`. Jangan taruh di dalam `notebooks`. Jika sudah ada `app.py`, simpan versi lama dengan nama berbeda terlebih dahulu.

## 2. Pastikan notebook 05 selesai

Harus tersedia `outputs/evaluation/evaluation_summary.json`, `evaluation_record.json`, dan `prediksi_holdout.csv`, bersama hasil notebook 03 dan model dari notebook 04. Jika aplikasi menyebut file belum ditemukan, selesaikan notebook yang bersangkutan; jangan membuat file kosong sebagai pengganti.

## 3. Gunakan Python yang sama

Di satu cell notebook yang memakai kernel 04/05, jalankan:

```python
import sys
print(sys.executable)
```

Salin alamat Python yang muncul. Di terminal PowerShell VS Code, masuk ke folder proyek (sesuaikan jika lokasi proyek sudah dipindah):

```powershell
cd "C:\Users\MSI MODERN\Downloads\Rasen_Portofolio\02_Projects\groundwater_forecasting"
```

Lalu jalankan dua perintah berikut satu per satu. Ganti teks `ALAMAT_PYTHON_DARI_NOTEBOOK` dengan alamat hasil cell tadi, tetap gunakan tanda kutip dan `&`:

```powershell
& "ALAMAT_PYTHON_DARI_NOTEBOOK" -m pip install -r requirements_dashboard.txt
& "ALAMAT_PYTHON_DARI_NOTEBOOK" -m streamlit run app.py
```

Jika `python` di terminal sudah menunjuk interpreter yang sama, boleh gunakan bentuk singkat:

```powershell
python -m pip install -r requirements_dashboard.txt
python -m streamlit run app.py
```

Ini tidak mengharuskan pindah versi Python. File model scikit-learn perlu dimuat dengan versi yang sama seperti saat dibuat. Aplikasi mengecek versi ketika fitur prediksi dipakai.

## 4. Buka di browser

Streamlit menampilkan `Local URL`, biasanya `http://localhost:8501`. Buka alamat tersebut jika browser tidak terbuka otomatis. Biarkan terminal tetap berjalan. Untuk menghentikan aplikasi, tekan `Ctrl+C` pada terminal.

Menjalankan localhost belum memublikasikan website. Tahap publikasi GitHub/hosting dikerjakan terpisah setelah aplikasi lokal diperiksa.

## 5. Menu yang tersedia

- **Ringkasan:** tujuan proyek, skor holdout, dan grafik utama.
- **Data historis:** filter rentang tahun, head, hujan, evaporasi, dan unduhan CSV.
- **Evaluasi model:** filter tahun, pembanding pada tanggal sama, dan kesalahan terbesar.
- **Coba prediksi:** pilih riwayat proyek atau unggah CSV mingguan dengan format yang sama. Fitur dihitung sampai tanggal prediksi saja. Tidak ada pelatihan ulang.
- **Tentang proyek:** alur notebook, sumber data, dan batasan.

CSV unggahan bukan CSV mentah harian. Gunakan keluaran mingguan sesuai notebook 01, satuan yang sama, dan label tanggal Minggu. Kalender yang hilang tetap menjadi gap; tidak diinterpolasi. File unggahan tidak ditulis ke folder proyek.

## Jika ada kendala

- `No module named streamlit/plotly`: instal dengan alamat Python yang muncul dari notebook.
- File hasil tidak ditemukan: periksa posisi `app.py` dan selesaikan notebook 01–05.
- Model/data tidak cocok dengan evaluasi: jangan mencampur hasil eksperimen berbeda. Periksa catatan evaluasi dan hasil notebook 03–05.
- Prediksi mengatakan riwayat belum cukup: periksa minimal delapan minggu cuaca lengkap dan riwayat head yang dibutuhkan. Jangan mengisi target kosong dengan tebakan untuk melewati pengecekan.
- Tampilan belum memperbarui hasil: gunakan tombol **Muat ulang hasil notebook**.

File ini adalah panduan lokal. README portofolio akhir dan langkah publikasi akan menyesuaikan hasil yang benar-benar muncul di laptopmu.
