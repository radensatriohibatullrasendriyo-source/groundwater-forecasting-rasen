from pathlib import Path
import html
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from dashboard_utils import REQUIRED, load_project, make_features, metrics, normalize_weekly

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Groundwater Lab | Rasen", page_icon="💧", layout="wide")
st.markdown("""<style>
.block-container{max-width:1280px;padding-top:2rem;padding-bottom:3rem}
.hero{padding:26px 28px;border:1px solid #284b59;border-radius:18px;
background:linear-gradient(115deg,#102c38,#163d45);margin-bottom:24px}
.eyebrow{color:#8bdbd0;letter-spacing:2px;font-size:12px;text-transform:uppercase}
.hero h1{font-size:38px;color:#f5f7f8;margin:8px 0}
.hero p{color:#c3d6db;max-width:800px;margin:0;line-height:1.65}
[data-testid="stMetric"]{border:1px solid #284b59;border-radius:12px;padding:16px}
</style>""", unsafe_allow_html=True)

def snapshot():
    return tuple((p, (ROOT/p).stat().st_mtime_ns, (ROOT/p).stat().st_size)
        for p in REQUIRED if (ROOT/p).is_file())

@st.cache_data(show_spinner=False)
def read_project(signature):
    return load_project(ROOT)

@st.cache_resource(show_spinner=False)
def read_model(model_hash, expected_version):
    import sklearn
    import joblib
    if sklearn.__version__ != expected_version:
        raise ValueError(f"Prediksi memerlukan kernel scikit-learn {expected_version}, sama seperti saat model dibuat. Versi aktif: {sklearn.__version__}.")
    # Hanya muat model lokal hasil notebook 04, bukan unggahan pengunjung.
    return joblib.load(ROOT/"models/groundwater_ml_candidate.joblib")["pipeline"]

def line_chart(frame, columns, title, ylabel):
    fig=go.Figure()
    for col,label,color in columns:
        fig.add_trace(go.Scatter(x=frame.index,y=frame[col],name=label,mode="lines",
            connectgaps=False,line=dict(color=color,width=2)))
    fig.update_layout(title=title,yaxis_title=ylabel,xaxis_title="Tanggal akhir minggu",
        template="plotly_dark",height=370,margin=dict(l=12,r=12,t=50,b=20),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h",y=1.12),hovermode="x unified")
    return fig

def download_csv(frame,name,label="Unduh tabel CSV"):
    st.download_button(label,frame.to_csv(index=False).encode("utf-8"),name,"text/csv")

st.sidebar.markdown("## 💧 Groundwater Lab")
st.sidebar.caption("Machine Learning & AI for Beginner · Batch 25")
page=st.sidebar.radio("Navigasi",["Ringkasan","Data historis","Evaluasi model","Coba prediksi","Tentang proyek"])
st.sidebar.divider()
st.sidebar.caption("Studi kasus Drenthe, Belanda\n\nPrediksi head satu minggu ke depan")
if st.sidebar.button("Muat ulang hasil notebook"):
    st.cache_data.clear()
    st.cache_resource.clear()
try:
    weekly,predicted,summary,meta,schema=read_project(snapshot())
except (FileNotFoundError,ValueError,KeyError,AssertionError) as exc:
    st.title("Siapkan hasil notebook terlebih dahulu")
    st.info("Letakkan app.py dan dashboard_utils.py langsung di folder groundwater_forecasting, sejajar dengan folder notebooks, data, models, dan outputs. Jalankan notebook 01–05 sampai selesai.")
    st.error(str(exc))
    st.code("groundwater_forecasting/app.py\ngroundwater_forecasting/dashboard_utils.py\ngroundwater_forecasting/notebooks/\ngroundwater_forecasting/data/\ngroundwater_forecasting/models/\ngroundwater_forecasting/outputs/")
    st.stop()

ml=next(m for m in summary["metrics_main"] if m["metode"]=="Kandidat ML terkunci")
baseline=next(m for m in summary["metrics_main"] if m["metode"]=="Persistence")
skill=100*(1-ml["MAE_m"]/baseline["MAE_m"]) if baseline["MAE_m"]>0 else np.nan

if page=="Ringkasan":
    st.markdown('''<div class="hero"><div class="eyebrow">Computational geophysics · Machine learning</div>
    <h1>Membaca pola air tanah.</h1><p>Eksplorasi data historis dan prediksi rata-rata muka air tanah satu minggu ke depan.
    Dari pengamatan mingguan menuju evaluasi model yang bisa ditelusuri.</p></div>''',unsafe_allow_html=True)
    st.caption("PROTOTIPE HISTORIS · Bukan pemantauan real-time")
    cols=st.columns(4)
    cols[0].metric("MAE model · holdout",f"{ml['MAE_m']*100:.2f} cm")
    cols[1].metric("MAE persistence",f"{baseline['MAE_m']*100:.2f} cm")
    cols[2].metric("Pengurangan MAE",f"{skill:.2f}%" if np.isfinite(skill) else "—")
    cols[3].metric("Minggu yang diuji",str(summary["n_holdout"]))
    st.caption("MAE adalah rata-rata kesalahan absolut. Pengurangan MAE bukan persentase akurasi; nilai negatif berarti ML lebih buruk daripada baseline.")
    dates=predicted.set_index("target_date").reindex(pd.date_range(predicted.target_date.min(),predicted.target_date.max(),freq="W-SUN"))
    st.plotly_chart(line_chart(dates,[("aktual_head_m","Aktual","#81d8cb"),("prediksi_ml_m","Prediksi ML","#ffbb6a")],
        "Aktual dan prediksi pada holdout","Hydraulic head (m)"),width="stretch")
    left,right=st.columns([1.2,1])
    with left:
        st.subheader("Pertanyaan proyek")
        st.write("Apakah riwayat air tanah dan cuaca memberi prediksi minggu depan yang lebih baik daripada memakai nilai minggu ini?")
        st.write("Model dipilih menggunakan validasi waktu pada data pengembangan. Periode mulai 2017 disisihkan sebagai holdout dan baru dinilai setelah model dikunci.")
    with right:
        st.subheader("Model yang diuji")
        st.write(f"**Algoritma:** {meta['model_family']}")
        st.write(f"**Kelompok fitur:** {meta['feature_group']} · {len(meta['feature_columns'])} fitur")
        st.write(f"**Periode target uji:** {summary['evaluated_target_start']} — {summary['evaluated_target_end']}")
        st.caption("Head adalah elevasi terhadap acuan dataset, bukan kedalaman langsung di bawah tanah.")

elif page=="Data historis":
    st.title("Data historis")
    st.write("Lihat perubahan air tanah dan cuaca. Celah pengamatan tetap kosong; grafik tidak mengisi nilai yang tidak diamati.")
    years=sorted(weekly.index.year.unique().tolist())
    lo,hi=st.select_slider("Rentang tahun",options=years,value=(years[0],years[-1]))
    subset=weekly.loc[(weekly.index.year>=lo)&(weekly.index.year<=hi)]
    c=st.columns(3)
    c[0].metric("Minggu kalender",len(subset))
    c[1].metric("Minggu layak",int(subset.week_valid.sum()))
    c[2].metric("Minggu tidak layak",int((~subset.week_valid).sum()))
    st.plotly_chart(line_chart(subset,[("groundwater_head_m","Head","#81d8cb")],"Air tanah mingguan","Rata-rata head (m)"),width="stretch")
    st.plotly_chart(line_chart(subset,[("rainfall_mm","Hujan","#7cb8ef"),("potential_evaporation_mm","Evaporasi potensial","#ffbb6a")],
        "Cuaca mingguan","Total mingguan (mm)"),width="stretch")
    st.caption("Minggu layak: minimal 4 hari observasi head dan 7 hari cuaca lengkap. Cuaca minggu tidak lengkap disembunyikan agar total parsial tidak dibandingkan dengan total satu minggu.")
    with st.expander("Lihat dan unduh data dalam rentang ini"):
        st.dataframe(subset.reset_index(),width="stretch")
        download_csv(subset.reset_index(),"data_historis_terpilih.csv")

elif page=="Evaluasi model":
    st.title("Evaluasi model")
    st.write("Hasil uji pada periode yang tidak dipakai untuk memilih algoritma atau tuning. Semua perbandingan di bawah menyamakan tanggal antar-metode.")
    options=["Semua tahun"]+[str(y) for y in sorted(predicted.target_date.dt.year.unique())]
    year=st.selectbox("Periode target yang ditampilkan",options)
    subset=predicted if year=="Semua tahun" else predicted.loc[predicted.target_date.dt.year.eq(int(year))]
    rows=[]
    for name,col in [("Kandidat ML","prediksi_ml_m"),("Persistence","baseline_persistence")]:
        rows.append({"Metode":name,**metrics(subset.aktual_head_m,subset[col])})
    table=pd.DataFrame(rows)
    st.dataframe(table.round(4),width="stretch",hide_index=True)
    st.caption(f"Cakupan tabel: {len(subset)} minggu pada pilihan {year.lower()}. MAE dan RMSE dalam meter; R² bukan akurasi.")
    common=subset.dropna(subset=["baseline_seasonal_52w"])
    with st.expander("Bandingkan juga dengan seasonal naïve"):
        st.write(f"Riwayat musiman tersedia pada {len(common)} dari {len(subset)} tanggal.")
        if len(common):
            seasonal=pd.DataFrame([{"Metode":name,**metrics(common.aktual_head_m,common[col])}
                for name,col in [("Kandidat ML","prediksi_ml_m"),("Persistence","baseline_persistence"),("Seasonal naïve 52 minggu","baseline_seasonal_52w")]])
            st.dataframe(seasonal.round(4),width="stretch",hide_index=True)
            st.caption("Ketiga metode di tabel ini memakai irisan tanggal yang sama. Cakupannya dapat berbeda dari tabel utama.")
    full=subset.set_index("target_date").reindex(pd.date_range(subset.target_date.min(),subset.target_date.max(),freq="W-SUN"))
    st.plotly_chart(line_chart(full,[("aktual_head_m","Aktual","#81d8cb"),("prediksi_ml_m","Prediksi ML","#ffbb6a"),("baseline_persistence","Persistence","#7cb8ef")],
        "Perbandingan prediksi","Head (m)"),width="stretch")
    st.subheader("Kesalahan terbesar dalam periode terpilih")
    st.dataframe(subset.nlargest(10,"absolute_error_ml_m")[["target_date","aktual_head_m","prediksi_ml_m","absolute_error_ml_m","absolute_error_persistence_m"]].round(4),width="stretch",hide_index=True)
    download_csv(subset,"prediksi_holdout_terpilih.csv")
    with st.expander("Kesimpulan lengkap dari notebook 05"):
        for item in summary["conclusion"]: st.write("• "+item)

elif page=="Coba prediksi":
    st.title("Coba prediksi satu minggu ke depan")
    st.write("Pilih akhir minggu pengamatan. Aplikasi menghitung fitur dari riwayat sampai tanggal tersebut, lalu memprediksi rata-rata head minggu berikutnya.")
    mode=st.radio("Sumber riwayat",["Dataset proyek","Unggah data mingguan"],horizontal=True)
    table=weekly
    if mode=="Unggah data mingguan":
        st.info("Gunakan struktur CSV hasil notebook 01, dengan satuan dan acuan elevasi yang sama. Minimal 8 minggu kalender berurutan diperlukan untuk fitur cuaca. Data lokasi lain belum tervalidasi untuk model ini.")
        sample=weekly.tail(12).reset_index()
        st.download_button("Unduh contoh format terisi",sample.to_csv(index=False).encode(),"contoh_riwayat_mingguan.csv","text/csv")
        upload=st.file_uploader("CSV mingguan, maksimal 5 MB",type=["csv"])
        if upload is None: st.stop()
        if upload.size>5*1024*1024:
            st.error("Ukuran CSV melebihi 5 MB."); st.stop()
        try: table=normalize_weekly(pd.read_csv(upload))
        except (ValueError,TypeError,KeyError) as exc:
            st.error(str(exc)); st.stop()
        st.caption("CSV diproses dalam sesi aplikasi; tidak ditulis ke folder data proyek.")
    features=make_features(table)
    common_features=schema["feature_groups"]["head_weather_season"]
    valid=features[common_features].notna().all(axis=1) & table.week_valid
    valid &= table.index > pd.Timestamp(meta["fit_target_end"])
    choices=features.index[valid]
    if len(choices)==0:
        st.info("Belum ada minggu yang memiliki riwayat lengkap setelah periode latihan. Periksa gap, kelengkapan head, dan delapan minggu cuaca."); st.stop()
    date=st.selectbox("Akhir minggu pengamatan",choices.tolist(),index=len(choices)-1,
        format_func=lambda x:x.strftime("%d %B %Y"))
    date=pd.Timestamp(date)
    target=date+pd.Timedelta(weeks=1)
    st.caption(f"Riwayat berhenti pada {date.date()}. Target: rata-rata minggu { (date+pd.Timedelta(days=1)).date() } sampai {target.date()}.")
    if st.button("Hitung prediksi",type="primary"):
        try:
            pipeline=read_model(summary["identity"]["model_sha256"],meta["versions"]["sklearn"])
            # Recompute only the past prefix as an extra safeguard against using future rows.
            past=make_features(table.loc[:date])
            row=past.loc[[date],meta["feature_columns"]]
            value=float(pipeline.predict(row)[0])
            if not np.isfinite(value): raise ValueError("Model menghasilkan nilai tidak terhingga.")
        except (ValueError,ImportError,KeyError) as exc:
            st.error(str(exc)); st.stop()
        persist=float(table.loc[date,"groundwater_head_m"])
        columns=st.columns(3)
        columns[0].metric("Prediksi ML · head",f"{value:.3f} m")
        columns[1].metric("Persistence · head",f"{persist:.3f} m")
        columns[2].metric("Perubahan prediksi dari saat ini",f"{(value-persist)*100:+.2f} cm")
        actual=table.loc[target,"groundwater_head_m"] if target in table.index else np.nan
        if pd.notna(actual):
            st.write(f"Observasi target sudah ada di arsip: **{actual:.3f} m**. Kesalahan absolut ML: **{abs(value-actual)*100:.2f} cm**.")
        else: st.info("Observasi target belum tersedia atau tidak layak pada riwayat ini. Kesalahan prediksi belum dapat dihitung.")
        st.caption("Tidak ada interval ketidakpastian yang dikalibrasi. Prediksi ini bukan status aman/bahaya dan tidak mengendalikan dewatering.")
        out=pd.DataFrame([{"forecast_date":date.date(),"target_date":target.date(),"prediksi_ml_m":value,"baseline_persistence":persist,"aktual_head_m":actual}])
        download_csv(out,"hasil_prediksi_satu_minggu.csv","Unduh hasil prediksi")
        with st.expander("Lihat input yang dipakai model"):
            st.dataframe(row.T.rename(columns={date:"Nilai fitur"}),width="stretch")

else:
    st.title("Tentang proyek")
    st.write("This project was developed as the final project for the Machine Learning & AI for Beginner Bootcamp, Batch 25. It was created by Raden Satrio Hibatull Rasendriyo, a physics student interested in computational geophysics and data analysis.")
    st.markdown("**Contact**  \nEmail: [radensatriohibatullrasendriyo@gmail.com](mailto:radensatriohibatullrasendriyo@gmail.com)  \nLinkedIn: [linkedin.com/in/raden-satrio](https://www.linkedin.com/in/raden-satrio)")
    st.markdown("**Project links**  \n[GitHub repository](https://github.com/radensatriohibatullrasendriyo-source/groundwater-forecasting-rasen) · [Live Streamlit dashboard](https://groundwaterforecasting-rasen.streamlit.app/)")
    st.subheader("Dari data menjadi hasil yang bisa diperiksa")
    st.dataframe(pd.DataFrame([
        ["01","Problem identification & data understanding","Memeriksa sumber, tanggal hilang, dan agregasi mingguan"],
        ["02","EDA","Pola dan hubungan pada data pengembangan"],
        ["03","Feature engineering","Riwayat head/cuaca, target t+1, dan baseline"],
        ["04","Modeling & tuning","8 algoritma, 3 kelompok fitur, validasi waktu"],
        ["05","Evaluasi akhir","Kandidat terkunci diuji pada holdout"]],columns=["Notebook","Tahap","Kegiatan"]),hide_index=True,width="stretch")
    st.subheader("Sumber dan batasan")
    st.markdown("[Dataset dan kode challenge](https://github.com/gwmodeling/challenge) · [Dokumentasi Netherlands](https://github.com/gwmodeling/challenge/blob/main/data/Netherlands/README.md) · [Publikasi Collenteur dkk. (2024)](https://doi.org/10.5194/hess-28-5193-2024)")
    st.write("Dataset merupakan paket riset yang sudah disiapkan penyedia. Proyek ini memakai rancangan prediksi mingguan sendiri, bukan mereplikasi aturan kompetisi asal.")
    st.write("Evaluasi hanya mencakup satu lokasi di Belanda. Data cuaca berupa arsip; ketersediaan tepat waktu untuk operasi real-time belum diperiksa. Penerapan pada sumur lain atau di Indonesia memerlukan pengujian lokal.")
    st.write("Tidak ada ambang toleransi operasional yang ditetapkan. Dashboard ini membantu menjelaskan eksperimen prediksi; tidak menyimpulkan keamanan lereng atau kondisi geoteknik tambang.")
    st.caption("Versi model: "+html.escape(meta["experiment_id"])+" · Fitur: "+str(len(meta["feature_columns"])))

st.divider()
st.caption("GROUNDWATER LAB / RASEN · Data historis, model terkunci, hasil yang dapat ditelusuri.")
