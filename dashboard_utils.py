"""Pembacaan hasil notebook dan fitur prediksi; tidak melatih model."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

WEATHER = ["rainfall_mm", "temperature_mean_c", "temperature_min_c", "temperature_max_c",
    "sea_level_pressure_hpa", "relative_humidity_pct", "wind_speed_ms",
    "global_radiation_wm2", "potential_evaporation_mm"]
REQUIRED = ["data/processed/netherlands_weekly.csv", "data/processed/feature_schema.json",
    "data/processed/netherlands_holdout.csv", "models/groundwater_ml_candidate.joblib",
    "models/model_metadata.json", "outputs/evaluation/evaluation_summary.json",
    "outputs/evaluation/evaluation_record.json", "outputs/evaluation/prediksi_holdout.csv"]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def normalize_weekly(frame):
    """Validasi satuan/struktur mingguan; lengkapi kalender tanpa interpolasi."""
    frame = frame.copy()
    required = ["date", "groundwater_head_m", "head_obs_count", "weather_obs_count", "week_valid", *WEATHER]
    missing = set(required)-set(frame.columns)
    if missing:
        raise ValueError("Kolom belum lengkap: " + ", ".join(sorted(missing)))
    if frame.empty:
        raise ValueError("CSV tidak mempunyai baris data.")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame["date"].isna().any() or frame["date"].duplicated().any():
        raise ValueError("Tanggal kosong atau duplikat ditemukan.")
    if frame["date"].dt.tz is not None:
        raise ValueError("Gunakan tanggal tanpa zona waktu: YYYY-MM-DD.")
    if not frame["date"].eq(frame["date"].dt.normalize()).all() or not frame["date"].dt.dayofweek.eq(6).all():
        raise ValueError("Tanggal harus berupa tanggal akhir minggu (Minggu), tanpa komponen jam.")
    numeric = ["groundwater_head_m", "head_obs_count", "weather_obs_count", *WEATHER]
    for col in numeric:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    if np.isinf(frame[numeric].to_numpy(dtype=float)).any():
        raise ValueError("Ada nilai tak hingga pada CSV.")
    counts = frame[["head_obs_count", "weather_obs_count"]]
    if counts.isna().any().any() or not counts.ge(0).all().all() or not counts.le(7).all().all() or not counts.eq(counts.round()).all().all():
        raise ValueError("Jumlah hari pengamatan harus bilangan bulat 0–7.")
    flags = frame["week_valid"].astype(str).str.strip().str.lower()
    if not flags.isin(["true", "false"]).all():
        raise ValueError("week_valid harus True atau False.")
    frame["week_valid"] = flags.eq("true")
    rule = frame["head_obs_count"].ge(4) & frame["weather_obs_count"].eq(7)
    if not frame["week_valid"].eq(rule).all():
        raise ValueError("week_valid tidak sesuai aturan: head minimal 4 hari dan cuaca 7 hari.")
    if frame.loc[rule, ["groundwater_head_m", *WEATHER]].isna().any().any():
        raise ValueError("Minggu layak masih mempunyai nilai kosong.")
    if frame.loc[~rule, "groundwater_head_m"].notna().any():
        raise ValueError("Head pada minggu tidak layak harus kosong, sesuai notebook 01.")
    frame = frame.set_index("date").sort_index()
    span = (frame.index.max()-frame.index.min()).days
    if span > 36525:
        raise ValueError("Rentang data melebihi 100 tahun; periksa penulisan tanggal.")
    frame = frame.reindex(pd.date_range(frame.index.min(), frame.index.max(), freq="W-SUN", name="date"))
    frame["week_valid"] = frame["week_valid"].eq(True)
    frame.loc[~frame["weather_obs_count"].eq(7), WEATHER] = np.nan
    return frame

def make_features(table):
    """Sama dengan buat_fitur() notebook 03; tidak membaca target atau masa depan."""
    features = pd.DataFrame(index=table.index)
    head = table["groundwater_head_m"]
    features["head_t"] = head
    for lag in [1, 2, 4]:
        features[f"head_lag_{lag}w"] = head.shift(lag)
    features["head_mean_4w"] = head.rolling(4, min_periods=4).mean()
    features["head_change_1w"] = head.diff()
    for col in WEATHER:
        features[f"{col}_t"] = table[col]
    for size in [4, 8]:
        features[f"rainfall_sum_{size}w"] = table["rainfall_mm"].rolling(size, min_periods=size).sum()
        features[f"evaporation_sum_{size}w"] = table["potential_evaporation_mm"].rolling(size, min_periods=size).sum()
    days = np.where(table.index.is_leap_year, 366, 365)
    phase = 2*np.pi*(table.index.dayofyear-1)/days
    features["season_sin"] = np.sin(phase)
    features["season_cos"] = np.cos(phase)
    return features

def metrics(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if not len(actual) or len(actual) != len(predicted) or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Pasangan nilai aktual dan prediksi tidak valid.")
    error = actual-predicted
    denominator = np.square(actual-actual.mean()).sum()
    return {"n": len(actual), "MAE_m": float(np.abs(error).mean()),
        "RMSE_m": float(np.sqrt(np.square(error).mean())),
        "R2": float(1-np.square(error).sum()/denominator) if denominator>0 and len(actual)>1 else np.nan}

def load_project(root):
    root = Path(root)
    missing = [p for p in REQUIRED if not (root/p).is_file()]
    if missing:
        raise FileNotFoundError("Belum ditemukan: " + ", ".join(missing))
    schema = read_json(root/REQUIRED[1])
    meta = read_json(root/"models/model_metadata.json")
    summary = read_json(root/"outputs/evaluation/evaluation_summary.json")
    record = read_json(root/"outputs/evaluation/evaluation_record.json")
    if not summary.get("holdout_evaluated") or record.get("status") != "completed":
        raise ValueError("Evaluasi belum selesai. Jalankan notebook 05 sampai cell terakhir.")
    identity = summary["identity"]
    checks = {
        "model_sha256": root/"models/groundwater_ml_candidate.joblib",
        "holdout_sha256": root/"data/processed/netherlands_holdout.csv",
        "schema_sha256": root/"data/processed/feature_schema.json"}
    if identity != record["identity"] or any(digest(path) != identity[key] for key,path in checks.items()):
        raise ValueError("Model/data berbeda dari evaluasi tersimpan. Cocokkan hasil notebook 03–05; hasil lama tidak boleh digabung dengan model baru.")
    if digest(root/"data/processed/netherlands_weekly.csv") != schema["source_sha256"]:
        raise ValueError("Data mingguan berubah sejak feature engineering.")
    if schema != meta["feature_schema"] or meta["feature_columns"] != summary["feature_columns"] or meta["experiment_id"] != summary["experiment_id"]:
        raise ValueError("Metadata model tidak cocok dengan evaluasi.")
    weekly = normalize_weekly(pd.read_csv(root/"data/processed/netherlands_weekly.csv"))
    predicted = pd.read_csv(root/"outputs/evaluation/prediksi_holdout.csv", parse_dates=["forecast_date","target_date"])
    if predicted["target_date"].duplicated().any() or len(predicted) != summary["n_holdout"]:
        raise ValueError("Jumlah/tanggal prediksi tidak sesuai ringkasan evaluasi.")
    predicted = predicted.sort_values("target_date").reset_index(drop=True)
    if not (predicted["target_date"]-predicted["forecast_date"]).eq(pd.Timedelta(weeks=1)).all():
        raise ValueError("Horizon prediksi tidak sesuai.")
    for name,col in [("Kandidat ML terkunci","prediksi_ml_m"),("Persistence","baseline_persistence")]:
        calculated = metrics(predicted["aktual_head_m"],predicted[col])
        expected = next(row for row in summary["metrics_main"] if row["metode"]==name)
        if not np.isclose(calculated["MAE_m"],expected["MAE_m"],atol=1e-10,rtol=1e-8):
            raise ValueError("CSV prediksi tidak cocok dengan skor tersimpan.")
    return weekly, predicted, summary, meta, schema
