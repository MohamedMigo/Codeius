# train_pro.py — Build dataset from NASA POWER (last 30d, multiple cities),
# feature engineering, train an ENSEMBLE (RF + GBDT -> Logistic meta),
# calibrate probabilities, export models/risk_model.pkl

import datetime as dt
import requests, os, pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance

TIMEOUT = 25

CITIES = [
    {"name":"Assiut",     "lat":27.1800, "lon":31.1837},
    {"name":"Cairo",      "lat":30.0444, "lon":31.2357},
    {"name":"Alexandria", "lat":31.2001, "lon":29.9187},
    {"name":"Dubai",      "lat":25.2048, "lon":55.2708},
    {"name":"London",     "lat":51.5074, "lon":-0.1278},
]

def fetch_power_hourly(lat: float, lon: float, days: int = 30) -> pd.DataFrame:
    base = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    end_dt   = dt.datetime.utcnow()
    start_dt = end_dt - dt.timedelta(days=days)
    params = {
        "parameters": "T2M,WS10M,PRECTOTCORR",
        "community": "re",
        "latitude":  lat,
        "longitude": lon,
        "start":     start_dt.strftime("%Y%m%d"),
        "end":       end_dt.strftime("%Y%m%d"),
        "format":    "JSON",
    }
    r = requests.get(base, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()["properties"]["parameter"]
    hours = list(next(iter(data.values())).keys())
    rows = []
    for h in hours:
        rows.append({
            "ts":      pd.to_datetime(h, format="%Y%m%d%H", errors="coerce"),
            "t2m":     data["T2M"].get(h, np.nan),
            "ws10m":   data["WS10M"].get(h, np.nan),
            "prec_mm": data["PRECTOTCORR"].get(h, 0.0),  # mm/hr
        })
    df = pd.DataFrame(rows).dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Proxies محاذية لخصائص التطبيق الحية
    df["precip_probability"] = np.clip(df["prec_mm"]*15.0, 0, 100)  # proxy prob%
    df["nowcast_mmph"]       = np.clip(df["prec_mm"], 0, 6)         # nowcast proxy
    df["wind10m"]            = pd.to_numeric(df["ws10m"], errors="coerce").fillna(0)
    # Temporal + Lags
    df["hour"] = df["ts"].dt.hour
    df["dow"]  = df["ts"].dt.dayofweek
    df["lag1_mm"]  = df["prec_mm"].shift(1).fillna(0)
    df["lag2_mm"]  = df["prec_mm"].shift(2).fillna(0)
    df["roll3_mm"] = df["prec_mm"].rolling(3, min_periods=1).mean()
    df["roll6_mm"] = df["prec_mm"].rolling(6, min_periods=1).mean()

    # Labels: green/yellow/red من mm/hr
    cond_green  = df["prec_mm"] < 0.1
    cond_yellow = (df["prec_mm"] >= 0.1) & (df["prec_mm"] < 1.0)
    df["risk_band"] = np.where(cond_green, "green", np.where(cond_yellow, "yellow", "red"))

    feats = [
        "precip_probability","prec_mm","wind10m","nowcast_mmph",
        "hour","dow","lag1_mm","lag2_mm","roll3_mm","roll6_mm"
    ]
    return df[["ts"] + feats + ["risk_band"]].copy()

def build_corpus() -> pd.DataFrame:
    frames = []
    for c in CITIES:
        try:
            raw = fetch_power_hourly(c["lat"], c["lon"], days=30)
            eng = engineer_features(raw)
            eng["city"] = c["name"]
            frames.append(eng)
            print(f"OK {c['name']}: {len(eng)} rows")
        except Exception as e:
            print(f"FAIL {c['name']}: {e}")
    if not frames:
        raise RuntimeError("No data from NASA POWER.")
    df = pd.concat(frames, ignore_index=True).dropna().reset_index(drop=True)
    return df

def train_ensemble(df: pd.DataFrame):
    feats = [
        "precip_probability","prec_mm","wind10m","nowcast_mmph",
        "hour","dow","lag1_mm","lag2_mm","roll3_mm","roll6_mm"
    ]
    X = df[feats].values
    y = df["risk_band"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.18, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=350, min_samples_split=4, min_samples_leaf=2,
        class_weight="balanced_subsample", random_state=42
    )
    gb = GradientBoostingClassifier(random_state=42)

    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)

    # Meta features = probs من RF & GB
    def probs(model, X):
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X)
            # تأكد الترتيب aligned
            return p
        # fallback
        # one-hot pseudo
        lab = model.predict(X)
        classes = np.unique(y_train)
        arr = np.zeros((len(lab), len(classes)))
        for i,cl in enumerate(classes):
            arr[:,i] = (lab==cl).astype(float)
        return arr

    # ترتيب الفئات
    classes_ = ['green','yellow','red']
    # أعد ترتيب أعمدة الاحتمالات حسب classes_
    def order_cols(p, model_classes):
        idx = [list(model_classes).index(c) for c in classes_]
        return p[:, idx]

    p_rf_train = order_cols(rf.predict_proba(X_train), rf.classes_)
    p_gb_train = order_cols(gb.predict_proba(X_train), gb.classes_)
    meta_train = np.hstack([p_rf_train, p_gb_train])

    p_rf_test = order_cols(rf.predict_proba(X_test), rf.classes_)
    p_gb_test = order_cols(gb.predict_proba(X_test), gb.classes_)
    meta_test = np.hstack([p_rf_test, p_gb_test])

    meta = LogisticRegression(max_iter=500, multi_class="ovr")
    meta.fit(meta_train, y_train)

    # Calibration على الميتا
    meta_cal = CalibratedClassifierCV(meta, cv=3, method="isotonic")
    meta_cal.fit(meta_train, y_train)

    # تقييم
    y_pred = meta_cal.predict(meta_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    print("Ensemble Accuracy:", round(acc,3), "| F1-macro:", round(f1m,3))
    print(classification_report(y_test, y_pred))

    # Feature importance تقريبية (Permutation على RF كتفسير سريع)
    pi = permutation_importance(rf, X_test, y_test, n_repeats=8, random_state=42)
    rf_pi = pd.DataFrame({"feature": feats, "importance": pi.importances_mean}).sort_values("importance", ascending=False)

    payload = {
        "features": feats,
        "classes": classes_,
        "rf": rf,
        "gb": gb,
        "meta_cal": meta_cal,
        "rf_perm_importance": rf_pi
    }
    return payload

def main():
    os.makedirs("models", exist_ok=True)
    df = build_corpus()
    payload = train_ensemble(df)

    with open("models/risk_model.pkl","wb") as f:
        pickle.dump(payload, f)
    print("Saved models/risk_model.pkl")

    df.sample(600, random_state=42).to_csv("models/training_sample.csv", index=False)
    print("Saved models/training_sample.csv")

if __name__ == "__main__":
    main()
