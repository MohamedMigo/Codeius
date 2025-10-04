# app.py — ParadeGuard v9 (Show-stopper UI + Ensemble AI)
import datetime as dt
import requests, os, pickle
import pandas as pd
import numpy as np
import streamlit as st

# Optional maps
try:
    import folium
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except Exception:
    HAS_FOLIUM = False

st.set_page_config(page_title="ParadeGuard — NASA AI", layout="wide")
TIMEOUT = 20

# ===== Ultra UI (glass hero + gauge + chips) =====
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
<style>
:root{ --bg:#0b1220; --card:#ffffff; --muted:#64748B; --line:#E2E8F0; --accent:#0EA5E9; --ok:#10B981; --warn:#F59E0B; --danger:#EF4444;}
html, body, [class*="css"] { font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
.block-container{max-width:1200px;}
.pg-hero{
  margin:-1rem -1rem 16px -1rem; padding:24px; color:#e8f4ff;
  background: radial-gradient(1200px 400px at 10% -10%, #15a0f533, transparent 60%),
              radial-gradient(1200px 500px at 110% -20%, #22d3ee22, transparent 60%),
              linear-gradient(180deg, #0c1526 0%, #0b1220 60%, #0b1220 100%);
  border-bottom:1px solid #0b223d; position:relative; overflow:hidden;
}
.pg-hero:after{
  content:""; position:absolute; inset:-30% -20% auto auto; width:60%; height:140%;
  background: radial-gradient(50% 50% at 50% 50%, rgba(14,165,233,.20), transparent 60%);
  filter: blur(40px); pointer-events:none;
}
.pg-hero .title{ font-weight:900; font-size:26px; display:flex; gap:10px; align-items:center;}
.pg-hero .sub{ color:#cfe7ff; font-size:13px; margin-top:4px;}
.pg-card{ background:var(--card); border:1px solid var(--line); border-radius:16px; padding:14px 16px; box-shadow:0 10px 22px rgba(2,6,23,0.06);}
.pg-stat .k{font-size:12px;color:var(--muted);margin-bottom:4px;}
.pg-stat .v{font-size:22px;font-weight:800;}
.pg-chip{ display:inline-flex; gap:6px; align-items:center; padding:5px 11px; border-radius:999px; font-size:12px; font-weight:700; border:1px solid var(--line); background:#F8FAFC; color:#0F172A;}
.pg-chip.ok{ border-color:#10B98133; background:#10B9811a; color:#10B981;}
.pg-chip.warn{ border-color:#F59E0B33; background:#F59E0B1a; color:#F59E0B;}
.pg-chip.danger{ border-color:#EF444433; background:#EF44441a; color:#EF4444;}
.pg-section{ font-weight:900; letter-spacing:.2px; margin:12px 0 6px; display:flex; align-items:center; gap:8px; color:#0F172A;}
.pg-hour{ text-align:center; }
.pg-hour .t{color:var(--muted);font-size:12px;} .pg-hour .s{font-weight:900;font-size:20px;} .pg-hour .b{color:var(--muted);font-size:11px;}
.pg-gauge{ --size:128px; width:var(--size); height:var(--size); border-radius:50%;
  background: conic-gradient(#10B981 var(--p,0%), #F59E0B 0 var(--m,0%), #EF4444 0 100%), radial-gradient(closest-side, #fff 72%, transparent 73% 100%);
  display:grid; place-items:center; box-shadow:0 10px 24px rgba(2,6,23,0.08);}
.pg-gauge .num{ font-weight:900; font-size:28px; } .pg-gauge .lab{ font-size:11px; color:var(--muted); margin-top:-4px; }
</style>
""", unsafe_allow_html=True)

def chip(text, kind="ok"):
    st.markdown(f'<span class="pg-chip {kind}">{text}</span>', unsafe_allow_html=True)

def stat(label, value, help_text=""):
    st.markdown(f'''
    <div class="pg-card pg-stat"><div class="k">{label}</div><div class="v">{value}</div>
    <div style="color:#64748B;font-size:12px;">{help_text}</div></div>''', unsafe_allow_html=True)

def gauge(score:int, green:int, yellow:int):
    s = int(max(0,min(100,score))); green_end=min(s,green); yellow_span=min(max(s-green,0), (yellow-green))
    p= (green_end/100)*100; m= ((green_end+yellow_span)/100)*100
    st.markdown(f"""
    <div class="pg-card" style="display:flex;align-items:center;gap:14px;justify-content:center;">
      <div class="pg-gauge" style="--p:{green_end}%; --m:{green_end+yellow_span}%;">
        <div><div class="num">{s}</div><div class="lab">Risk</div></div>
      </div>
      <div>
        <div class="pg-chip ok">Green ≤ {green}</div>
        <div class="pg-chip warn">Yellow ≤ {yellow}</div>
        <div class="pg-chip danger">High &gt; {yellow}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ===== Sidebar tuning =====
with st.sidebar:
    st.header("Risk tuning")
    w_prob = st.slider("Weight: precip. probability", 0.0, 1.0, 0.50, 0.05)
    w_mm   = st.slider("Weight: precip. mm/hr",      0.0, 5.0, 2.50, 0.10)
    w_cloud= st.slider("Weight: cloud %",            0.0, 0.5, 0.20, 0.05)
    w_now  = st.slider("Weight: nowcast boost",      0.0,10.0, 8.00, 0.50)
    w_wind = st.slider("Weight: wind bonus",         0.0, 3.0, 1.00, 0.10)
    green_max  = st.slider("Green max", 10, 50, 30)
    yellow_max = st.slider("Yellow max", 40, 90, 60)
    dev = st.toggle("Developer metrics", value=False)

# ===== Data helpers =====
def ensure_ts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    try: df["ts"] = df["ts"].dt.tz_localize(None)
    except Exception: pass
    return df

@st.cache_data(show_spinner=False)
def geocode(place: str):
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={place}&count=1&language=en&format=json"
    r = requests.get(url, timeout=TIMEOUT); r.raise_for_status()
    j = r.json()
    if j.get("results"):
        g = j["results"][0]
        return {"name": g.get("name"), "country": g.get("country"),
                "lat": float(g.get("latitude")), "lon": float(g.get("longitude"))}
    raise ValueError("Location not found")

@st.cache_data(show_spinner=False)
def fetch_openmeteo_hourly(lat: float, lon: float):
    url = ("https://api.open-meteo.com/v1/forecast"
           f"?latitude={lat}&longitude={lon}"
           "&hourly=precipitation_probability,precipitation,cloud_cover,wind_speed_10m"
           "&forecast_days=7&timezone=auto")
    r = requests.get(url, timeout=TIMEOUT); r.raise_for_status()
    j = r.json()
    times = j["hourly"]["time"]
    df = pd.DataFrame({
        "ts": pd.to_datetime(times, errors="coerce"),
        "precip_probability": j["hourly"].get("precipitation_probability", [0]*len(times)),
        "precip_mm":          j["hourly"].get("precipitation",             [0]*len(times)),
        "cloud_cover":        j["hourly"].get("cloud_cover",               [0]*len(times)),
        "wind10m":            j["hourly"].get("wind_speed_10m",            [0]*len(times)),
    })
    tzname = j.get("timezone", "local")
    return ensure_ts(df), tzname

@st.cache_data(show_spinner=False)
def fetch_power_recent(lat: float, lon: float, days: int = 2):
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
    r = requests.get(base, params=params, timeout=TIMEOUT); r.raise_for_status()
    data = r.json()["properties"]["parameter"]
    hours = list(next(iter(data.values())).keys())
    rows = []
    for h in hours:
        rows.append({
            "ts": pd.to_datetime(h, format="%Y%m%d%H", errors="coerce"),
            "t2m": data["T2M"].get(h, np.nan),
            "ws10m": data["WS10M"].get(h, np.nan),
            "prec_mmph": data["PRECTOTCORR"].get(h, 0.0)
        })
    return ensure_ts(pd.DataFrame(rows))

def power_nowcast_mmph(dfp: pd.DataFrame) -> float:
    if dfp is None or dfp.empty: return 0.0
    tail6 = dfp.tail(6)["prec_mmph"].fillna(0.0)
    val = float(tail6.iloc[-1]);  val = val if val > 0 else float(tail6.max())
    return max(0.0, min(6.0, val))

@st.cache_data(show_spinner=False)
def recent_power_nowcast(lat: float, lon: float) -> float:
    try:
        return power_nowcast_mmph(fetch_power_recent(lat, lon, days=2))
    except Exception:
        return 0.0

def compute_confidence(df: pd.DataFrame):
    if df.empty: return 0.5, "medium"
    n = min(6, len(df)); sub = df.head(n)
    f_rain = (sub["precip_mm"] > 0).astype(int)
    p_rain = (sub["sat_rain_mmph"] > 0).astype(int)
    conf = (f_rain == p_rain).sum() / max(1, n)
    label = "high" if conf >= 0.8 else ("medium" if conf >= 0.55 else "low")
    return float(conf), label

# ===== Baseline risk =====
def clamp(x, lo, hi): return max(lo, min(hi, x))
def score_row(row: pd.Series) -> float:
    pp = clamp(row.get("precip_probability", 0) or 0, 0, 100)
    pm = clamp(row.get("precip_mm", 0) or 0, 0, 20)
    cc = clamp(row.get("cloud_cover", 0) or 0, 0, 100)
    w  = clamp(row.get("wind10m", 0) or 0, 0, 25)
    sr = clamp(row.get("sat_rain_mmph", 0) or 0, 0, 20)
    base = w_prob*pp + w_mm*pm + w_cloud*cc
    nowcast_boost = w_now*min(sr, 6)
    wind_bonus    = w_wind*max(w-10, 0)
    return float(min(100, base + nowcast_boost + wind_bonus))

def band(score: float) -> str:
    if score <= green_max: return "green"
    if score <= yellow_max: return "yellow"
    return "red"

def compute_safe_windows(df: pd.DataFrame, min_hours: int = 1) -> list:
    out, current = [], None
    for _, r in df.iterrows():
        ok = (r["risk"] <= 35) and (r["precip_mm"] < 1.0)
        if ok:
            if current is None: current = {"start": r["ts"], "end": r["ts"], "max_score": float(r["risk"]), "len": 1}
            else: current["end"] = r["ts"]; current["max_score"] = max(current["max_score"], float(r["risk"])); current["len"] += 1
        else:
            if current is not None and current["len"] >= min_hours: out.append(current)
            current = None
    if current is not None and current["len"] >= min_hours: out.append(current)
    return out

# ===== Hero =====
st.markdown("""
<div class="pg-hero">
  <div class="title">🌦️ ParadeGuard — NASA-powered AI</div>
  <div class="sub">Nowcast: NASA POWER · Map: NASA GIBS IMERG · Ensemble ML (RF + GBDT → Logistic + Calibration)</div>
</div>
""", unsafe_allow_html=True)

# ===== Controls =====
q = st.query_params
place_default = q.get("place", "Assiut, Egypt")

c_top = st.columns([2,2,1,1])
with c_top[0]:
    place = st.text_input("City or place", value=place_default)
with c_top[1]:
    event_time = st.time_input("Event time (today)", value=dt.datetime.now().time())
    def _idx(val, fallback=1):
        try:
            i = int(val) - 1
            return i if i in [0,1,2,3] else fallback
        except Exception:
            return fallback
    event_len = st.selectbox("Event length (hours)", [1,2,3,4], index=_idx(q.get("len", 2)))
with c_top[2]:
    show_imerge = st.toggle("Show NASA IMERG map", value=(q.get("imerge","True")=="True"))
with c_top[3]:
    use_power_boost = st.toggle("Use NASA POWER boost", value=(q.get("power","True")=="True"))

c_opts = st.columns([1,1,2])
with c_opts[0]:
    min_safe = st.selectbox("Min safe window (hours)", [1,2,3,4], index=1)
with c_opts[1]:
    safe_mode = st.toggle("Safe-mode (avoid external maps)", value=False)
with c_opts[2]:
    run = st.button("Check risk", use_container_width=True)

# ===== Load Ensemble model =====
MODEL = None
if os.path.exists("models/risk_model.pkl"):
    try:
        with open("models/risk_model.pkl","rb") as f:
            MODEL = pickle.load(f)
    except Exception:
        MODEL = None

def predict_ensemble(df: pd.DataFrame, model_payload):
    """Return predicted class and proba using RF+GB → Logistic (calibrated)."""
    feats = model_payload["features"]
    rf = model_payload["rf"]; gb = model_payload["gb"]; meta_cal = model_payload["meta_cal"]; classes = model_payload["classes"]

    dm = df.copy()
    dm["hour"] = dm["ts"].dt.hour
    dm["dow"]  = dm["ts"].dt.dayofweek
    dm["lag1_mm"] = dm["precip_mm"].shift(1).fillna(0)
    dm["lag2_mm"] = dm["precip_mm"].shift(2).fillna(0)
    dm["roll3_mm"] = dm["precip_mm"].rolling(3, min_periods=1).mean()
    dm["roll6_mm"] = dm["precip_mm"].rolling(6, min_periods=1).mean()
    dm["nowcast_mmph"] = dm["sat_rain_mmph"].astype(float)

    for f in feats:
        if f not in dm.columns: dm[f] = 0.0

    X = dm[feats].values

    def order_cols(p, model_classes):
        idx = [list(model_classes).index(c) for c in classes]
        return p[:, idx]
    p_rf = order_cols(rf.predict_proba(X), rf.classes_)
    p_gb = order_cols(gb.predict_proba(X), gb.classes_)
    meta_X = np.hstack([p_rf, p_gb])

    pred = meta_cal.predict(meta_X)
    if hasattr(meta_cal, "predict_proba"):
        proba = meta_cal.predict_proba(meta_X)
    else:
        # fallback equal probs
        proba = np.ones((len(pred), len(classes))) / len(classes)
    return pred, proba, classes

def pick_best_windows(df, duration_hours:int, top_k:int=3):
    """Search best (lowest max-risk) windows in next 72h."""
    horizon = df[df["ts"] <= (df["ts"].iloc[0] + pd.Timedelta(hours=72))]
    wins = []
    for i in range(0, max(0, len(horizon)-duration_hours+1)):
        block = horizon.iloc[i:i+duration_hours]
        if len(block) < duration_hours: break
        score = float(block["risk"].max())
        wins.append((block["ts"].iloc[0], block["ts"].iloc[-1], score))
    wins.sort(key=lambda x: x[2])
    return wins[:top_k]

if run:
    try:
        # 1) Geocode
        loc = geocode(place); lat, lon = loc["lat"], loc["lon"]
        # 2) Forecast
        df, tzname = fetch_openmeteo_hourly(lat, lon)
        for c in ["precip_probability","precip_mm","cloud_cover","wind10m"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        # 3) POWER nowcast boost
        df["sat_rain_mmph"] = 0.0
        if use_power_boost and not safe_mode and len(df) > 0:
            df.loc[df.index[:6], "sat_rain_mmph"] = float(recent_power_nowcast(lat, lon))
        # 4) Baseline risk
        df["risk"] = df.apply(score_row, axis=1)
        df["band"] = df["risk"].apply(band)

        st.success(f"{loc['name']}, {loc['country']} — times in {tzname}")
        st.session_state["green_max"] = green_max; st.session_state["yellow_max"] = yellow_max

        # Confidence
        conf, conf_lbl = compute_confidence(df)

        # Event window
        event_dt = dt.datetime.combine(dt.date.today(), event_time)
        event_ts = pd.Timestamp(event_dt)
        mask = (df["ts"] >= event_ts) & (df["ts"] < event_ts + pd.Timedelta(hours=event_len))
        block = df.loc[mask]
        if block.empty:
            idx = (df["ts"] - event_ts).abs().idxmin()
            selected_row = df.loc[idx]; block_risk_text = ""
            st.warning("Event window is outside the forecast horizon; showing the closest hour instead.")
        else:
            selected_row = block.iloc[0]
            block_risk_text = f" — Event window max risk ({event_len}h): **{int(block['risk'].max())}**"

        # ===== Header Cards =====
        c_h = st.columns([3,3,2,2,2])
        with c_h[0]: stat("Location", f"{loc['name']}, {loc['country']}", "Timezone: "+str(tzname))
        with c_h[1]: stat("Event", selected_row["ts"].strftime("%a %d %b %H:%M"), f"Length: {event_len}h")
        with c_h[2]: gauge(int(round(selected_row["risk"])), green_max, yellow_max)
        with c_h[3]: stat("Confidence", f"{int(conf*100)}%", f"Agreement {conf_lbl.upper()}")
        with c_h[4]:
            if selected_row["risk"] <= green_max: chip("Go / Low risk","ok")
            elif selected_row["risk"] <= yellow_max: chip("Be cautious","warn")
            else: chip("Consider reschedule","danger")

        # ===== Tabs =====
        tab_overview, tab_map, tab_details, tab_ai = st.tabs(["Overview", "Map", "Details", "AI Model"])

        # --- Overview ---
        with tab_overview:
            df_chart = df.set_index("ts"); df_chart.index = pd.DatetimeIndex(df_chart.index)
            c1, c2 = st.columns([3,2])
            with c1:
                st.markdown('<div class="pg-section">Risk timeline</div>', unsafe_allow_html=True)
                st.line_chart(df_chart[["risk"]], height=280)
            with c2:
                st.markdown('<div class="pg-section">Precipitation (mm/hr)</div>', unsafe_allow_html=True)
                st.area_chart(df_chart[["precip_mm"]], height=280)

            st.markdown('<div class="pg-section">Next 6 hours</div>', unsafe_allow_html=True)
            cols = st.columns(6)
            for i, (_, r) in enumerate(df.head(6).iterrows()):
                clr = "ok" if r["risk"]<=green_max else ("warn" if r["risk"]<=yellow_max else "danger")
                with cols[i]:
                    st.markdown(f"""
                    <div class="pg-card pg-hour">
                      <div class="t">{r["ts"].strftime("%a %H:%M")}</div>
                      <div class="s">{int(round(r['risk']))}</div>
                      <div class="b"><span class="pg-chip {clr}">{r["band"].upper()}</span></div>
                    </div>""", unsafe_allow_html=True)

            # Auto-optimizer: أفضل 3 نوافذ في 72 ساعة
            st.markdown('<div class="pg-section">Optimized windows (next 72h)</div>', unsafe_allow_html=True)
            recs = pick_best_windows(df, event_len, top_k=3)
            if not recs:
                st.info("No candidate windows found under current horizon.")
            else:
                for s,e,sc in recs:
                    msg = f"✅ {s.strftime('%a %d %b %H:%M')} → {e.strftime('%a %d %b %H:%M')}  (max risk {int(round(sc))})"
                    st.write(msg)

        # --- Map ---
        with tab_map:
            st.markdown('<div class="pg-section">NASA IMERG (GIBS)</div>', unsafe_allow_html=True)
            if show_imerge and not safe_mode and HAS_FOLIUM:
                try:
                    op = st.slider("IMERG layer opacity", 0.2, 1.0, 0.75, 0.05, key="op_map")
                    m = folium.Map(location=[lat, lon], zoom_start=6, control_scale=True)
                    layer = "IMERG_Precipitation_Rate_30min"
                    time_str = dt.datetime.utcnow().strftime("%Y-%m-%d")
                    tilematrix = "GoogleMapsCompatible_Level9"
                    tiles = (f"https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/{layer}/default/{time_str}/{tilematrix}/{{z}}/{{y}}/{{x}}.png")
                    folium.raster_layers.TileLayer(
                        tiles=tiles, attr="NASA GIBS", name="IMERG 30-min",
                        overlay=True, control=True, opacity=float(op)
                    ).add_to(m)
                    folium.Marker([lat, lon], tooltip=f"{loc['name']}, {loc['country']}").add_to(m)
                    st_folium(m, width=1000, height=520)
                except Exception as e:
                    st.warning(f"Could not load IMERG map (GIBS): {e}")
            else:
                st.map(pd.DataFrame({"lat":[lat], "lon":[lon]}))
                st.caption("Enable IMERG map for the satellite rain layer.")

        # --- Details ---
        with tab_details:
            st.markdown('<div class="pg-section">Explainability (baseline)</div>', unsafe_allow_html=True)
            sr = float(selected_row["sat_rain_mmph"])
            contrib = pd.DataFrame({
                "factor": ["prob %","mm/hr","cloud %","nowcast","wind"],
                "weight": [w_prob, w_mm, w_cloud, w_now, w_wind],
                "value":  [
                    float(selected_row["precip_probability"]),
                    float(selected_row["precip_mm"]),
                    float(selected_row["cloud_cover"]),
                    min(sr,6),
                    max(float(selected_row["wind10m"])-10,0)
                ]
            })
            contrib["contribution"] = contrib["weight"]*contrib["value"]
            st.bar_chart(contrib.set_index("factor")[["contribution"]])

            st.markdown('<div class="pg-section">Table</div>', unsafe_allow_html=True)
            show = df[["ts","band","risk","precip_probability","precip_mm","cloud_cover","wind10m","sat_rain_mmph"]].copy()
            show.rename(columns={"ts":"time","precip_probability":"precip_prob_%","cloud_cover":"cloud_%","wind10m":"wind_10m","sat_rain_mmph":"nowcast_mmph"}, inplace=True)
            st.dataframe(show, use_container_width=True, hide_index=True)
            st.download_button("Download CSV", show.to_csv(index=False).encode("utf-8"),
                               file_name="paradeguard_export.csv", mime="text/csv")

        # --- AI Model ---
        with tab_ai:
            st.markdown('<div class="pg-section">Ensemble classifier (green / yellow / red)</div>', unsafe_allow_html=True)
            if MODEL is None:
                st.info("No ML model found. Run:  `python train_pro.py`  to create models/risk_model.pkl")
            else:
                feats = MODEL["features"]; classes = MODEL["classes"]
                pred, proba, cls = predict_ensemble(df, MODEL)
                df["ml_pred"] = pred
                # event hour
                sel_idx = df.index.get_loc(selected_row.name)
                ml_label = str(df.iloc[sel_idx]["ml_pred"]).upper()
                st.metric("ML predicted band @ event", ml_label)

                # Probabilities @ event (nice little table)
                if proba is not None:
                    ci = list(cls)
                    prow = proba[sel_idx]
                    st.write(pd.DataFrame({"class":ci, "probability":[f"{p*100:.1f}%" for p in prow]}))

                # Compare first 24h
                comp = pd.DataFrame({"time": df["ts"].head(24), "baseline": df["band"].head(24), "ml": df["ml_pred"].head(24)})
                st.markdown('<div class="pg-section">Baseline vs ML (next 24h)</div>', unsafe_allow_html=True)
                st.dataframe(comp, use_container_width=True, hide_index=True)

                # Show permutation importances from training (as guidance)
                if "rf_perm_importance" in MODEL and MODEL["rf_perm_importance"] is not None:
                    st.markdown('<div class="pg-section">Feature importance (training, RF permutation)</div>', unsafe_allow_html=True)
                    fi = MODEL["rf_perm_importance"].set_index("feature")
                    st.bar_chart(fi)

        # Share URL & fallback small map
        st.query_params.update({"place": place,"time": event_time.strftime("%H:%M"),"len": str(event_len),"imerge": str(show_imerge),"power": str(use_power_boost)})
        if not show_imerge or safe_mode: st.map(pd.DataFrame({"lat":[lat], "lon":[lon]}))

    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
