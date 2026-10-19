"""
Planificateur de nuit — zone de ciel visible (Vespera Pro 2)

Lancer :  python app.py   (ou : streamlit run app.py)
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body, get_sun
from astropy.time import Time
from astropy.utils import iers
from streamlit import config
from streamlit.runtime import exists as _streamlit_running

# Permet de lancer simplement `python app.py` (ou le bouton Run de PyCharm) :
# si le script n'est pas exécuté par Streamlit, il se relance tout seul avec.
if __name__ == "__main__" and not _streamlit_running():
    try:
        sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", __file__]))
    except KeyboardInterrupt:
        sys.exit(0)

# Tables IERS fournies avec le paquet astropy-iers-data (pas de téléchargement à l'exécution)
iers.conf.auto_download = False
iers.conf.iers_degraded_accuracy = "ignore"

DATA_DIR = Path(__file__).parent / "data"
OPENNGC_URL = "https://raw.githubusercontent.com/mattiaverga/OpenNGC/master/database_files/"

# Caractéristiques officielles du Vespera Pro 2 (vaonis.com/products/vespera-pro2)
VESPERA = {
    "Ouverture": "50 mm",
    "Focale": "245 mm (f/4,9)",
    "Capteur": "Sony IMX676 couleur, 3536 × 3536 px (12,5 Mpx), 7,0 × 7,0 mm",
    "Taille des pixels": "2 µm",
    "Champ natif": "1,6° × 1,6°",
    "Champ max. en mosaïque (CovalENS)": "4,18° × 2,45°",
    "Résolution de détail": "1,6″/pixel",
    "Magnitude limite": "16",
    "Monture": "azimutale",
}
FOV_NATIVE = (1.6, 1.6)  # degrés
FOV_MOSAIC = (4.18, 2.45)
PIXEL_SCALE = 1.6  # arcsec/pixel

SECTORS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSO", "SO", "OSO", "O", "ONO", "NO", "NNO"]
SECTOR_WIDTH = 360 / len(SECTORS)  # 22.5°
DEFAULT_DIRS = ["E", "ESE", "SE", "SSE", "S"]

CATEGORIES = {
    "Nébuleuses": ["HII", "EmN", "Neb", "RfN", "SNR", "PN", "Cl+N"],
    "Galaxies": ["G", "GPair", "GTrpl", "GGroup"],
    "Amas": ["OCl", "GCl", "*Ass"],
}
TYPE_FR = {
    "HII": "Nébuleuse en émission", "EmN": "Nébuleuse en émission", "Neb": "Nébuleuse",
    "RfN": "Nébuleuse par réflexion", "SNR": "Rémanent de supernova",
    "PN": "Nébuleuse planétaire", "Cl+N": "Amas + nébuleuse",
    "G": "Galaxie", "GPair": "Paire de galaxies", "GTrpl": "Trio de galaxies",
    "GGroup": "Groupe de galaxies", "OCl": "Amas ouvert", "GCl": "Amas globulaire",
    "*Ass": "Association d'étoiles",
}

FILTERS = ["Sans filtre", "CLS", "Dual band (Ha/OIII)"]
EMISSION = {"HII", "EmN", "SNR", "Cl+N", "Neb", "PN"}
WEEKDAYS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]


def filter_fit(code: str, flt: str) -> str:
    """Règle empirique : adéquation d'un type d'objet avec le filtre optique."""
    emission = code in EMISSION
    if flt.startswith("Dual"):
        return "★ idéal" if emission else "✖ déconseillé"
    if flt == "CLS":
        return "★ idéal" if emission else "✔ ok"
    return "✔ ok" if emission else "★ idéal"  # sans filtre


def framing(major_arcmin) -> str:
    """Cadrage avec le champ du Vespera Pro 2 (marge 10 % : rotation de champ de la monture azimutale)."""
    if pd.isna(major_arcmin):
        return "ok"
    if major_arcmin < 2:
        return "très petit"
    if major_arcmin <= 0.9 * min(FOV_NATIVE) * 60:
        return "ok"
    if major_arcmin <= 0.9 * min(FOV_MOSAIC) * 60:
        return "mosaïque"
    return "trop grand"


st.set_page_config(page_title="Ciel visible — planificateur", page_icon="🔭", layout="wide",
                   initial_sidebar_state="auto")


# --------------------------------------------------------------------------- #
# Catalogue (OpenNGC, licence CC-BY-SA 4.0 — Mattia Verga)
# --------------------------------------------------------------------------- #
def _download(name: str) -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / name
    if not path.exists():
        r = requests.get(OPENNGC_URL + name, timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)
    return path


def _hms_to_deg(s: pd.Series) -> np.ndarray:
    parts = s.str.split(":", expand=True).astype(float)
    return ((parts[0] + parts[1] / 60 + parts[2] / 3600) * 15).to_numpy()


def _dms_to_deg(s: pd.Series) -> np.ndarray:
    sign = np.where(s.str.startswith("-"), -1.0, 1.0)
    parts = s.str.lstrip("+-").str.split(":", expand=True).astype(float)
    return (sign * (parts[0] + parts[1] / 60 + parts[2] / 3600)).to_numpy()


@st.cache_data(show_spinner="Chargement du catalogue…")
def load_catalog() -> pd.DataFrame:
    frames = []
    for name in ("NGC.csv", "addendum.csv"):
        df = pd.read_csv(_download(name), sep=";", dtype=str)
        df["from_addendum"] = name == "addendum.csv"
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["Type"].isin(sum(CATEGORIES.values(), []))].dropna(subset=["RA", "Dec"]).copy()

    for col in ("MajAx", "MinAx", "V-Mag", "B-Mag", "M"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ra"] = _hms_to_deg(df["RA"])
    df["dec"] = _dms_to_deg(df["Dec"])
    df["mag"] = df["V-Mag"].fillna(df["B-Mag"])
    df["size"] = df["MajAx"]
    df["minor"] = df["MinAx"]

    def label(r):
        if pd.notna(r["M"]):
            return f"M{int(r['M'])}"
        n = r["Name"]
        for pre in ("NGC", "IC"):
            if n.startswith(pre):
                return f"{pre} {n[len(pre):].lstrip('0')}"
        return n

    df["label"] = df.apply(label, axis=1)
    df["common"] = df["Common names"].fillna("").str.split(",").str[0]
    df["famous"] = df["M"].notna() | df["common"].ne("") | df["from_addendum"]
    cat_of = {t: c for c, ts in CATEGORIES.items() for t in ts}
    df["category"] = df["Type"].map(cat_of)
    df["type_fr"] = df["Type"].map(TYPE_FR)
    keep = ["label", "common", "Name", "Type", "type_fr", "category", "Const",
            "ra", "dec", "mag", "size", "minor", "famous"]
    return df[keep].drop_duplicates("label").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Lieu
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=3600, show_spinner=False)
def geocode(city: str) -> list[dict]:
    r = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 5, "language": "fr"},
        timeout=15,
    )
    r.raise_for_status()
    return r.json().get("results", [])


# --------------------------------------------------------------------------- #
# Astronomie (astropy : précession, nutation, aberration, réfraction)
# --------------------------------------------------------------------------- #
STRIDE = 5  # le calcul est à la minute ; l'enchaînement du plan se décide toutes les 5 min


def _tmpl() -> str:
    """Thème des graphiques = thème choisi par le visiteur dans le menu Réglages."""
    return "plotly_white" if st.session_state.get("theme_mode") == "Clair" else "plotly_dark"


def _fc() -> str:
    return "#0e1633" if st.session_state.get("theme_mode") == "Clair" else "#e6ecff"


# Graphiques tactiles : pas de zoom/déplacement qui bloque le défilement de la page, pas de barre d'outils
PLOT_CFG = {"displayModeBar": False, "scrollZoom": False, "doubleClick": False}


def _n(dt):
    """Datetime « mur » local sans fuseau : plotly n'interprète pas les décalages horaires."""
    return dt.replace(tzinfo=None)


def az_to_cardinal(az: float) -> str:
    return SECTORS[int(((az + SECTOR_WIDTH / 2) % 360) // SECTOR_WIDTH)]


def altaz_frame(times, loc, elev, temp_c):
    """Repère horizontal *apparent* : réfraction atmosphérique incluse (pression d'après l'altitude)."""
    pressure = 1013.25 * (1 - 2.25577e-5 * elev) ** 5.25588  # atmosphère standard, hPa
    return AltAz(obstime=times, location=loc, pressure=pressure * u.hPa,
                 temperature=temp_c * u.deg_C, relative_humidity=0.5, obswl=0.55 * u.micron)


def find_dark_window(lat, lon, elev, tz, night: date, sun_limit: float):
    """Instants UTC (pas d'1 min) de la nuit noire qui suit le soir du `night` (midi→midi, heure locale)."""
    loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=elev * u.m)
    start_local = datetime(night.year, night.month, night.day, 12, 0, tzinfo=tz)
    start_utc = start_local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
    grid = Time(start_utc, scale="utc") + np.arange(0, 24 * 60) * u.min
    # Soleil : altitude géométrique du centre du disque (définition des crépuscules)
    sun_alt = get_sun(grid).transform_to(AltAz(obstime=grid, location=loc)).alt.deg

    for limit in (-18.0, -12.0, -6.0, -0.8):
        if limit < sun_limit:
            continue
        dark = sun_alt < limit
        if dark.any():
            d = np.diff(np.concatenate([[0], dark.astype(int), [0]]))
            s, e = np.where(d == 1)[0], np.where(d == -1)[0]
            i = np.argmax(e - s)
            return grid[s[i]:e[i]], limit, loc
    raise RuntimeError("Pas de nuit exploitable à cette date/ce lieu.")


def intervals_from_sectors(table: pd.DataFrame) -> list[tuple]:
    """Secteurs de 22,5° → intervalles d'azimut (début, fin, alt min, alt max), contigus fusionnés."""
    items = []
    for i, r in table.reset_index(drop=True).iterrows():
        if r["Visible"]:
            c = i * SECTOR_WIDTH
            items.append([(c - SECTOR_WIDTH / 2) % 360, (c + SECTOR_WIDTH / 2) % 360,
                          float(r["Alt min"]), float(r["Alt max"])])
    merged = []
    for it in items:
        if merged and abs(merged[-1][1] - it[0]) < 1e-9 and merged[-1][2:] == it[2:]:
            merged[-1][1] = it[1]
        else:
            merged.append(it)
    if len(merged) > 1 and abs(merged[-1][1] - merged[0][0]) < 1e-9 and merged[-1][2:] == merged[0][2:]:
        merged[0][0] = merged[-1][0]
        merged.pop()
    return [tuple(m) for m in merged]


def az_span(a0, a1) -> float:
    span = (a1 - a0) % 360
    return 360.0 if (span == 0 and a1 != a0) or (a1 - a0) >= 360 else span


def in_mask(az, alt, intervals):
    """Vrai si (azimut, altitude) est dans l'un des intervalles dégagés."""
    out = np.zeros(np.shape(az), bool)
    for a0, a1, lo, hi in intervals:
        inside = ((az - a0) % 360) <= az_span(a0, a1)
        out |= inside & (alt >= lo) & (alt <= hi)
    return out & (alt > 0)


def runs(v: np.ndarray):
    d = np.diff(np.concatenate([[0], np.asarray(v).astype(int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


@st.cache_data(show_spinner="Calcul de la nuit…")
def compute_night(lat, lon, elev, tz_name, night, sun_limit, intervals, mag_limit,
                  min_size, cats, famous_unrated, temp_c):
    tz = ZoneInfo(tz_name)
    times, used_limit, loc = find_dark_window(lat, lon, elev, tz, night, sun_limit)
    T = len(times)
    frame = lambda t: altaz_frame(t, loc, elev, temp_c)  # noqa: E731

    cat = load_catalog()
    sel = cat[cat["category"].isin(cats)].copy()
    has_mag = sel["mag"].notna() & (sel["mag"] <= mag_limit)
    unrated = sel["mag"].isna() & sel["famous"] & famous_unrated
    sel = sel[(has_mag | unrated) & (sel["size"].fillna(min_size) >= min_size)].reset_index(drop=True)

    # Lune (à la minute)
    moon = get_body("moon", times, loc).transform_to(frame(times))
    elong = get_body("sun", times, loc).separation(get_body("moon", times, loc)).rad
    illum = float(np.mean((1 - np.cos(elong)) / 2))
    m_alt, m_az = moon.alt.deg, moon.az.deg
    t_loc = np.array([t.replace(tzinfo=ZoneInfo("UTC")).astimezone(tz) for t in times.to_datetime()])

    def coords(df):
        return SkyCoord(df["ra"].to_numpy()[:, None] * u.deg, df["dec"].to_numpy()[:, None] * u.deg)

    # Passe 1 (toutes les 5 min) : quels objets traversent la zone dégagée ?
    ci = np.arange(0, T, 5)
    aa = coords(sel).transform_to(frame(times[ci][None, :]))
    keep = in_mask(aa.az.deg, aa.alt.deg, intervals).any(axis=1)
    sel = sel[keep].reset_index(drop=True)

    base = {"sel": sel, "t_loc": t_loc, "moon_alt": m_alt, "moon_az": m_az,
            "illum": illum, "used_limit": used_limit}
    if sel.empty:
        return {**base, "res": pd.DataFrame(), "alt": np.zeros((0, T)), "az": np.zeros((0, T)),
                "ok": np.zeros((0, T), bool), "sep": np.zeros((0, T)), "intervals": {}}

    # Passe 2 (à la minute) : position apparente précise des objets retenus
    aa = coords(sel).transform_to(frame(times[None, :]))
    alt, az = aa.alt.deg, aa.az.deg
    ok = in_mask(az, alt, intervals)

    a1, a2 = np.radians(alt), np.radians(m_alt)[None, :]
    cosd = np.sin(a1) * np.sin(a2) + np.cos(a1) * np.cos(a2) * np.cos(np.radians(az - m_az[None, :]))
    sep = np.degrees(np.arccos(np.clip(cosd, -1, 1)))

    rows, win = [], {}
    for i, r in sel.iterrows():
        v = ok[i]
        if not v.any():
            continue
        rr = runs(v)
        s, e = max(rr, key=lambda x: x[1] - x[0])
        best = np.where(v)[0][np.argmax(alt[i][v])]
        rows.append({
            "idx": i, "Objet": r["label"], "Nom": r["common"], "Type": r["type_fr"], "code": r["Type"],
            "Const.": r["Const"], "Mag": r["mag"], "Taille (')": r["size"],
            "Visible": f"{t_loc[s]:%H:%M} → {t_loc[e]:%H:%M}", "_s": int(s), "_e": int(e),
            "Durée (h)": round((e - s + 1) / 60, 2),
            "Meilleur": f"{t_loc[best]:%H:%M}",
            "Alt": round(float(alt[i, best]), 1), "Az": round(float(az[i, best]), 1),
            "Dir": az_to_cardinal(az[i, best]),
            "Lune (°)": round(float(sep[i, best]), 1) if m_alt[best] > 0 else None,
            "⚠ Lune": "oui" if (m_alt[best] > 0 and sep[i, best] < 40) else "",
        })
        win[i] = [(t_loc[a], t_loc[b]) for a, b in rr]

    return {**base, "res": pd.DataFrame(rows), "alt": alt, "az": az, "ok": ok, "sep": sep,
            "intervals": win}


# --------------------------------------------------------------------------- #
# Graphiques
# --------------------------------------------------------------------------- #
def _layout(fig, **kw):
    fig.update_layout(template=_tmpl(), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      dragmode=False,
                      font=dict(family="Inter, system-ui, -apple-system, sans-serif", size=12, color=_fc()),
                      **kw)
    fig.update_xaxes(fixedrange=True, gridcolor="rgba(150,160,185,0.14)", zeroline=False)
    fig.update_yaxes(fixedrange=True, gridcolor="rgba(150,160,185,0.10)", zeroline=False)
    return fig


def _moon_bands(fig, data):
    for a, b in runs(data["moon_alt"] > 0):
        fig.add_vrect(x0=_n(data["t_loc"][a]), x1=_n(data["t_loc"][b]),
                      fillcolor="rgba(255,196,110,0.13)", line_width=0, layer="below")


def _now_line(fig, data):
    now = datetime.now(data["t_loc"][0].tzinfo)
    if data["t_loc"][0] <= now <= data["t_loc"][-1]:
        fig.add_shape(type="line", x0=_n(now), x1=_n(now), y0=0, y1=1, yref="paper",
                      line=dict(color="#ffb86b", dash="dot", width=2))


def polar_chart(intervals, data, k, top_n=25):
    fig = go.Figure()
    for a0, a1, lo, hi in intervals:
        span = az_span(a0, a1)
        fig.add_trace(go.Barpolar(
            r=[hi - lo], base=[90 - hi], theta=[(a0 + span / 2) % 360], width=[span],
            marker=dict(color="rgba(122,162,255,0.22)", line=dict(color="rgba(122,162,255,0.75)", width=1)),
            hoverinfo="skip", showlegend=False))

    res, sel = data["res"], data["sel"]
    if len(res):
        ok_now = data["ok"][:, k]
        idxs = [i for i in res["idx"] if ok_now[i]]
        sub = sel.loc[idxs].copy()
        sub["alt"], sub["az"] = data["alt"][idxs, k], data["az"][idxs, k]
        sub = sub.assign(rank=sub["mag"].fillna(99)).sort_values("rank")
        colors = {"Nébuleuses": "#ff7a90", "Galaxies": "#7aa2ff", "Amas": "#ffd166"}
        for cat, g in sub.groupby("category"):
            top = g.head(top_n)
            fig.add_trace(go.Scatterpolar(
                r=90 - top["alt"], theta=top["az"], mode="markers+text", name=cat,
                text=top["label"], textposition="top center", textfont=dict(size=10),
                marker=dict(size=8, color=colors.get(cat, "#aaa")),
                hovertext=top["label"] + " " + top["common"] + "<br>alt " + top["alt"].round(1).astype(str)
                + "° · az " + top["az"].round(1).astype(str) + "°", hoverinfo="text"))

    m_alt, m_az = data["moon_alt"][k], data["moon_az"][k]
    if m_alt > 0:
        fig.add_trace(go.Scatterpolar(r=[90 - m_alt], theta=[m_az], mode="markers+text", text=["🌙"],
                                      textfont=dict(size=18), name="Lune",
                                      marker=dict(size=1, opacity=0)))
    return _layout(
        fig, height=440, margin=dict(l=24, r=24, t=10, b=10), legend=dict(orientation="h", y=-0.05),
        polar=dict(bgcolor="rgba(0,0,0,0)",
                   angularaxis=dict(direction="clockwise", rotation=90, tickmode="array",
                                    tickvals=[0, 90, 180, 270], ticktext=["N", "E", "S", "O"],
                                    gridcolor="rgba(140,155,205,.32)", linecolor="rgba(140,155,205,.5)",
                                    tickfont=dict(color=_fc(), size=13)),
                   radialaxis=dict(range=[0, 90], tickvals=[0, 30, 60, 90],
                                   ticktext=["90°", "60°", "30°", "0°"], angle=45,
                                   gridcolor="rgba(140,155,205,.32)", linecolor="rgba(140,155,205,.5)",
                                   tickfont=dict(color=_fc(), size=10))))


def timeline_chart(data, ids):
    """Tous les objets `ids` (déjà triés chronologiquement), sur toute la durée de la nuit."""
    sel, t_loc = data["sel"], data["t_loc"]
    labels = [sel.loc[i, "label"] for i in ids]
    fig = go.Figure()
    _moon_bands(fig, data)
    for i, label in zip(ids, labels):
        for a, b in data["intervals"][i]:
            fig.add_trace(go.Bar(
                base=[_n(a)], x=[(b - a + timedelta(minutes=1)).total_seconds() * 1000],
                y=[label], orientation="h", marker_color="#7aa2ff",
                hovertemplate=f"{label}<br>{a:%H:%M} → {b:%H:%M}<extra></extra>", showlegend=False))
    _now_line(fig, data)
    return _layout(fig, barmode="overlay", height=max(260, 22 * len(ids) + 80),
                   margin=dict(l=10, r=10, t=10, b=30),
                   xaxis=dict(type="date", tickformat="%H:%M",
                              range=[_n(t_loc[0]), _n(t_loc[-1] + timedelta(minutes=1))]),
                   yaxis=dict(categoryorder="array", categoryarray=labels, autorange="reversed"))


# --------------------------------------------------------------------------- #
# Plan de nuit séquencé
# --------------------------------------------------------------------------- #
GAP = 2  # pas de 5 min entre deux objets (pointage, centrage) = 10 min
PLAN_STYLES = {  # (durée min, durée max) par objet, en pas de 5 min
    "Poses longues (2–4 h / objet)": (24, 48),
    "Équilibré (1–2 h / objet)": (12, 24),
    "Découverte (40–80 min / objet)": (8, 16),
}


def fmt_dur(minutes: int) -> str:
    return f"{minutes // 60} h {minutes % 60:02d}" if minutes >= 60 else f"{minutes} min"


def coarse_arrays(data, res, flt):
    """Grille de 5 min : visibilité dans la zone et score instantané de chaque objet de `res`."""
    idxs = res["idx"].to_numpy()
    alt = data["alt"][idxs][:, ::STRIDE]
    sep = data["sep"][idxs][:, ::STRIDE]
    ok = data["ok"][idxs][:, ::STRIDE].copy()
    mag = res["Mag"].to_numpy(float)
    base = np.where(np.isnan(mag), 0.9, np.clip(1 + (11 - mag) * 0.07, 0.6, 1.5))
    base = base * np.where(res["Filtre"].eq("★ idéal"), 1.0, 0.75)
    base = base * res["Cadrage"].map({"ok": 1.0, "mosaïque": 0.6, "très petit": 0.6}).fillna(0.3).to_numpy()
    pen = 0.3 if flt.startswith("Dual") else (0.7 if flt == "CLS" else 1.0)  # sensibilité à la Lune
    moon_up = (data["moon_alt"][::STRIDE] > 0)[None, :]
    moon_f = 1 - pen * data["illum"] * np.clip(1 - sep / 70, 0, 1) * moon_up
    quality = base[:, None] * (0.5 + 0.5 * np.sin(np.radians(np.clip(alt, 0, 90)))) * moon_f
    return ok, quality, ok.shape[0], ok.shape[1]


def build_plan(data, res, flt, min_steps, max_steps, gap=GAP, blocked=None, exclude=()):
    """Enchaînement optimal de créneaux (un objet à la fois, jamais deux fois le même).

    Programmation dynamique sur une grille de 5 min : on maximise la somme d'un score
    instantané (luminosité, hauteur, filtre, cadrage, Lune) sur des créneaux pendant lesquels
    l'objet reste *entièrement* dans la zone dégagée. Retourne [(ligne_de_res, début, longueur)]
    en pas de 5 min ; `refine_segments` précise ensuite les bornes à la minute.
    """
    if res.empty:
        return []
    ok, quality, n, T = coarse_arrays(data, res, flt)
    if blocked is not None:
        ok[:, blocked] = False
    for o in exclude:
        ok[o, :] = False

    segs = []
    for _ in range(40):
        score = np.where(ok, quality, 0.0)
        cum = np.concatenate([np.zeros((n, 1)), np.cumsum(score, axis=1)], axis=1)
        run = np.zeros((n, T + 1), int)
        for t in range(T - 1, -1, -1):
            run[:, t] = np.where(ok[:, t], run[:, t + 1] + 1, 0)

        cand = {}
        for s0 in range(T):
            for L in range(min_steps, max_steps + 1):
                if s0 + L > T:
                    break
                valid = run[:, s0] >= L
                if not valid.any():
                    break
                vals = np.where(valid, cum[:, s0 + L] - cum[:, s0], -1.0)
                o = int(np.argmax(vals))
                cand[(s0, L)] = (float(vals[o]), o)

        dp, choice = np.zeros(T + 1), [None] * (T + 1)
        for i in range(1, T + 1):
            dp[i] = dp[i - 1]
            for L in range(min_steps, max_steps + 1):
                s0 = i - L
                if s0 < 0:
                    break
                c = cand.get((s0, L))
                if c is None:
                    continue
                v = dp[max(s0 - gap, 0)] + c[0]
                if v > dp[i] + 1e-9:
                    dp[i], choice[i] = v, (c[1], s0, L)
        segs, i = [], T
        while i > 0:
            if choice[i] is None:
                i -= 1
            else:
                segs.append(choice[i])
                i = max(choice[i][1] - gap, 0)
        segs.sort(key=lambda x: x[1])

        by_obj = {}
        for o, s0, L in segs:
            by_obj.setdefault(o, []).append((s0, L))
        dups = {o: v for o, v in by_obj.items() if len(v) > 1}
        if not dups:
            return segs
        for o, v in dups.items():  # on interdit le créneau le moins bon de l'objet en double
            s0, L = min(v, key=lambda x: score[o, x[0]:x[0] + x[1]].sum())
            ok[o, s0:s0 + L] = False

    seen, out = set(), []  # repli : un seul créneau par objet
    for o, s0, L in segs:
        if o not in seen:
            seen.add(o)
            out.append((o, s0, L))
    return out


def refine_segments(data, res, segs, max_len=None):
    """Bornes exactes à la minute : l'objet doit être dans la zone à *chaque* minute du créneau,
    et le créneau est étiré tant que c'est possible (en gardant 10 min de pointage entre objets)."""
    T = len(data["t_loc"])
    segs = sorted(segs, key=lambda x: x[1])
    out = []
    for n, (o, s0, L) in enumerate(segs):
        okf = data["ok"][int(res.iloc[o]["idx"])]
        fs, fe = s0 * STRIDE, min((s0 + L) * STRIDE, T)
        while fs < fe and not okf[fs]:
            fs += 1
        while fe > fs and not okf[fe - 1]:
            fe -= 1
        out.append([o, fs, fe])
    for n, seg in enumerate(out):
        okf = data["ok"][int(res.iloc[seg[0]]["idx"])]
        prev_end = out[n - 1][2] + 10 if n else 0
        next_start = out[n + 1][1] - 10 if n + 1 < len(out) else T
        room = lambda: max_len is None or seg[2] - seg[1] < max_len  # noqa: E731
        while seg[1] > prev_end and okf[seg[1] - 1] and room():
            seg[1] -= 1
        while seg[2] < next_start and seg[2] < T and okf[seg[2]] and room():
            seg[2] += 1
    return [tuple(x) for x in out if x[2] > x[1]]


def plan_table(data, res, segs) -> pd.DataFrame:
    t_loc, T = data["t_loc"], len(data["t_loc"])
    rows = []
    for o, fs, fe in segs:
        r = res.iloc[o]
        i = int(r["idx"])
        end = t_loc[fe] if fe < T else t_loc[-1] + timedelta(minutes=1)
        mid = (fs + fe - 1) // 2
        moon = f"{data['sep'][i, mid]:.0f}°" if data["moon_alt"][mid] > 0 else "couchée"
        rows.append({
            "Créneau": f"{t_loc[fs]:%H:%M} → {end:%H:%M}",
            "Durée": fmt_dur(fe - fs),
            "Objet": r["Objet"], "Nom": r["Nom"], "Type": r["Type"], "Mag": r["Mag"],
            "Altitude": f"{data['alt'][i, fs]:.1f}° → {data['alt'][i, fe - 1]:.1f}°",
            "Azimut": f"{data['az'][i, fs]:.1f}° → {data['az'][i, fe - 1]:.1f}°",
            "Dir": az_to_cardinal(data["az"][i, mid]),
            "Lune": moon,
            "Visible de": r["Visible"], "Filtre": r["Filtre"], "Cadrage": r["Cadrage"],
            "_i": i, "_start": t_loc[fs], "_end": end,
        })
    return pd.DataFrame(rows)


def plan_chart(data, plan: pd.DataFrame):
    fig = go.Figure()
    _moon_bands(fig, data)
    palette = ["#7aa2ff", "#b48cff", "#5eead4", "#ffb86b", "#ff7ab6", "#9be15d", "#67d4ff", "#ffd166"]
    order = []
    for n, r in plan.iterrows():
        label = f"{r['Objet']}"
        order.append(label)
        for a, b in data["intervals"][r["_i"]]:  # fenêtre complète de visibilité (fond pâle)
            fig.add_trace(go.Bar(base=[_n(a)], x=[(b - a + timedelta(minutes=1)).total_seconds() * 1000],
                                 y=[label], orientation="h", marker_color="rgba(150,160,185,0.26)",
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Bar(base=[_n(r["_start"])], x=[(r["_end"] - r["_start"]).total_seconds() * 1000],
                             y=[label], orientation="h", marker_color=palette[n % len(palette)],
                             text=r["Durée"], textposition="inside",
                             hovertemplate=f"{label}<br>{r['Créneau']}<extra></extra>", showlegend=False))
    _now_line(fig, data)
    return _layout(fig, barmode="overlay", xaxis=dict(type="date", tickformat="%H:%M"),
                   yaxis=dict(categoryorder="array", categoryarray=order, autorange="reversed"),
                   height=max(220, 56 * len(plan) + 80), margin=dict(l=10, r=10, t=10, b=30))


def multipass_plan(data, res, flt, style, exclude=()):
    """Plan complet : le style choisi en priorité, puis passes plus courtes pour combler les trous."""
    styles = list(PLAN_STYLES)
    blocked = np.zeros(-(-len(data["t_loc"]) // STRIDE), bool)
    taken, segs, used = set(exclude), [], None
    for name in styles[styles.index(style):]:
        mn, mx = PLAN_STYLES[name]
        new = build_plan(data, res, flt, mn, mx, blocked=blocked, exclude=taken)
        if new:
            used = used or name
            for o, s0, L in new:
                blocked[max(s0 - GAP, 0):s0 + L + GAP] = True
                taken.add(o)
            segs += new
    return sorted(segs, key=lambda x: x[1]), used


def generate_plans(data, res, flt, style, n_plans):
    """Plans alternatifs : chaque plan n'utilise aucun objet des plans précédents."""
    plans, used_objs = [], set()
    for n in range(n_plans):
        segs, used = multipass_plan(data, res, flt, style, exclude=used_objs)
        if not segs:
            break
        plans.append({"name": chr(65 + n), "segs": segs, "style": used})
        used_objs |= {o for o, _, _ in segs}
    return plans


def optimize_selected(data, res, flt, rows, min_steps, max_steps, gap=GAP):
    """Meilleur enchaînement des objets choisis (`rows` = lignes de res) : chacun au plus une fois,
    le plus d'objets possible d'abord, puis le meilleur score. Programmation dynamique sur sous-ensembles."""
    ok, quality, _, T = coarse_arrays(data, res, flt)
    ok, q = ok[rows], quality[rows]
    k, NEG = len(rows), -1e9
    cum = np.concatenate([np.zeros((k, 1)), np.cumsum(np.where(ok, q, 0.0), axis=1)], axis=1)
    run = np.zeros((k, T + 1), int)
    for t in range(T - 1, -1, -1):
        run[:, t] = np.where(ok[:, t], run[:, t + 1] + 1, 0)

    full = 1 << k
    f = np.full((full, T + 1), NEG)
    f[0, :] = 0.0
    chO, chL = np.full((full, T + 1), -1), np.zeros((full, T + 1), int)
    for mask in range(1, full):
        cur, co, cl = np.full(T + 1, NEG), np.full(T + 1, -1), np.zeros(T + 1, int)
        for o in range(k):
            if not (mask >> o) & 1:
                continue
            prev = mask ^ (1 << o)
            for L in range(min_steps, min(max_steps, T) + 1):
                s0 = np.arange(0, T - L + 1)
                valid = run[o, s0] >= L
                if not valid.any():
                    break
                pv = f[prev, np.maximum(s0 - gap, 0)]
                valid &= pv > NEG / 2
                # rendement décroissant : encourage à répartir le temps entre les objets
                cand = np.where(valid, pv + (cum[o, s0 + L] - cum[o, s0]) * (1 - 0.25 * L / max_steps), NEG)
                e = s0 + L
                upd = cand > cur[e]
                cur[e[upd]], co[e[upd]], cl[e[upd]] = cand[upd], o, L
        for e in range(1, T + 1):  # le dernier créneau finit au plus tard à e
            if cur[e - 1] > cur[e]:
                cur[e], co[e] = cur[e - 1], -2
        f[mask], chO[mask], chL[mask] = cur, co, cl

    feasible = [m for m in range(1, full) if f[m, T] > NEG / 2]
    if not feasible:
        return [], list(rows)
    best = max(feasible, key=lambda m: (bin(m).count("1"), f[m, T]))
    segs, mask, e = [], best, T
    while mask:
        if chO[mask, e] == -2:
            e -= 1
            continue
        o, L = int(chO[mask, e]), int(chL[mask, e])
        segs.append((rows[o], e - L, L))
        mask ^= 1 << o
        e = max(e - L - gap, 0)
    dropped = [rows[o] for o in range(k) if not (best >> o) & 1]
    return sorted(segs, key=lambda x: x[1]), dropped


def validate_custom(data, res, rows_df, lab2i):
    """Contrôle d'un plan saisi à la main. Retourne (tableau de contrôle, créneaux valides)."""
    t_loc, T = data["t_loc"], len(data["t_loc"])
    label2o = {r["Objet"]: o for o, r in res.iterrows()}
    items = []
    for _, r in rows_df.iterrows():
        name = r.get("Objet")
        a, b = lab2i.get(str(r.get("Début")).strip()), lab2i.get(str(r.get("Fin")).strip())
        err = None
        if not name or name not in label2o:
            err = "❌ choisis un objet"
        elif a is None or b is None:
            err = "❌ heure invalide (format HH:MM, pendant la nuit)"
        elif b <= a:
            err = "❌ la fin doit être après le début"
        else:
            i = int(res.iloc[label2o[name]]["idx"])
            bad = np.where(~data["ok"][i][a:b])[0]
            if len(bad):
                wins = ", ".join(f"{x:%H:%M}→{y:%H:%M}" for x, y in data["intervals"][i])
                err = f"❌ hors de ta zone à {t_loc[a + bad[0]]:%H:%M} (fenêtres possibles : {wins})"
        items.append({"name": name, "a": a, "b": b, "err": err, "warn": []})

    good = sorted([it for it in items if not it["err"]], key=lambda it: it["a"])
    for p_, c_ in zip(good, good[1:]):
        if c_["a"] < p_["b"]:
            c_["err"] = f"❌ chevauche {p_['name']} ({t_loc[p_['b'] - 1]:%H:%M})"
        elif c_["a"] - p_["b"] < 10:
            c_["warn"].append(f"⚠ seulement {c_['a'] - p_['b']} min pour pointer depuis {p_['name']}")
    seen = set()
    for it in good:
        if it["err"]:
            continue
        if it["name"] in seen:
            it["warn"].append("⚠ objet déjà dans le plan")
        seen.add(it["name"])
        if it["b"] - it["a"] < 20:
            it["warn"].append("⚠ pose très courte")

    rows, segs = [], []
    for it in items:
        ok_ = not it["err"]
        span = (f"{t_loc[it['a']]:%H:%M} → " + (f"{t_loc[it['b']]:%H:%M}" if it["b"] < T else
                f"{t_loc[-1] + timedelta(minutes=1):%H:%M}")) if it["a"] is not None and it["b"] is not None else ""
        rows.append({"Objet": it["name"] or "", "Créneau": span,
                     "Durée": fmt_dur(it["b"] - it["a"]) if it["a"] is not None and it["b"] is not None and it["b"] > it["a"] else "",
                     "Contrôle": it["err"] or ("; ".join(it["warn"]) if it["warn"] else "✅ OK")})
        if ok_:
            segs.append((label2o[it["name"]], it["a"], it["b"]))
    return pd.DataFrame(rows), sorted(segs, key=lambda x: x[1])


# --------------------------------------------------------------------------- #
# Message WhatsApp + envoi direct
# --------------------------------------------------------------------------- #
def build_message(plan: pd.DataFrame, ctx: dict, emoji: bool = True) -> str:
    e = (lambda sym: sym + " ") if emoji else (lambda sym: "")
    head = [
        f"{e('🔭')}*Plan de nuit{ctx.get('plan', '')} – {ctx['city']}*",
        f"{e('📅')}{WEEKDAYS[ctx['night'].weekday()]} {ctx['night']:%d/%m/%Y} · nuit {ctx['t0']:%H:%M} → {ctx['t1']:%H:%M} (heure locale, {ctx['tz']})",
        f"{e('🌙')}Lune éclairée à {ctx['illum']:.0f} %",
        f"{e('🧭')}Zone : {ctx['zone']}",
        f"{e('🔶')}Filtre : {ctx['flt']}",
        "",
    ]
    lines = []
    for n, r in plan.iterrows():
        mag = f"mag {r['Mag']:.1f}" if pd.notna(r["Mag"]) else "mag ?"
        name = f" {r['Nom']}" if r["Nom"] else ""
        lines.append(f"{n + 1}. {e('⏱')}*{r['Créneau']}* ({r['Durée']})\n"
                     f"    *{r['Objet']}*{name} – {r['Type']} · {mag}\n"
                     f"    alt {r['Altitude']} · az {r['Azimut']} ({r['Dir']})")
    return "\n".join(head + lines)


# --------------------------------------------------------------------------- #
# Utilitaires d'affichage
# --------------------------------------------------------------------------- #
def col_config(df: pd.DataFrame) -> dict:
    """Largeur de chaque colonne ajustée à son contenu (en pixels)."""
    out = {}
    for c in df.columns:
        lens = [len(str(c)) + 4] + [len(str(v)) for v in df[c].head(300) if pd.notna(v)]
        out[c] = st.column_config.Column(width=int(min(max(max(lens) * 8 + 24, 56), 460)))
    return out




def send_callmebot(phone: str, apikey: str, text: str):
    r = requests.get("https://api.callmebot.com/whatsapp.php",
                     params={"phone": phone, "text": text, "apikey": apikey}, timeout=30)
    return r.ok, r.text[:300]


def send_cloud_api(token: str, phone_id: str, to: str, text: str, version: str = ""):
    base = "https://graph.facebook.com/" + (version.strip("/") + "/" if version.strip() else "")
    r = requests.post(f"{base}{phone_id}/messages", headers={"Authorization": f"Bearer {token}"},
                      json={"messaging_product": "whatsapp", "to": to.replace("+", "").replace(" ", ""),
                            "type": "text", "text": {"body": text[:4000]}}, timeout=30)
    return r.ok, r.text[:300]



# --------------------------------------------------------------------------- #
# Interface (pensée pour le téléphone : onglets, boutons larges, cartes)
# --------------------------------------------------------------------------- #
import html as _html


_STARS = ("radial-gradient(1px 1px at 12% 18%, rgba(255,255,255,.55), transparent),"
          "radial-gradient(1px 1px at 38% 72%, rgba(255,255,255,.40), transparent),"
          "radial-gradient(1.5px 1.5px at 66% 34%, rgba(255,255,255,.50), transparent),"
          "radial-gradient(1px 1px at 83% 81%, rgba(255,255,255,.35), transparent),"
          "radial-gradient(1px 1px at 92% 12%, rgba(255,255,255,.50), transparent),"
          "radial-gradient(1px 1px at 24% 90%, rgba(255,255,255,.35), transparent),"
          "radial-gradient(1.5px 1.5px at 51% 8%, rgba(255,255,255,.45), transparent)")


def theme_css(light: bool = False) -> str:
    """Deux palettes « argent & bleu nuit » (sombre / clair), indépendantes du thème Streamlit."""
    if not light:
        v = dict(panel="rgba(120,150,255,.07)", panel2="rgba(120,150,255,.13)", line="rgba(150,175,255,.17)",
                 line2="rgba(170,190,255,.38)", field="rgba(120,150,255,.10)",
                 text="#e6ecff", muted="#8e9bc4", accent="#7aa2ff", accent2="#9b8cff",
                 side="rgba(9,15,38,.96)", strong="#ffffff", shadow="rgba(3,8,30,.40)", pop="#0f1936",
                 codebg="rgba(5,9,24,.7)", flip="none",
                 sheen="linear-gradient(145deg, rgba(255,255,255,.16) 0%, rgba(255,255,255,.03) 38%, rgba(255,255,255,.08) 100%)",
                 hi="rgba(255,255,255,.30)", shine="rgba(255,255,255,.24)", glow="rgba(122,162,255,.55)",
                 silver="linear-gradient(180deg,#ffffff 0%,#cdd6ef 48%,#8d9cc6 100%)",
                 btn="linear-gradient(180deg,#ffffff 0%,#dde4f5 48%,#aab6d3 100%)", btn_fg="#0a1230",
                 btn_edge="rgba(255,255,255,.55)",
                 bg=("radial-gradient(1200px 560px at 82%% -10%%, rgba(70,110,255,.30), transparent 62%%),"
                     "radial-gradient(900px 520px at -8%% 6%%, rgba(120,90,255,.18), transparent 58%%),"
                     "radial-gradient(1000px 500px at 50%% 115%%, rgba(30,120,200,.16), transparent 60%%),"
                     f"{_STARS}, linear-gradient(180deg, #0b1533 0%%, #070d24 55%%, #050a1c 100%%)"))
    else:
        v = dict(panel="rgba(255,255,255,.80)", panel2="#ffffff", line="rgba(30,50,110,.14)",
                 line2="rgba(30,50,110,.34)", field="#ffffff",
                 text="#0e1633", muted="#5a678a", accent="#3a62ee", accent2="#7a4df0",
                 side="rgba(246,249,255,.97)", strong="#0a1230", shadow="rgba(40,70,160,.14)", pop="#ffffff",
                 codebg="rgba(232,238,255,.9)", flip="invert(1) hue-rotate(180deg)",
                 sheen="linear-gradient(145deg, rgba(255,255,255,.98) 0%, rgba(222,230,248,.60) 46%, rgba(255,255,255,.85) 100%)",
                 hi="rgba(255,255,255,1)", shine="rgba(255,255,255,.85)", glow="rgba(58,98,238,.35)",
                 silver="linear-gradient(180deg,#2b3a78 0%,#4b5c9e 50%,#7e8fc4 100%)",
                 btn="linear-gradient(180deg,#ffffff 0%,#e1e8f7 46%,#b7c3df 100%)", btn_fg="#0a1230",
                 btn_edge="rgba(30,50,110,.28)",
                 bg=("radial-gradient(1100px 520px at 85%% -10%%, rgba(70,110,255,.16), transparent 60%%),"
                     "radial-gradient(800px 480px at -5%% 0%%, rgba(150,110,255,.10), transparent 55%%),"
                     "linear-gradient(180deg, #f5f8ff 0%%, #e8eefb 100%%)"))
    v = {k: s.replace("%%", "%") for k, s in v.items()}  # (les %% ci-dessus sont des % littéraux)
    css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@200;300;400;500;600&display=swap');
:root{--panel:%(panel)s;--panel2:%(panel2)s;--line:%(line)s;--line2:%(line2)s;--field:%(field)s;--text:%(text)s;
      --muted:%(muted)s;--accent:%(accent)s;--accent2:%(accent2)s;--strong:%(strong)s;--shadow:%(shadow)s;
      --sheen:%(sheen)s;--hi:%(hi)s;--shine:%(shine)s;--glow:%(glow)s;--silver:%(silver)s}
.stApp{background:%(bg)s;background-attachment:fixed;color:var(--text)}
.stApp,.stApp p,.stApp label,.stApp button,.stApp input,.stApp textarea,.stApp li,.stApp small,
.stApp h1,.stApp h2,.stApp h3,.stApp h4{font-family:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif}
.stApp p,.stApp li,.stApp label,[data-testid="stWidgetLabel"] *,[data-testid="stMarkdownContainer"] *{color:var(--text)}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] *{color:var(--muted)!important}
[data-testid="stHeader"]{background:transparent}
[data-testid="stHeader"] *,[data-testid="stToolbar"] *{color:var(--text)}
footer{visibility:hidden}
.block-container{padding-top:3.5rem;max-width:1180px}
[data-testid="stSidebar"]{background:%(side)s;border-right:1px solid var(--line);
  box-shadow:inset -1px 0 0 var(--hi)}
[data-testid="stSidebar"] .block-container{padding-top:2rem}
hr{border-color:var(--line)!important}
/* onglets */
[data-baseweb="tab-list"]{gap:.2rem;border-bottom:1px solid var(--line)}
button[role="tab"]{font-size:.76rem;letter-spacing:.16em;text-transform:uppercase;font-weight:500;
  color:var(--muted)!important;padding:.85rem 1rem}
button[role="tab"][aria-selected="true"],button[role="tab"][aria-selected="true"] *{color:var(--text)!important}
[data-baseweb="tab-highlight"]{background:linear-gradient(90deg,var(--accent2),var(--accent),var(--accent2));
  height:2px;box-shadow:0 0 12px var(--glow)}
/* boutons : verre brillant, et bouton principal en argent brossé */
.stButton>button,.stDownloadButton>button,[data-testid^="stBaseLinkButton"]{border-radius:999px;
  border:1px solid var(--line2)!important;background:var(--sheen),var(--panel)!important;color:var(--text);
  font-weight:500;letter-spacing:.03em;min-height:46px;transition:all .2s ease;
  box-shadow:inset 0 1px 0 var(--hi),0 4px 14px var(--shadow)}
.stButton>button:hover,.stDownloadButton>button:hover,[data-testid^="stBaseLinkButton"]:hover{
  border-color:var(--accent)!important;box-shadow:inset 0 1px 0 var(--hi),0 0 16px var(--glow);color:var(--text)}
[data-testid="stBaseButton-primary"],[data-testid="stBaseLinkButton-primary"]{background:%(btn)s!important;
  color:%(btn_fg)s!important;border:1px solid %(btn_edge)s!important;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.95),inset 0 -1px 0 rgba(0,0,0,.10),0 6px 18px var(--shadow)!important}
[data-testid="stBaseButton-primary"] *,[data-testid="stBaseLinkButton-primary"] *{color:%(btn_fg)s!important}
/* pastilles (directions, catégories) */
[data-testid="stBaseButton-pills"],[data-testid="stBaseButton-pillsActive"]{border-radius:999px!important;
  font-size:.8rem;min-height:34px;padding:.1rem .8rem;transition:all .15s ease}
[data-testid="stBaseButton-pills"]{background:var(--field)!important;border:1px solid var(--line2)!important;
  color:var(--muted)!important}
[data-testid="stBaseButton-pills"] *{color:var(--muted)!important}
[data-testid="stBaseButton-pillsActive"]{background:var(--sheen),rgba(122,162,255,.28)!important;
  border:1px solid var(--accent)!important;color:var(--text)!important;box-shadow:0 0 12px var(--glow)}
[data-testid="stBaseButton-pillsActive"] *{color:var(--text)!important}
/* champs de saisie */
[data-baseweb="input"],[data-baseweb="base-input"],[data-baseweb="textarea"],[data-baseweb="select"]>div{
  background:var(--field)!important;border-color:var(--line2)!important;border-radius:12px!important}
[data-baseweb="select"] div{background-color:transparent}
[data-baseweb="select"]>div{background:var(--field)!important}
[data-baseweb="input"] input,[data-baseweb="textarea"] textarea{background:transparent!important;
  color:var(--text)!important;-webkit-text-fill-color:var(--text)}
[data-baseweb="select"] *{color:var(--text)!important}
[data-baseweb="select"] svg,[data-testid="stDateInput"] svg{fill:var(--muted)!important;color:var(--muted)!important}
[data-testid="stNumberInputStepDown"],[data-testid="stNumberInputStepUp"]{background:var(--field)!important;
  color:var(--text)!important}
[data-baseweb="popover"] [data-baseweb="menu"],[data-baseweb="popover"] ul{background:%(pop)s!important}
[data-baseweb="popover"] li,[data-baseweb="popover"] li *{color:var(--text)!important}
/* boutons radio / cases à cocher : cercle et carré non sélectionnés */
label[data-baseweb="radio"]:has(input:not(:checked))>div:first-child{background:var(--field)!important;
  border-color:var(--line2)!important}
label[data-baseweb="checkbox"]:has(input:not(:checked))>span:first-child{background:var(--field)!important;
  border-color:var(--line2)!important}
/* panneaux */
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:16px;
  background:var(--sheen),var(--panel);box-shadow:inset 0 1px 0 var(--hi)}
[data-testid="stExpander"] summary *{color:var(--text)!important}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:16px;overflow:hidden}
[data-testid="stDataFrame"],[data-testid="stDataFrameResizable"]{filter:%(flip)s}
[data-testid="stAlert"]{border-radius:16px}
[data-testid="stAlert"] *{color:var(--text)!important}
[data-testid="stCode"],pre{border-radius:16px;background:%(codebg)s!important}
pre,code,[data-testid="stCode"] *{color:var(--text)!important}
[data-testid="stSliderThumbValue"],[data-testid="stSliderThumbValue"] *,[data-testid="stTickBarMin"],
[data-testid="stTickBarMax"]{color:var(--text)!important}
[data-testid="stTable"] table,[data-testid="stTable"] td,[data-testid="stTable"] th{color:var(--text)!important;
  border-color:var(--line)!important}
/* composants maison */
.hero{padding:.2rem 0 1rem}
.eyebrow{font-size:.7rem;letter-spacing:.24em;text-transform:uppercase;color:var(--muted)!important;
  font-weight:500;margin:.2rem 0 .6rem}
.hero .title{font-weight:200;font-size:clamp(2.2rem,6vw,3.6rem);letter-spacing:-.025em;line-height:1.08;
  margin:.1rem 0 .4rem;
  background:linear-gradient(100deg,transparent 38%%,rgba(255,255,255,.95) 50%%,transparent 62%%) 200%% 0/250%% 100%% no-repeat,var(--silver);
  -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;color:transparent;
  animation:titleshine 7s ease-in-out infinite}
@keyframes titleshine{0%%,55%%{background-position:200%% 0,0 0}100%%{background-position:-120%% 0,0 0}}
.hero .lead{color:var(--muted);font-weight:300;font-size:1.02rem;max-width:42rem}
.stats,.pgrid{display:grid;gap:.8rem;margin:.6rem 0 1rem}
.stats{grid-template-columns:repeat(auto-fit,minmax(170px,1fr))}
.pgrid{grid-template-columns:repeat(auto-fit,minmax(250px,1fr))}
.stat,.psum,.slot{position:relative;overflow:hidden;background:var(--sheen),var(--panel);
  border:1px solid var(--line2);border-radius:20px;backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
  box-shadow:inset 0 1px 0 var(--hi),0 10px 30px var(--shadow)}
.stat::after,.psum::after,.slot::after{content:"";position:absolute;top:0;left:-70%%;width:45%%;height:100%%;
  background:linear-gradient(100deg,transparent,var(--shine),transparent);transform:skewX(-20deg);
  opacity:0;pointer-events:none}
.stat:hover::after,.psum:hover::after,.slot:hover::after{animation:sweep 1.2s ease}
@keyframes sweep{0%%{left:-70%%;opacity:.9}100%%{left:140%%;opacity:0}}
.stat,.psum{padding:1rem 1.2rem}
.stat .k,.psum .k{font-size:.66rem;letter-spacing:.2em;text-transform:uppercase;color:var(--muted);font-weight:500}
.stat .v,.psum .v{font-size:1.75rem;font-weight:200;line-height:1.25;letter-spacing:-.01em;
  background:var(--silver);-webkit-background-clip:text;background-clip:text;
  -webkit-text-fill-color:transparent;color:transparent}
.stat .s,.psum .s{font-size:.8rem;color:var(--muted);font-weight:300;margin-top:.1rem}
.slot{padding:1.1rem 1.3rem;margin:.7rem 0}
.slot .top{display:flex;justify-content:space-between;align-items:baseline;gap:.6rem;flex-wrap:wrap}
.slot .time{font-size:1.5rem;font-weight:200;letter-spacing:-.01em;background:var(--silver);
  -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;color:transparent}
.slot .dur{color:var(--muted);font-size:.85rem;font-weight:400}
.slot .name{font-size:1.1rem;font-weight:500;margin-top:.35rem;color:var(--strong)}
.slot .name small{color:var(--muted);font-weight:300;margin-left:.5rem;font-size:.85rem}
.slot .sub{color:var(--muted);font-size:.86rem;font-weight:300}
.nightbar{position:relative;height:4px;border-radius:4px;background:var(--line);margin:.9rem 0 .7rem}
.nightbar i{position:absolute;top:0;bottom:0;border-radius:4px;
  background:linear-gradient(90deg,var(--accent),#e9eeff 50%%,var(--accent2));box-shadow:0 0 10px var(--glow)}
.chip{display:inline-block;padding:.18rem .7rem;border-radius:999px;border:1px solid var(--line2);
  background:var(--sheen);font-size:.76rem;color:var(--muted);margin:.18rem .3rem .18rem 0;white-space:nowrap;
  box-shadow:inset 0 1px 0 var(--hi)}
.chip.hl{color:var(--accent);border-color:var(--accent);box-shadow:inset 0 1px 0 var(--hi),0 0 10px var(--glow)}
@media (max-width:640px){
  .block-container{padding:3.2rem .7rem 3rem}
  .hero .title{font-size:2.3rem}
  .stat .v,.psum .v{font-size:1.45rem}
  .slot{padding:1rem}
  .slot .time{font-size:1.3rem}
}
""" % v
    return "<style>" + css + "</style>"


# Thème Streamlit « bleu nuit » (identique pour tous les visiteurs) : appliqué au démarrage du serveur,
# au besoin avec un unique rechargement. Le fichier .streamlit/config.toml fait la même chose.
_THEME = {"theme.base": "dark", "theme.primaryColor": "#7aa2ff", "theme.backgroundColor": "#080e22",
          "theme.secondaryBackgroundColor": "#0f1936", "theme.textColor": "#e6ecff"}
_changed = False
for _k, _v in _THEME.items():
    if config.get_option(_k) != _v:
        config.set_option(_k, _v)
        _changed = True
if _changed and not st.session_state.get("_themed"):
    st.session_state["_themed"] = True
    st.rerun()

# Choix du thème (par visiteur) : en tête du menu Réglages ; mémorisé dans l'URL (?theme=light|dark)
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "Clair" if st.query_params.get("theme") == "light" else "Sombre"
with st.sidebar:
    st.markdown('<div class="eyebrow">Réglages</div>', unsafe_allow_html=True)
    st.radio("Thème", ["Sombre", "Clair"], horizontal=True, key="theme_mode")
_want = "light" if st.session_state["theme_mode"] == "Clair" else "dark"
if st.query_params.get("theme") != _want:
    st.query_params["theme"] = _want

st.markdown(theme_css(light=_want == "light"), unsafe_allow_html=True)
st.markdown('<div class="hero"><div class="eyebrow">Vespera Pro 2 · Planificateur de nuit</div>'
            '<div class="title">Ciel visible</div>'
            '<div class="lead">Dessine ta zone de ciel dégagée et obtiens un plan de nuit séquencé : '
            'quel objet, à quelle heure, dans quelle direction.</div></div>', unsafe_allow_html=True)
header = st.container()  # rempli après le calcul (nuit, Lune, filtre)
tab_sky, tab_obj, tab_plans, tab_share = st.tabs(["Ciel", "Objets", "Plans", "Partage"])


def _secret(key, default):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


# =========================== ⚙️ RÉGLAGES (menu de gauche) ===================
with st.sidebar:
    st.markdown('<div class="eyebrow">Lieu & nuit</div>', unsafe_allow_html=True)
    city = st.text_input("Ville", "Nanterre")
    try:
        places = geocode(city) if city.strip() else []
    except Exception:
        places = None  # pas de réseau → saisie manuelle
    if places is None or not places:
        st.warning("Géocodage indisponible ou ville introuvable : saisie manuelle.")
        lat = st.number_input("Latitude", -90.0, 90.0, 48.8924, format="%.5f")
        lon = st.number_input("Longitude", -180.0, 180.0, 2.2071, format="%.5f")
        elev = st.number_input("Altitude (m)", 0, 5000, 40)
        tz_name = st.text_input("Fuseau horaire", "Europe/Paris")
        p = {"name": city or "Lieu"}
    else:
        labels = [f"{q['name']}, {q.get('admin1', '')} ({q.get('country_code', '')})" for q in places]
        p = places[st.selectbox("Résultat", range(len(places)), format_func=lambda i: labels[i])]
        lat, lon, elev = p["latitude"], p["longitude"], p.get("elevation", 0) or 0
        tz_name = p.get("timezone", "UTC")
        with st.expander("Coordonnées exactes de mon site (optionnel)"):
            st.caption("Le géocodage donne le centre de la ville ; saisis tes coordonnées pour plus de précision.")
            if st.checkbox("Utiliser mes coordonnées"):
                lat = st.number_input("Latitude", -90.0, 90.0, float(lat), format="%.5f")
                lon = st.number_input("Longitude", -180.0, 180.0, float(lon), format="%.5f")
                elev = st.number_input("Altitude (m)", 0, 5000, int(elev))

    # Nuit par défaut d'après l'heure du site (avant 6 h on est encore dans la nuit d'hier soir)
    _now = datetime.now(ZoneInfo(tz_name)) if tz_name else datetime.now()
    night = st.date_input("Nuit du (soir)", _now.date() - timedelta(days=1) if _now.hour < 6 else _now.date())
    twilight = st.radio("Début de nuit", ["Astronomique (−18°)", "Nautique (−12°)", "Civil (−6°)"], horizontal=True)
    sun_limit = {"A": -18.0, "N": -12.0, "C": -6.0}[twilight[0]]
    with st.expander("Précision du calcul"):
        temp_c = st.number_input("Température (°C)", -30, 40, 10,
                                 help="Sert à la réfraction atmosphérique (quelques minutes d'arc à 20°).")

    st.divider()
    st.markdown('<div class="eyebrow">Ma zone de ciel</div>', unsafe_allow_html=True)
    dirs = st.pills("Directions visibles", SECTORS, selection_mode="multi", default=DEFAULT_DIRS) or []
    amin, amax = st.slider("Hauteur visible (°)", 0, 90, (20, 50),
                           help="Altitude minimale (arbres, immeubles) et maximale (balcon, toit…).")
    table = pd.DataFrame({
        "Secteur": SECTORS, "Visible": [s_ in dirs for s_ in SECTORS],
        "Alt min": float(amin), "Alt max": float(amax),
    })
    with st.expander("Réglage fin par secteur (arbres, immeubles…)"):
        st.caption("Chaque secteur = 22,5° d'azimut. Ajuste l'altitude min/max par direction.")
        table = st.data_editor(table, hide_index=True, width="stretch", disabled=["Secteur"],
                               key=f"tbl_{'-'.join(dirs)}_{amin}_{amax}")
    intervals = intervals_from_sectors(table)
    with st.expander("Mode précis : intervalles d'azimut en degrés"):
        st.caption("Nord = 0°, Est = 90°, Sud = 180°, Ouest = 270°. Ajoute autant de lignes que nécessaire.")
        use_prec = st.checkbox("Utiliser ces intervalles à la place des secteurs")
        prec_df = pd.DataFrame(intervals, columns=["Az début (°)", "Az fin (°)", "Alt min (°)", "Alt max (°)"])
        prec_df = st.data_editor(prec_df, num_rows="dynamic", hide_index=True, width="stretch",
                                 key=f"prec_{intervals}")
        if use_prec:
            clean = prec_df.dropna()
            intervals = [tuple(float(v) for v in r) for r in clean.itertuples(index=False)
                         if float(r[2]) < float(r[3])]

    st.divider()
    st.markdown('<div class="eyebrow">Objets</div>', unsafe_allow_html=True)
    cats = st.pills("Catégories", list(CATEGORIES), selection_mode="multi", default=["Nébuleuses"]) or []
    flt = st.radio("Filtre optique", FILTERS, index=2, horizontal=True,
                   help="Le Vespera n'a pas de changement de filtre en cours de nuit : "
                        "choisis celui que tu as monté.")
    hide_bad = st.checkbox("Masquer les objets déconseillés avec ce filtre", True)
    mag_limit = st.slider("Magnitude max", 5.0, 16.0, 11.0, 0.5)
    min_size = st.slider("Taille min (')", 0.5, 30.0, 3.0, 0.5)
    no_mosaic = st.checkbox("Exclure ce qui ne tient pas dans le champ natif (1,6°)", False)
    unrated = st.checkbox("Inclure les objets connus sans magnitude", True)
    with st.expander("Instrument : Vespera Pro 2 (source : vaonis.com)"):
        st.table(pd.Series(VESPERA, name="Valeur"))

    st.divider()
    st.markdown('<div class="eyebrow">Plans</div>', unsafe_allow_html=True)
    style = st.radio("Durée par objet", list(PLAN_STYLES), index=0,
                     help="Poses longues : peu d'objets, beaucoup de temps de pose sur chacun.")
    n_plans = st.slider("Nombre de plans alternatifs", 1, 5, 3,
                        help="Chaque plan utilise des objets différents des plans précédents.")
    view = st.radio("Affichage des plans", ["Cartes", "Tableau"], horizontal=True,
                    help="Les cartes sont plus lisibles sur téléphone.")
    st.caption("Le thème clair/sombre suit celui de ton appareil (menu ⋮ → Settings pour le changer).")

# =========================== CALCULS ========================================
if not cats:
    st.info("Choisis au moins une catégorie d'objets dans le menu Réglages (à gauche).")
    st.stop()
if not intervals:
    st.info("Choisis au moins une direction visible dans le menu Réglages (à gauche).")
    st.stop()

mask_key = tuple(tuple(float(x) for x in iv) for iv in intervals)
try:
    data = compute_night(lat, lon, elev, tz_name, night, sun_limit, mask_key,
                         mag_limit, min_size, tuple(cats), unrated, float(temp_c))
except RuntimeError as e:
    st.error(str(e))
    st.stop()

t_loc = data["t_loc"]
T = len(t_loc)
tzinfo = t_loc[0].tzinfo


def _utc(dt):
    off = dt.utcoffset()
    sign = "+" if off >= timedelta(0) else "−"
    off = abs(off)
    return f"UTC{sign}{off.seconds // 3600:02d}:{off.seconds % 3600 // 60:02d}"


now_site = datetime.now(tzinfo)
tz_label = f"{tz_name} · {_utc(t_loc[0])} ({t_loc[0].tzname()})"

res = data["res"].copy()
if not res.empty:
    res["Filtre"] = res["code"].map(lambda c: filter_fit(c, flt))
    res["Cadrage"] = res["Taille (')"].map(framing)
    if hide_bad:
        res = res[res["Filtre"] != "✖ déconseillé"]
    if no_mosaic:
        res = res[res["Cadrage"].isin(["ok", "très petit"])]
    res = res.reset_index(drop=True)
    data["res"] = res
if res.empty:
    st.info("Aucun objet ne traverse ta zone avec ces filtres. Élargis la zone, la magnitude ou change de filtre "
            "dans le menu Réglages (à gauche).")
    st.stop()

# Plans automatiques (mis en cache dans la session tant que les réglages ne changent pas)
sig = (lat, lon, elev, tz_name, str(night), sun_limit, mask_key, tuple(cats), flt, hide_bad, no_mosaic,
       mag_limit, min_size, unrated, float(temp_c), style, n_plans)
if st.session_state.get("_plans_sig") != sig:
    st.session_state["_plans"] = generate_plans(data, res, flt, style, n_plans)
    st.session_state["_plans_sig"] = sig
plans = st.session_state["_plans"]

lab2i = {f"{t:%H:%M}": i for i, t in enumerate(t_loc)}
lab2i[f"{t_loc[-1] + timedelta(minutes=1):%H:%M}"] = T  # fin de nuit (borne exclusive)


def end_label(fe):
    return f"{t_loc[fe]:%H:%M}" if fe < T else f"{t_loc[-1] + timedelta(minutes=1):%H:%M}"


def summarize(plan_df):
    mins = int(((plan_df["_end"] - plan_df["_start"]).dt.total_seconds() / 60).sum())
    return {"Objets": len(plan_df), "Pose totale": fmt_dur(mins),
            "Mag. moyenne": round(float(plan_df["Mag"].mean()), 1) if plan_df["Mag"].notna().any() else None,
            "Parcours": " → ".join(plan_df["Objet"])}


def plan_cards(plan_df):
    """Un créneau = une carte : heure, objet, position dans la nuit, altitude / azimut."""
    t0 = t_loc[0]
    span = (t_loc[-1] + timedelta(minutes=1) - t0).total_seconds()
    for _, r in plan_df.iterrows():
        left = (r["_start"] - t0).total_seconds() / span * 100
        width = max((r["_end"] - r["_start"]).total_seconds() / span * 100, 1.5)
        nom = f"<small>{_html.escape(r['Nom'])}</small>" if r["Nom"] else ""
        mag = f" · mag {r['Mag']:.1f}" if pd.notna(r["Mag"]) else ""
        hl = " hl" if "idéal" in r["Filtre"] else ""
        st.markdown(
            '<div class="slot">'
            f'<div class="top"><span class="time">{r["Créneau"]}</span><span class="dur">{r["Durée"]}</span></div>'
            f'<div class="name">{_html.escape(r["Objet"])}{nom}</div>'
            f'<div class="sub">{_html.escape(r["Type"])}{mag}</div>'
            f'<div class="nightbar"><i style="left:{left:.1f}%;width:{width:.1f}%"></i></div>'
            f'<span class="chip">alt {r["Altitude"]}</span><span class="chip">az {r["Azimut"]}</span>'
            f'<span class="chip">{r["Dir"]}</span><span class="chip">Lune {r["Lune"]}</span>'
            f'<span class="chip{hl}">{_html.escape(r["Filtre"])}</span>'
            f'<span class="chip">cadrage {_html.escape(r["Cadrage"])}</span>'
            f'<span class="chip">visible {r["Visible de"]}</span></div>', unsafe_allow_html=True)


def fmt_mag(df, col="Mag"):
    df = df.copy()
    df[col] = df[col].map(lambda x: "—" if pd.isna(x) else f"{x:.1f}")
    return df


def show_plan(plan_df, key):
    if view == "Cartes":
        plan_cards(plan_df)
    else:
        shown = fmt_mag(plan_df.drop(columns=["_i", "_start", "_end"]))
        st.dataframe(shown, hide_index=True, width="stretch", column_config=col_config(shown))
    st.plotly_chart(plan_chart(data, plan_df), width="stretch", theme=None, config=PLOT_CFG, key=key)


plan_dfs = {}  # nom → tableau du plan (pour le partage)
for pl in plans:
    pl["fine"] = refine_segments(data, res, pl["segs"])
    pl["df"] = plan_table(data, res, pl["fine"])
    plan_dfs[f"Plan {pl['name']}"] = pl["df"]

with header:
    moon_txt = "levée une partie de la nuit" if (data["moon_alt"] > 0).any() else "couchée"
    st.markdown(
        '<div class="stats">'
        f'<div class="stat"><div class="k">Nuit</div><div class="v">{t_loc[0]:%H:%M} → {t_loc[-1]:%H:%M}</div>'
        f'<div class="s">{_html.escape(p["name"])}</div></div>'
        f'<div class="stat"><div class="k">Lune</div><div class="v">{data["illum"] * 100:.0f} %</div>'
        f'<div class="s">{moon_txt}</div></div>'
        f'<div class="stat"><div class="k">Filtre</div><div class="v">{_html.escape(flt.split(" (")[0])}</div>'
        '<div class="s">Vespera Pro 2</div></div>'
        f'<div class="stat"><div class="k">Heure locale</div><div class="v">{now_site:%H:%M}</div>'
        f'<div class="s">{_utc(t_loc[0])} · {_html.escape(tz_name)}</div></div></div>', unsafe_allow_html=True)
    st.caption("Heures locales du lieu · positions apparentes (réfraction incluse) · calcul à la minute.")
    if data["used_limit"] != sun_limit:
        st.warning(f"Pas de nuit {twilight.split(' ')[0].lower()} à cette date : "
                   f"j'utilise le Soleil à {data['used_limit']:.0f}° sous l'horizon.")


# =========================== 🗓️ PLANS =======================================
with tab_plans:
    if plans:
        if plans[0]["style"] != style:
            st.warning(f"Aucun créneau possible en « {style} » : plans calculés en « {plans[0]['style']} ».")
        cards = "".join(
            f'<div class="psum"><div class="k">Plan {pl["name"]}</div><div class="v">{summarize(pl["df"])["Pose totale"]}</div>'
            f'<div class="s">{summarize(pl["df"])["Objets"]} objets · {_html.escape(summarize(pl["df"])["Parcours"])}</div></div>'
            for pl in plans)
        st.markdown(f'<div class="pgrid">{cards}</div>', unsafe_allow_html=True)
        st.caption("Chaque plan utilise des objets différents. Un seul objet à la fois ; à chaque minute l'objet "
                   "est dans ta zone (10 min entre deux objets pour le pointage). Barres pâles = fenêtre complète "
                   "de visibilité ; Lune levée en jaune ; pointillés rouges = maintenant.")
    else:
        st.info("Aucun objet ne reste assez longtemps dans ta zone pour un plan automatique. "
                "Tu peux quand même composer ton plan dans « Mon plan ».")

    subtabs = st.tabs([f"Plan {pl['name']}" for pl in plans] + ["Mon plan"])
    for stab, pl in zip(subtabs, plans):
        with stab:
            show_plan(pl["df"], f"chart_{pl['name']}")

    # ----- Mon plan (créé / ajusté à la main) --------------------------------
    with subtabs[-1]:
        st.markdown("Choisis des objets, laisse l'appli proposer un enchaînement, puis ajuste. "
                    "Chaque créneau est contrôlé (zone dégagée, chevauchement, pointage).")
        st.session_state.setdefault("custom_ver", 0)
        st.session_state.setdefault("custom_rows", pd.DataFrame({c: pd.Series([], dtype=str)
                                                                  for c in ("Objet", "Début", "Fin")}))
        names = list(res["Objet"])

        def _opt(i):
            r = res.iloc[i]
            mag = f"mag {r['Mag']:.1f}" if pd.notna(r["Mag"]) else "mag ?"
            return f"{r['Objet']} {r['Nom']} · {mag} · visible {r['Visible']}".replace("  ", " ")

        picked = st.multiselect("Objets que je veux observer", list(range(len(res))), format_func=_opt,
                                key="custom_pick")
        pc1, pc2 = st.columns(2)
        min_pose = pc1.slider("Pose mini (min)", 20, 240, 60, 5)
        max_pose = pc2.slider("Pose maxi (min)", 60, 360, 180, 5)

        def _load(fine):
            st.session_state["custom_rows"] = pd.DataFrame(
                [{"Objet": res.iloc[o]["Objet"], "Début": f"{t_loc[fs]:%H:%M}", "Fin": end_label(fe)}
                 for o, fs, fe in fine])
            st.session_state["custom_ver"] += 1

        if st.button("Proposer un enchaînement", disabled=not picked, width="stretch"):
            if len(picked) > 8:
                st.error("Pour l'optimisation, choisis 8 objets au maximum.")
            else:
                segs_c, dropped = optimize_selected(data, res, flt, picked, max(min_pose // STRIDE, 1),
                                                    max(max_pose // STRIDE, 2))
                if not segs_c:
                    st.error("Impossible de placer ces objets avec ces durées : ils ne sont jamais "
                             "dans ta zone assez longtemps. Réduis la pose mini ou change d'objets.")
                else:
                    _load(refine_segments(data, res, segs_c, max_len=max_pose))
                    if dropped:
                        st.warning("Pas la place pour : " + ", ".join(res.iloc[o]["Objet"] for o in dropped) + ".")
                    st.rerun()
        if plans:
            lc1, lc2 = st.columns([3, 2])
            base_plan = lc1.selectbox("Partir d'un plan existant", [f"Plan {pl['name']}" for pl in plans],
                                      index=None, placeholder="Partir d'un plan existant…",
                                      label_visibility="collapsed")
            if lc2.button("Charger", disabled=base_plan is None, width="stretch"):
                _load(next(pl["fine"] for pl in plans if f"Plan {pl['name']}" == base_plan))
                st.rerun()

        # -- ajout d'un créneau (formulaire adapté au tactile)
        with st.expander("Ajouter un créneau"):
            obj_add = st.selectbox("Objet", names, key="add_obj")
            oi = int(res.loc[res["Objet"] == obj_add, "idx"].iloc[0])
            wins = ", ".join(f"{a:%H:%M}→{b:%H:%M}" for a, b in data["intervals"][oi])
            st.caption(f"Fenêtre possible dans ta zone : {wins}")
            with st.form("add_slot"):
                f1, f2 = st.columns(2)
                t_start = f1.time_input("Début", value=data["intervals"][oi][0][0].time(), step=60)
                t_end = f2.time_input("Fin", value=data["intervals"][oi][0][1].time(), step=60)
                if st.form_submit_button("Ajouter au plan", width="stretch"):
                    rows_now = st.session_state["custom_rows"].astype(str)
                    st.session_state["custom_rows"] = pd.concat(
                        [rows_now, pd.DataFrame([{"Objet": obj_add, "Début": f"{t_start:%H:%M}",
                                                  "Fin": f"{t_end:%H:%M}"}])], ignore_index=True)
                    st.session_state["custom_ver"] += 1
                    st.rerun()

        rows_df = st.session_state["custom_rows"].astype(str)
        rows_df = rows_df[rows_df["Objet"].isin(names)].reset_index(drop=True)
        with st.expander("Modifier les créneaux dans un tableau", expanded=bool(len(rows_df))):
            st.caption("Heures au format HH:MM (heure locale, fin exclue). La corbeille / le « + » du tableau "
                       "suppriment ou ajoutent des lignes.")
            edited = st.data_editor(
                rows_df, num_rows="dynamic", hide_index=True, width="stretch",
                key=f"custom_ed_{st.session_state['custom_ver']}",
                column_config={"Objet": st.column_config.SelectboxColumn("Objet", options=names, required=True),
                               "Début": st.column_config.TextColumn("Début", required=True),
                               "Fin": st.column_config.TextColumn("Fin", required=True)})
        st.session_state["custom_rows"] = edited.reset_index(drop=True)

        if len(edited):
            if st.button("Tout effacer", width="stretch"):
                st.session_state["custom_rows"] = rows_df.iloc[0:0]
                st.session_state["custom_ver"] += 1
                st.rerun()
            check, csegs = validate_custom(data, res, edited, lab2i)
            st.markdown("**Contrôle**")
            for _, r in check.iterrows():
                st.markdown(f"{r['Contrôle']}  \n<small>{r['Objet']} · {r['Créneau']} · {r['Durée']}</small>",
                            unsafe_allow_html=True)
            if csegs:
                cdf = plan_table(data, res, csegs)
                plan_dfs["Mon plan"] = cdf
                st.markdown("**Mon plan** — " + " · ".join(f"{k_}: {v_}" for k_, v_ in summarize(cdf).items()
                                                             if k_ != "Parcours"))
                show_plan(cdf, "chart_custom")
        else:
            st.info("Choisis des objets puis « Proposer un enchaînement », ou charge un plan existant.")

# =========================== 🌌 CIEL ========================================
with tab_sky:
    slider_opts = list(range(0, T, 5))
    default_k = (min(slider_opts, key=lambda i: abs((t_loc[i] - now_site).total_seconds()))
                 if t_loc[0] <= now_site <= t_loc[-1] else slider_opts[len(slider_opts) // 2])
    k = st.select_slider("Heure (locale)", options=slider_opts, value=default_k,
                         format_func=lambda i: f"{t_loc[i]:%H:%M}")
    st.plotly_chart(polar_chart(intervals, data, k), width="stretch", theme=None, config=PLOT_CFG)
    now_ids = [i for i in res["idx"] if data["ok"][i, k]]
    st.markdown(f"**{len(now_ids)} objets dans ta zone à {t_loc[k]:%H:%M}**")
    now = data["sel"].loc[now_ids].copy()
    now["Alt"] = data["alt"][now_ids, k].round(1)
    now["Az"] = data["az"][now_ids, k].round(1)
    now["Dir"] = [az_to_cardinal(a) for a in data["az"][now_ids, k]]
    now = now.sort_values("mag", na_position="last")
    now = now[["label", "common", "mag", "Alt", "Az", "Dir"]].rename(columns={
        "label": "Objet", "common": "Nom", "mag": "Mag"})
    now = fmt_mag(now)
    st.dataframe(now, hide_index=True, width="stretch", height=360, column_config=col_config(now))

# =========================== 📋 OBJETS ======================================
with tab_obj:
    sort_by = st.radio("Trier par", ["Heure de début", "Durée visible", "Luminosité", "Taille"], horizontal=True)
    full_cols = st.checkbox("Toutes les colonnes", False)
    order = {"Heure de début": (["_s", "Mag"], [True, True]),
             "Durée visible": (["Durée (h)", "Mag"], [False, True]),
             "Luminosité": (["Mag"], [True]), "Taille": (["Taille (')"], [False])}[sort_by]
    allres = res.sort_values(order[0], ascending=order[1], na_position="last")
    allshown = allres.drop(columns=["idx", "code", "_s", "_e"])
    if not full_cols:
        allshown = allshown[["Objet", "Nom", "Mag", "Visible", "Durée (h)", "Filtre", "Cadrage"]]
    allshown = fmt_mag(allshown)
    st.dataframe(allshown, hide_index=True, width="stretch", column_config=col_config(allshown))
    chrono = res.sort_values(["_s", "_e"])  # ordre chronologique d'observation
    st.markdown(f"**Tous les objets visibles ({len(chrono)}) sur toute la nuit, dans l'ordre d'observation** "
                "— fond jaune : Lune levée · pointillés rouges : maintenant")
    st.plotly_chart(timeline_chart(data, chrono["idx"].tolist()), width="stretch", theme=None, config=PLOT_CFG)
    st.caption("Catalogue : OpenNGC (CC-BY-SA 4.0, M. Verga) · Éphémérides : astropy · "
               "Caractéristiques instrument : Vaonis · Géocodage : Open-Meteo. "
               "Les recommandations de filtre sont des règles empiriques.")

# =========================== 📲 PARTAGE =====================================
with tab_share:
    if not plan_dfs:
        st.info("Aucun plan à partager pour l'instant.")
    else:
        share_name = st.selectbox("Plan à partager", list(plan_dfs))
        plan = plan_dfs[share_name]
        zone = " ; ".join(f"az {a0:.0f}°→{a1:.0f}° · alt {lo:.0f}°–{hi:.0f}°" for a0, a1, lo, hi in intervals)
        ctx = {"city": p["name"], "night": night, "t0": t_loc[0], "t1": t_loc[-1], "tz": _utc(t_loc[0]),
               "illum": data["illum"] * 100, "zone": zone, "flt": flt, "plan": f" ({share_name})"}
        text_emoji = build_message(plan, ctx, emoji=True)
        text_plain = build_message(plan, ctx, emoji=False)  # le lien wa.me perd les emojis

        st.link_button("Envoyer via WhatsApp", "https://wa.me/?text=" + quote(text_plain),
                       width="stretch", type="primary")
        st.download_button("Télécharger (.txt)", text_emoji, width="stretch",
                           file_name=f"plan_nuit_{night:%Y%m%d}_{share_name.replace(' ', '_')}.txt")
        st.caption("Sur téléphone, le bouton WhatsApp ouvre directement l'appli : tu choisis le contact.")
        st.code(text_emoji, language=None)  # bouton « copier » intégré

        with st.expander("Envoi direct à un contact (sans ouvrir WhatsApp)"):
            st.session_state.setdefault("contacts", [dict(c) for c in _secret("contacts", [])])
            st.session_state.setdefault("cloud", dict(_secret("cloud", {})))
            contacts, cloud = st.session_state["contacts"], st.session_state["cloud"]
            provider = st.radio("Service d'envoi",
                                ["CallMeBot (gratuit, le plus simple)", "WhatsApp Business Cloud API (Meta)"])
            use_cmb = provider.startswith("CallMeBot")
            if use_cmb:
                st.caption("Chaque destinataire doit activer CallMeBot **depuis son propre WhatsApp** "
                           "(procédure sur callmebot.com : on envoie « I allow callmebot to send me messages » "
                           "au bot) et te communiquer la clé API reçue en retour.")
            else:
                st.caption("Nécessite un compte Meta for Developers avec un numéro WhatsApp Business. "
                           "Le texte libre n'est accepté que si le contact a écrit à ton numéro Business "
                           "dans les dernières 24 h (sinon Meta exige un modèle de message approuvé).")
                cloud["token"] = st.text_input("Jeton d'accès", cloud.get("token", ""), type="password")
                cloud["phone_id"] = st.text_input("ID du numéro WhatsApp (Phone number ID)",
                                                  cloud.get("phone_id", ""))
                cloud["version"] = st.text_input("Version de l'API (optionnel, ex. v23.0)",
                                                 cloud.get("version", ""))

            with st.form("add_contact", clear_on_submit=True):
                st.markdown("**Ajouter un contact**")
                c_name = st.text_input("Nom")
                c_phone = st.text_input("Numéro (+33612345678)")
                c_key = st.text_input("Clé API CallMeBot", type="password") if use_cmb else ""
                if st.form_submit_button("Enregistrer le contact", width="stretch") and c_name and c_phone:
                    contacts.append({"name": c_name, "phone": c_phone.replace(" ", ""), "apikey": c_key})
                    st.rerun()

            if contacts:
                who = st.selectbox("Destinataire", range(len(contacts)),
                                   format_func=lambda i: f"{contacts[i]['name']} ({contacts[i]['phone']})")
                if st.button("Envoyer le plan", type="primary", width="stretch"):
                    c = contacts[who]
                    try:
                        if use_cmb:
                            if not c.get("apikey"):
                                raise ValueError("Ce contact n'a pas de clé API CallMeBot.")
                            ok_, msg = send_callmebot(c["phone"], c["apikey"], text_emoji)
                        else:
                            if not (cloud.get("token") and cloud.get("phone_id")):
                                raise ValueError("Renseigne le jeton et l'ID du numéro.")
                            ok_, msg = send_cloud_api(cloud["token"], cloud["phone_id"], c["phone"],
                                                      text_emoji, cloud.get("version", ""))
                        (st.success if ok_ else st.error)(f"{'Envoyé' if ok_ else 'Échec'} : {msg}")
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Échec : {e}")
                if st.button("Supprimer ce contact", width="stretch"):
                    contacts.pop(who)
                    st.rerun()
            st.caption("Contacts et clés ne sont gardés que le temps de ta session (rien n'est enregistré "
                       "sur le serveur) : ils sont à ressaisir si tu recharges la page.")