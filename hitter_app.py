#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hitter Dashboard
Statcast-powered batter analysis app built with Streamlit.
"""

import datetime as dt
import time
import unicodedata
from typing import Any, Optional, Tuple, Dict, List

import numpy as np
import pandas as pd

import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch

import plotly.graph_objects as go

from pybaseball import (
    statcast_batter,
    statcast,
    batting_stats,
    chadwick_register,
)

try:
    from pybaseball import cache as pyb_cache
    pyb_cache.enable()
except Exception:
    pass

try:
    import requests
    _REQ_EXC = (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError)
except Exception:
    _REQ_EXC = (Exception,)

APP_VERSION = "v1.1"

st.set_page_config(
    page_title="Hitter Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
    <style>
      .block-container { padding-top: 1.0rem; padding-bottom: 1.2rem; }
      div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
      .stMetric { border-radius: 12px; }
      h1, h2, h3 { margin-bottom: 0.4rem; }
      .tiny { font-size: 0.86rem; color: #6b7280; line-height: 1.25rem; }
      .muted { color: #6b7280; }
      .smallnote { font-size: 0.82rem; color: #6b7280; }
    </style>
""", unsafe_allow_html=True)

# =========================================================
# Config
# =========================================================
FASTBALLS = {"FF", "SI", "FT", "FC"}
OFFSPEED  = {"CH", "FS", "FO", "SC"}
BREAKING  = {"SL", "CU", "KC", "KN", "SV", "CS", "ST"}

SWING_DESCRIPTIONS = {
    "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
    "hit_into_play", "hit_into_play_no_out", "hit_into_play_score",
    "foul_bunt", "missed_bunt",
}
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked"}
CONTACT_DESCRIPTIONS = {
    "hit_into_play", "hit_into_play_no_out", "hit_into_play_score",
    "foul", "foul_tip",
}

STRIKEZONE = {"x0": -0.83, "z0": 1.5, "w": 1.66, "h": 2.0}
HEART_ZONE = {"x0": -0.56, "z0": 1.75, "w": 1.12, "h": 1.5}

PITCH_COLORS = {
    "FF": "#d62728", "FA": "#d62728", "SI": "#ff7f0e", "FT": "#ff7f0e",
    "FC": "#bcbd22", "CH": "#2ca02c", "FS": "#17becf", "FO": "#17becf",
    "SC": "#17becf", "SL": "#9467bd", "ST": "#8c564b", "SV": "#8c564b",
    "CU": "#1f77b4", "KC": "#1f77b4", "KN": "#1f77b4", "CS": "#1f77b4",
}

PITCH_NAMES = {
    "FF": "4-Seam Fastball", "SI": "Sinker", "FT": "2-Seam Fastball",
    "FC": "Cutter", "CH": "Changeup", "FS": "Splitter", "FO": "Forkball",
    "SC": "Screwball", "SL": "Slider", "ST": "Sweeper", "SV": "Curveball",
    "CU": "Curveball", "KC": "Knuckle Curve", "KN": "Knuckleball", "CS": "Curveball",
}

INVALID_PITCH_TYPES = {"", "None", "nan", "NaN", "PO"}
MIN_BASELINE_PITCHES = 100

TEAM_IDS = {
    "ARI":109,"ATL":144,"BAL":110,"BOS":111,"CHC":112,"CWS":145,"CIN":113,
    "CLE":114,"COL":115,"DET":116,"HOU":117,"KC":118,"LAA":108,"LAD":119,
    "MIA":146,"MIL":158,"MIN":142,"NYM":121,"NYY":147,"OAK":133,"PHI":143,
    "PIT":134,"SD":135,"SEA":136,"SF":137,"STL":138,"TB":139,"TEX":140,
    "TOR":141,"WSH":120,
}

TREND_LABELS = {
    "launch_speed": "Exit Velocity",
    "launch_angle": "Launch Angle",
    "estimated_woba_using_speedangle": "xwOBA",
    "is_swing": "Swing%",
    "is_whiff": "Whiff%",
    "is_chase": "Chase%",
    "is_z_swing": "Z-Swing%",
}

# =========================================================
# Session state helpers
# =========================================================
def _ss_get(key, default=None):
    return st.session_state.get(key, default)

def _ss_set(key, value):
    st.session_state[key] = value

def memo(key, builder):
    if key in st.session_state:
        return st.session_state[key]
    v = builder()
    st.session_state[key] = v
    return v

def memo_by_params(prefix, params, builder):
    key = f"{prefix}::{hash(params)}"
    return memo(key, builder)

# =========================================================
# Helpers
# =========================================================
def safe_num(x):
    return pd.to_numeric(x, errors="coerce")

def normalize_name(s):
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower().strip()

def valid_pitch_rows(df):
    if df is None or df.empty or "pitch_type" not in df.columns:
        return pd.DataFrame()
    p = df["pitch_type"].astype(str)
    return df.loc[p.notna() & (~p.isin(list(INVALID_PITCH_TYPES)))].copy()

def require_cols(df, cols):
    return all(c in df.columns for c in cols)

def retry_call(fn, tries=3, base_sleep=1.25):
    for i in range(tries):
        try:
            return fn()
        except _REQ_EXC as e:
            if i == tries - 1:
                raise
            time.sleep(base_sleep * (2 ** i))
        except Exception:
            raise

def pitch_group(pt):
    if pt in FASTBALLS: return "Fastballs"
    if pt in OFFSPEED:  return "Offspeed"
    if pt in BREAKING:  return "Breaking"
    return "Other"

def season_window(year):
    return (f"{year}-02-10", f"{year}-12-15")

def allowed_game_types(include_st, include_post):
    s = {"R"}
    if include_st: s.add("S")
    if include_post: s.add("P")
    return s

def filter_game_types(df, allowed):
    if df is None or df.empty or "game_type" not in df.columns:
        return df
    gt = df["game_type"].fillna("").astype(str)
    return df.loc[gt.isin(list(allowed))].copy()

def is_barrel(ev_mph, la_deg):
    if pd.isna(ev_mph) or pd.isna(la_deg) or ev_mph < 98:
        return False
    if ev_mph < 99:   lo, hi = 26, 30
    elif ev_mph < 100: lo, hi = 25, 31
    elif ev_mph < 101: lo, hi = 24, 33
    elif ev_mph < 102: lo, hi = 23, 34
    elif ev_mph < 103: lo, hi = 22, 35
    elif ev_mph < 104: lo, hi = 21, 36
    elif ev_mph < 105: lo, hi = 20, 37
    elif ev_mph < 106: lo, hi = 19, 38
    elif ev_mph < 107: lo, hi = 18, 39
    elif ev_mph < 108: lo, hi = 17, 40
    elif ev_mph < 109: lo, hi = 16, 41
    elif ev_mph < 110: lo, hi = 15, 42
    elif ev_mph < 111: lo, hi = 14, 43
    elif ev_mph < 112: lo, hi = 13, 44
    elif ev_mph < 113: lo, hi = 12, 45
    elif ev_mph < 114: lo, hi = 11, 46
    elif ev_mph < 115: lo, hi = 10, 47
    elif ev_mph < 116: lo, hi = 9, 48
    else:              lo, hi = 8, 50
    return lo <= la_deg <= hi

# =========================================================
# Data loading
# =========================================================
@st.cache_data(ttl=86400, show_spinner=False)
def load_hitter_dropdown():
    reg = chadwick_register().copy()
    reg = reg.dropna(subset=["key_mlbam"]).copy()
    reg["key_mlbam"] = pd.to_numeric(reg["key_mlbam"], errors="coerce").astype("Int64")
    if "key_fangraphs" in reg.columns:
        reg["key_fangraphs"] = pd.to_numeric(reg["key_fangraphs"], errors="coerce").astype("Int64")
    else:
        reg["key_fangraphs"] = pd.Series([pd.NA] * len(reg), dtype="Int64")
    reg = reg.dropna(subset=["key_mlbam"]).copy()
    reg["display"] = (
        reg.get("name_first", "").fillna("").astype(str).str.strip()
        + " "
        + reg.get("name_last", "").fillna("").astype(str).str.strip()
    ).str.strip()
    reg["display_norm"] = reg["display"].map(normalize_name)
    reg = reg[reg["display"].astype(str).str.len() > 0].copy()
    reg = reg.drop_duplicates(subset=["display", "key_mlbam"], keep="first")

    # Filter to 2023+ batters only via FanGraphs batting stats
    try:
        from pybaseball import batting_stats as _bat_stats
        recent_ids = set()
        recent_names = set()
        for yr in [2023, 2024, 2025]:
            try:
                df_fg = _bat_stats(yr, qual=0)
                if df_fg is not None and not df_fg.empty:
                    id_cols = [c for c in ["IDfg", "idfg", "playerid"] if c in df_fg.columns]
                    if id_cols:
                        recent_ids.update(pd.to_numeric(df_fg[id_cols[0]], errors="coerce").dropna().astype(int).tolist())
                    if "Name" in df_fg.columns:
                        recent_names.update(df_fg["Name"].str.lower().str.strip().tolist())
                        recent_names.update(df_fg["Name"].str.lower().str.strip().str.split().str[-1].tolist())
            except Exception:
                pass
        if recent_ids:
            fg_ids = pd.to_numeric(reg["key_fangraphs"], errors="coerce")
            last_names = reg["display"].str.lower().str.strip().str.split().str[-1]
            name_match = reg["display"].str.lower().str.strip().isin(recent_names) | last_names.isin(recent_names)
            reg = reg[fg_ids.isin(recent_ids) | name_match].copy()
    except Exception:
        pass

    reg = reg.sort_values(["display"]).reset_index(drop=True)
    return reg[["display", "display_norm", "key_mlbam", "key_fangraphs"]]

def resolve_name_from_mlbam(hitter_df, mlbam_id):
    hit = hitter_df.loc[hitter_df["key_mlbam"].astype("Int64") == int(mlbam_id)]
    if not hit.empty:
        return str(hit.iloc[0]["display"])
    try:
        r = requests.get(f"https://statsapi.mlb.com/api/v1/people/{mlbam_id}", timeout=5)
        if r.status_code == 200:
            people = r.json().get("people", [])
            if people:
                return people[0].get("fullName", f"MLBAM {mlbam_id}")
    except Exception:
        pass
    return f"MLBAM {mlbam_id}"

def fetch_statcast_batter(mlbam_id, start_date, end_date, allowed_gt):
    today = dt.date.today().isoformat()
    def _build():
        df = retry_call(lambda: statcast_batter(start_date, end_date, mlbam_id), tries=3)
        df = pd.DataFrame(df) if df is not None else pd.DataFrame()
        df = filter_game_types(df, allowed_gt)
        return df
    return memo_by_params("sc_batter_v11", (APP_VERSION, mlbam_id, start_date, end_date, tuple(sorted(list(allowed_gt))), today), _build)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_statcast_league(start_date, end_date, allowed_gt):
    df = retry_call(lambda: statcast(start_date, end_date), tries=3)
    df = pd.DataFrame(df) if df is not None else pd.DataFrame()
    df = filter_game_types(df, set(allowed_gt))
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fg_batting_year(year):
    try:
        import pybaseball as pyb
        # Set user agent to avoid blocks
        df = batting_stats(year, qual=0)
        return pd.DataFrame(df) if df is not None else pd.DataFrame()
    except Exception as e:
        # Try direct FanGraphs URL as fallback
        try:
            import requests, io
            url = f"https://www.fangraphs.com/api/leaders/major-league/data?age=0&pos=all&stats=bat&lg=all&qual=0&season={year}&season1={year}&startdate=&enddate=&month=0&hand=&team=0&pageitems=500&pagenum=1&ind=0&rost=0&players=0&type=8&postseason=&sortdir=default&sortstat=WAR"
            headers = {{"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}}
            r = requests.get(url, headers=headers, timeout=30)
            data = r.json()
            if "data" in data:
                return pd.DataFrame(data["data"])
        except Exception:
            pass
        return pd.DataFrame()

# =========================================================
# Feature engineering
# =========================================================
def add_helpers(df):
    df = df.copy()
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    if "pfx_x" in df.columns and "pfx_z" in df.columns:
        df["HB_in"]  = -safe_num(df["pfx_x"]) * 12.0
        df["iVB_in"] = safe_num(df["pfx_z"]) * 12.0

    if "description" in df.columns:
        df["is_swing"]   = df["description"].isin(SWING_DESCRIPTIONS)
        df["is_whiff"]   = df["description"].isin(WHIFF_DESCRIPTIONS)
        df["is_contact"] = df["description"].isin(CONTACT_DESCRIPTIONS)
    else:
        df["is_swing"] = df["is_whiff"] = df["is_contact"] = False

    if "zone" in df.columns:
        zone_num = safe_num(df["zone"])
        df["in_zone"]    = zone_num.between(1, 9)
        df["in_heart"]   = zone_num.isin([5])  # zone 5 = heart of plate
        df["is_z_swing"] = df["is_swing"] & df["in_zone"].fillna(False)
        df["is_chase"]   = df["is_swing"] & (~df["in_zone"].fillna(True))
        df["is_heart_swing"] = df["is_swing"] & df["in_heart"].fillna(False)
    else:
        df["in_zone"] = df["in_heart"] = df["is_z_swing"] = df["is_chase"] = df["is_heart_swing"] = np.nan

    if "pitch_type" in df.columns:
        df["pitch_type"]  = df["pitch_type"].astype(str)
        df["pitch_group"] = df["pitch_type"].apply(pitch_group)

    return df

# =========================================================
# Styling (red/green shading)
# =========================================================
def _interp_rgb(c0, c1, t):
    t = float(np.clip(t, 0.0, 1.0))
    return (
        int(round(c0[0] + (c1[0]-c0[0])*t)),
        int(round(c0[1] + (c1[1]-c0[1])*t)),
        int(round(c0[2] + (c1[2]-c0[2])*t)),
    )

def style_red_green(df, directions, fmt_map=None, group_col=None, baselines=None,
                    neutral_sd=0.35, clip_sd=2.0, qualify_col=None, qualify_min=50):
    tmp = df.copy()
    sty = tmp.style
    if fmt_map:
        sty = sty.format(fmt_map, na_rep="—")

    green = (64, 160, 92)
    red   = (210, 78, 78)
    white = (255, 255, 255)

    table_stats = {}
    for c in directions:
        if c in tmp.columns:
            x = pd.to_numeric(tmp[c], errors="coerce")
            table_stats[c] = (float(x.mean()) if pd.notna(x.mean()) else np.nan,
                              float(x.std(ddof=0)) if pd.notna(x.std(ddof=0)) else np.nan)

    def pick_mu_sd(row, col):
        if baselines and group_col and group_col in tmp.columns:
            key = str(row.get(group_col, ""))
            d = baselines.get(key) or baselines.get("_ALL_") or {}
            mu, sd = d.get(col, (np.nan, np.nan))
            if pd.notna(mu) and pd.notna(sd) and sd not in (0, 0.0):
                return mu, sd
        return table_stats.get(col, (np.nan, np.nan))

    def style_cell(val, mu, sd, direction):
        v = pd.to_numeric(val, errors="coerce")
        if pd.isna(v) or pd.isna(mu) or pd.isna(sd) or sd == 0:
            return ""
        z = (float(v) - float(mu)) / float(sd)
        if direction == "low_good":
            z = -z
        if abs(z) <= neutral_sd:
            return "background-color: rgb(255,255,255); color: black;"
        z = float(np.clip(z, -clip_sd, clip_sd))
        t = (z + clip_sd) / (2 * clip_sd)
        if t < 0.5:
            r, g, b = _interp_rgb(red, white, t / 0.5)
        else:
            r, g, b = _interp_rgb(white, green, (t - 0.5) / 0.5)
        return f"background-color: rgb({r},{g},{b}); color: black;"

    for c, direction in directions.items():
        if c not in tmp.columns:
            continue
        def apply_col(s, col=c, dirn=direction):
            return [style_cell(v, *pick_mu_sd(tmp.loc[idx], col), dirn) for idx, v in s.items()]
        sty = sty.apply(apply_col, subset=[c])

    if group_col and group_col in tmp.columns:
        name_to_abbrev = {v: k for k, v in PITCH_NAMES.items()}
        def group_chip(s):
            out = []
            for v in s.astype(str):
                abbrev = name_to_abbrev.get(v, v)
                color = PITCH_COLORS.get(abbrev, PITCH_COLORS.get(v, "#9e9e9e"))
                out.append(f"background-color: {color}; color: white; font-weight: 800;")
            return out
        sty = sty.apply(group_chip, subset=[group_col])

    return sty

# =========================================================
# Compute batter plate discipline stats
# =========================================================
def compute_plate_discipline(df, label="Overall"):
    if df is None or df.empty:
        return None
    pitches = len(df)
    if pitches == 0:
        return None

    swings     = int(df["is_swing"].sum()) if "is_swing" in df.columns else 0
    whiffs     = int(df["is_whiff"].sum()) if "is_whiff" in df.columns else 0
    in_zone    = df["in_zone"].fillna(False).astype(bool) if "in_zone" in df.columns else pd.Series([False]*len(df))
    z_pitches  = int(in_zone.sum())
    z_swings   = int(df["is_z_swing"].sum()) if "is_z_swing" in df.columns else 0
    z_whiffs   = int((df["is_whiff"] & in_zone).sum()) if "is_whiff" in df.columns else 0
    z_contacts = int((df["is_contact"] & in_zone).sum()) if "is_contact" in df.columns else 0
    chases     = int(df["is_chase"].sum()) if "is_chase" in df.columns else 0
    out_zone   = (~in_zone)
    oz_pitches = int(out_zone.sum())

    heart      = df["in_heart"].fillna(False).astype(bool) if "in_heart" in df.columns else pd.Series([False]*len(df))
    heart_p    = int(heart.sum())
    heart_sw   = int(df["is_heart_swing"].sum()) if "is_heart_swing" in df.columns else 0
    heart_ct   = int((df["is_contact"] & heart).sum()) if "is_contact" in df.columns else 0

    return {
        "Group": label,
        "Pitches": pitches,
        "Zone%":    round(z_pitches/pitches*100, 1) if pitches else None,
        "Swing%":   round(swings/pitches*100, 1) if pitches else None,
        "Z-Swing%": round(z_swings/z_pitches*100, 1) if z_pitches else None,
        "Z-Contact%": round(z_contacts/z_swings*100, 1) if z_swings else None,
        "Chase%":   round(chases/oz_pitches*100, 1) if oz_pitches else None,
        "Whiff%":   round(whiffs/swings*100, 1) if swings else None,
        "Heart Swing%": round(heart_sw/heart_p*100, 1) if heart_p else None,
        "Heart Contact%": round(heart_ct/heart_sw*100, 1) if heart_sw else None,
    }

# =========================================================
# Compute batted ball stats
# =========================================================
def compute_batted_ball(df):
    if df is None or df.empty:
        return {}
    bbe = df.dropna(subset=["launch_speed", "launch_angle"]).copy() if require_cols(df, ["launch_speed", "launch_angle"]) else pd.DataFrame()
    # BIP only - true batted balls (excludes foul tips etc)
    bip = bbe[bbe["bb_type"].notna()].copy() if "bb_type" in bbe.columns else bbe

    out = {}
    if not bbe.empty:
        ev = safe_num(bbe["launch_speed"])
        la = safe_num(bbe["launch_angle"])
        bip_ev = safe_num(bip["launch_speed"]) if not bip.empty else ev
        bip_la = safe_num(bip["launch_angle"]) if not bip.empty else la
        out["Avg EV"]    = round(float(bip_ev.mean()), 1) if bip_ev.notna().any() else None
        out["90th EV"]   = round(float(bip_ev.quantile(0.9)), 1) if bip_ev.notna().any() else None
        out["Max EV"]    = round(float(bip_ev.max()), 1) if bip_ev.notna().any() else None
        out["Avg LA"]    = round(float(bip_la.mean()), 1) if bip_la.notna().any() else None
        out["HardHit%"]  = round(float((bip_ev >= 95).mean()*100), 1) if bip_ev.notna().any() else None
        bip_la = safe_num(bip["launch_angle"]) if not bip.empty else la
        out["Barrel%"]   = round(float(sum(is_barrel(float(e), float(l)) for e, l in zip(bip_ev, bip_la))/len(bip)*100), 1) if not bip.empty else None
        out["SweetSpot%"] = round(float(((bip_la >= 8) & (bip_la <= 32)).mean()*100), 1) if not bip.empty else None

        # AirPull% - pulled LD/FB / all BIP, 15deg threshold matching Savant
        if "bb_type" in df.columns and "hc_x" in df.columns and "hc_y" in df.columns and "stand" in df.columns:
            ld_fb = df[df["bb_type"].isin(["line_drive","fly_ball"])].dropna(subset=["hc_x","hc_y","stand"]).copy()
            all_bip = df[df["bb_type"].isin(["ground_ball","line_drive","fly_ball","popup"])].dropna(subset=["hc_x","hc_y"])
            if not ld_fb.empty and len(all_bip):
                lx = safe_num(ld_fb["hc_x"])
                ly = safe_num(ld_fb["hc_y"])
                lspray = np.degrees(np.arctan((lx - 125.42) / (198.27 - ly))) * 0.75
                lst = ld_fb["stand"].astype(str)
                pulled = ((lst == "R") & (lspray < -15)) | ((lst == "L") & (lspray > 15))
                out["AirPull%"] = round(float(pulled.sum() / len(all_bip) * 100), 1)
            else:
                out["AirPull%"] = None

    if "bb_type" in df.columns:
        bt = df["bb_type"].fillna("").astype(str)
        bip = bt[bt.isin(["ground_ball","line_drive","fly_ball","popup"])]
        if len(bip):
            out["GB%"] = round(float((bip=="ground_ball").mean()*100), 1)
            out["LD%"] = round(float((bip=="line_drive").mean()*100), 1)
            out["FB%"] = round(float((bip=="fly_ball").mean()*100), 1)
            out["PU%"] = round(float((bip=="popup").mean()*100), 1)

    # Bat speed
    if "bat_speed" in df.columns:
        bs = safe_num(df["bat_speed"]).dropna()
        out["Bat Speed"] = round(float(bs.mean()), 1) if len(bs) else None
    else:
        out["Bat Speed"] = None

    return out

# =========================================================
# Compute production stats
# =========================================================
def compute_production(df):
    if df is None or df.empty:
        return {}

    pa_end_mask = None
    if require_cols(df, ["game_pk", "at_bat_number", "pitch_number"]):
        last_idx = df.groupby(["game_pk","at_bat_number"])["pitch_number"].idxmax()
        pa_end = df.loc[last_idx].copy()
        evs = pa_end["events"].fillna("").astype(str) if "events" in pa_end.columns else pd.Series(dtype=str)
    else:
        evs = pd.Series(dtype=str)

    h   = int(evs.isin(["single","double","triple","home_run"]).sum())
    hr  = int((evs=="home_run").sum())
    so  = int(evs.isin(["strikeout","strikeout_double_play"]).sum())
    bb  = int(evs.isin(["walk","intent_walk"]).sum())
    hbp = int((evs=="hit_by_pitch").sum())
    non_ab = {"walk","intent_walk","hit_by_pitch","sac_fly","sac_bunt","catcher_interf"}
    ab  = int((~evs.isin(list(non_ab)) & evs.ne("")).sum())
    pa  = ab + bb + hbp
    doubles = int((evs=="double").sum())
    triples = int((evs=="triple").sum())
    tb  = h - hr - doubles - triples + 2*doubles + 3*triples + 4*hr

    avg  = h/ab if ab else None
    obp  = (h+bb+hbp)/pa if pa else None
    slg  = tb/ab if ab else None
    ops  = (obp or 0)+(slg or 0) if obp is not None and slg is not None else None

    xwoba_val = None
    if "estimated_woba_using_speedangle" in df.columns:
        xw = safe_num(df["estimated_woba_using_speedangle"]).dropna()
        xwoba_val = float(xw.mean()) if len(xw) else None

    woba_val = None
    if "woba_value" in df.columns and "woba_denom" in df.columns:
        wd = safe_num(df["woba_denom"])
        wv = safe_num(df["woba_value"])
        ok = (wd > 0) & wd.notna()
        if ok.any():
            woba_val = float(wv[ok].sum() / wd[ok].sum())

    bat_speed = None
    if "bat_speed" in df.columns:
        bs = safe_num(df["bat_speed"]).dropna()
        bat_speed = round(float(bs.mean()), 1) if len(bs) else None

    return {
        "PA": pa, "AB": ab, "H": h, "HR": hr, "BB": bb, "SO": so,
        "AVG": round(avg, 3) if avg is not None else None,
        "OBP": round(obp, 3) if obp is not None else None,
        "SLG": round(slg, 3) if slg is not None else None,
        "OPS": round(ops, 3) if ops is not None else None,
        "xwOBA": round(xwoba_val, 3) if xwoba_val is not None else None,
        "wOBA":  round(woba_val, 3) if woba_val is not None else None,
        "Bat Speed": bat_speed,
    }

# =========================================================
# Compute per-pitch-type stats for batter
# =========================================================
def compute_pitch_type_stats(df):
    if df is None or df.empty or "pitch_type" not in df.columns:
        return pd.DataFrame()

    vp = valid_pitch_rows(df)
    if vp.empty:
        return pd.DataFrame()

    total = len(vp)
    rows = []

    for ptype, g in vp.groupby("pitch_type", dropna=True):
        pitches = len(g)
        pct = round(pitches/total*100, 1)

        pd_stats = compute_plate_discipline(g, label=str(ptype))
        bb_stats = compute_batted_ball(g)

        avg_ev  = bb_stats.get("Avg EV")
        hard_hit = bb_stats.get("HardHit%")
        barrel  = bb_stats.get("Barrel%")
        xwoba_v = None
        if "estimated_woba_using_speedangle" in g.columns:
            xw = safe_num(g["estimated_woba_using_speedangle"]).dropna()
            xwoba_v = round(float(xw.mean()), 3) if len(xw) else None

        rows.append({
            "Pitch": PITCH_NAMES.get(str(ptype), str(ptype)),
            "Pitch%": pct,
            "Pitches": pitches,
            "Swing%":   pd_stats.get("Swing%") if pd_stats else None,
            "Whiff%":   pd_stats.get("Whiff%") if pd_stats else None,
            "Chase%":   pd_stats.get("Chase%") if pd_stats else None,
            "Z-Swing%": pd_stats.get("Z-Swing%") if pd_stats else None,
            "Z-Contact%": pd_stats.get("Z-Contact%") if pd_stats else None,
            "Avg EV":   avg_ev,
            "HardHit%": hard_hit,
            "Barrel%":  barrel,
            "xwOBA":    xwoba_v,
            "Heart Swing%": pd_stats.get("Heart Swing%") if pd_stats else None,
            "Heart Contact%": pd_stats.get("Heart Contact%") if pd_stats else None,
        })

    # Add Overall row at bottom
    overall_pd = compute_plate_discipline(vp, "Overall")
    overall_bb = compute_batted_ball(vp)
    overall_xwoba = float(safe_num(vp["estimated_woba_using_speedangle"]).dropna().mean()) if "estimated_woba_using_speedangle" in vp.columns else None
    rows.append({
        "Pitch": "Overall",
        "Pitch%": 100.0,
        "Pitches": len(vp),
        "Swing%": overall_pd.get("Swing%") if overall_pd else None,
        "Whiff%": overall_pd.get("Whiff%") if overall_pd else None,
        "Chase%": overall_pd.get("Chase%") if overall_pd else None,
        "Z-Swing%": overall_pd.get("Z-Swing%") if overall_pd else None,
        "Z-Contact%": overall_pd.get("Z-Contact%") if overall_pd else None,
        "Avg EV": overall_bb.get("Avg EV"),
        "HardHit%": overall_bb.get("HardHit%"),
        "Barrel%": overall_bb.get("Barrel%"),
        "xwOBA": round(overall_xwoba, 3) if overall_xwoba else None,
        "Heart Swing%": overall_pd.get("Heart Swing%") if overall_pd else None,
        "Heart Contact%": overall_pd.get("Heart Contact%") if overall_pd else None,
    })
    out = pd.DataFrame(rows).sort_values("Pitches", ascending=False).reset_index(drop=True)
    return out

# =========================================================
# Compute stats vs pitch group
# =========================================================
def compute_pitch_group_stats(df):
    if df is None or df.empty:
        return pd.DataFrame()

    vp = valid_pitch_rows(df)
    if vp.empty or "pitch_group" not in vp.columns:
        return pd.DataFrame()

    rows = []
    for grp in ["Fastballs", "Offspeed", "Breaking"]:
        g = vp[vp["pitch_group"] == grp]
        if g.empty:
            continue
        pd_stats = compute_plate_discipline(g, label=grp)
        bb_stats = compute_batted_ball(g)
        xwoba_v = None
        if "estimated_woba_using_speedangle" in g.columns:
            xw = safe_num(g["estimated_woba_using_speedangle"]).dropna()
            xwoba_v = round(float(xw.mean()), 3) if len(xw) else None
        rows.append({
            "Group": grp,
            "Pitches": len(g),
            "Swing%":   pd_stats.get("Swing%") if pd_stats else None,
            "Whiff%":   pd_stats.get("Whiff%") if pd_stats else None,
            "Chase%":   pd_stats.get("Chase%") if pd_stats else None,
            "Z-Swing%": pd_stats.get("Z-Swing%") if pd_stats else None,
            "Z-Contact%": pd_stats.get("Z-Contact%") if pd_stats else None,
            "Avg EV":   bb_stats.get("Avg EV"),
            "HardHit%": bb_stats.get("HardHit%"),
            "Barrel%":  bb_stats.get("Barrel%"),
            "xwOBA":    xwoba_v,
            "Heart Swing%": pd_stats.get("Heart Swing%") if pd_stats else None,
            "Heart Contact%": pd_stats.get("Heart Contact%") if pd_stats else None,
        })

    # Add Overall row
    overall_pd = compute_plate_discipline(vp, "Overall")
    overall_bb = compute_batted_ball(vp)
    overall_xwoba = float(safe_num(vp["estimated_woba_using_speedangle"]).dropna().mean()) if "estimated_woba_using_speedangle" in vp.columns else None
    rows.append({
        "Group": "Overall",
        "Pitches": len(vp),
        "Swing%": overall_pd.get("Swing%") if overall_pd else None,
        "Whiff%": overall_pd.get("Whiff%") if overall_pd else None,
        "Chase%": overall_pd.get("Chase%") if overall_pd else None,
        "Z-Swing%": overall_pd.get("Z-Swing%") if overall_pd else None,
        "Z-Contact%": overall_pd.get("Z-Contact%") if overall_pd else None,
        "Avg EV": overall_bb.get("Avg EV"),
        "HardHit%": overall_bb.get("HardHit%"),
        "Barrel%": overall_bb.get("Barrel%"),
        "xwOBA": round(overall_xwoba, 3) if overall_xwoba else None,
        "Heart Swing%": overall_pd.get("Heart Swing%") if overall_pd else None,
        "Heart Contact%": overall_pd.get("Heart Contact%") if overall_pd else None,
    })
    return pd.DataFrame(rows)

# =========================================================
# Season summary from FanGraphs
# =========================================================
def build_season_summary(fg_id, mlbam_id, display_name, current_year, allowed_gt):
    years = [2026, 2025]
    rows = []
    for yr in years:
        fg = fetch_fg_batting_year(yr)
        row = {}
        if not fg.empty:
            name_cols = [c for c in ["Name","name"] if c in fg.columns]
            if name_cols:
                fg["_norm"] = fg[name_cols[0]].map(normalize_name)
            id_cols = [c for c in ["IDfg","idfg","playerid"] if c in fg.columns]
            if id_cols:
                x = pd.to_numeric(fg[id_cols[0]], errors="coerce")
                hit = fg[x == float(fg_id)] if fg_id else pd.DataFrame()
                if hit.empty and name_cols:
                    dn = normalize_name(display_name)
                    hit = fg[fg["_norm"] == dn]
                if not hit.empty:
                    row = hit.iloc[0].to_dict()

        def _n(k): return pd.to_numeric(row.get(k, np.nan), errors="coerce")

        # Also get xwOBA from Statcast
        xwoba_sc = None
        try:
            s, e = season_window(yr)
            sc_y = fetch_statcast_batter(mlbam_id, s, e, allowed_gt={"R"})
            if sc_y is not None and not sc_y.empty and "estimated_woba_using_speedangle" in sc_y.columns:
                xw = safe_num(sc_y["estimated_woba_using_speedangle"]).dropna()
                xwoba_sc = round(float(xw.mean()), 3) if len(xw) else None
        except Exception:
            pass

        ip = _n("PA")
        k_raw = _n("K%")
        bb_raw = _n("BB%")
        k_pct = round(float(k_raw)*100, 1) if pd.notna(k_raw) and float(k_raw) <= 1.0 else (round(float(k_raw),1) if pd.notna(k_raw) else np.nan)
        bb_pct = round(float(bb_raw)*100, 1) if pd.notna(bb_raw) and float(bb_raw) <= 1.0 else (round(float(bb_raw),1) if pd.notna(bb_raw) else np.nan)
        rows.append({
            "Season": yr,
            "PA": int(ip) if pd.notna(ip) else "—",
            "AVG": round(float(_n("AVG")), 3) if pd.notna(_n("AVG")) else np.nan,
            "OBP": round(float(_n("OBP")), 3) if pd.notna(_n("OBP")) else np.nan,
            "SLG": round(float(_n("SLG")), 3) if pd.notna(_n("SLG")) else np.nan,
            "OPS": round(float(_n("OPS")), 3) if pd.notna(_n("OPS")) else np.nan,
            "K%": k_pct,
            "BB%": bb_pct,
            "wOBA": round(float(_n("wOBA")), 3) if pd.notna(_n("wOBA")) else np.nan,
            "xwOBA": xwoba_sc if xwoba_sc else np.nan,
        })

    return pd.DataFrame(rows)

# =========================================================
# League baselines for batters
# =========================================================
def compute_league_baselines(lg_df):
    if lg_df is None or lg_df.empty:
        return {}
    df = add_helpers(lg_df)
    vp = valid_pitch_rows(df)
    if vp.empty:
        return {}

    baselines = {}
    all_stats = {}

    def _compute_airpull(g):
        if "bb_type" not in g.columns or "hc_x" not in g.columns or "hc_y" not in g.columns or "stand" not in g.columns:
            return np.nan
        ld_fb = g[g["bb_type"].isin(["line_drive","fly_ball"])].dropna(subset=["hc_x","hc_y","stand"]).copy()
        all_bip = g[g["bb_type"].isin(["ground_ball","line_drive","fly_ball","popup"])].dropna(subset=["hc_x","hc_y"])
        if ld_fb.empty or len(all_bip) == 0:
            return np.nan
        lx = safe_num(ld_fb["hc_x"]); ly = safe_num(ld_fb["hc_y"])
        lspray = np.degrees(np.arctan((lx-125.42)/(198.27-ly)))*0.75
        lst = ld_fb["stand"].astype(str)
        pulled = ((lst=="R")&(lspray<-15))|((lst=="L")&(lspray>15))
        return round(float(pulled.sum()/len(all_bip)*100),1)

    def rate_block(g):
        pitches = len(g)
        if pitches == 0:
            return {}
        swings   = int(g["is_swing"].sum()) if "is_swing" in g.columns else 0
        whiffs   = int(g["is_whiff"].sum()) if "is_whiff" in g.columns else 0
        in_zone  = g["in_zone"].fillna(False).astype(bool) if "in_zone" in g.columns else pd.Series([False]*len(g))
        z_pitches = int(in_zone.sum())
        z_swings = int(g["is_z_swing"].sum()) if "is_z_swing" in g.columns else 0
        z_contacts = int((g["is_contact"] & in_zone).sum()) if "is_contact" in g.columns else 0
        out_zone  = (~in_zone)
        oz_pitches = int(out_zone.sum())
        chases   = int(g["is_chase"].sum()) if "is_chase" in g.columns else 0

        bbe = g.dropna(subset=["launch_speed","launch_angle"]) if require_cols(g, ["launch_speed","launch_angle"]) else pd.DataFrame()
        avg_ev = float(safe_num(bbe["launch_speed"]).mean()) if not bbe.empty else np.nan
        hh = float((safe_num(bbe["launch_speed"]) >= 95).mean()*100) if not bbe.empty else np.nan
        barrel_pct = float(sum(is_barrel(float(e),float(l)) for e,l in zip(safe_num(bbe["launch_speed"]),safe_num(bbe["launch_angle"])))/len(bbe)*100) if not bbe.empty else np.nan

        xwoba_v = float(safe_num(g["estimated_woba_using_speedangle"]).dropna().mean()) if "estimated_woba_using_speedangle" in g.columns else np.nan

        # Bat tracking stats
        bip_mask = g["bb_type"].notna() if "bb_type" in g.columns else pd.Series([False]*len(g))
        bat_spd = float(safe_num(g["bat_speed"]).dropna().mean()) if "bat_speed" in g.columns else np.nan
        bat_spd_bip = float(safe_num(g.loc[bip_mask, "bat_speed"]).dropna().mean()) if "bat_speed" in g.columns else np.nan
        swing_len = float(safe_num(g["swing_length"]).dropna().mean()) if "swing_length" in g.columns else np.nan
        bs_all = safe_num(g["bat_speed"]).dropna() if "bat_speed" in g.columns else pd.Series(dtype=float)
        fast_swing = float((bs_all >= 75).mean()*100) if len(bs_all) else np.nan
        # Batted ball types
        if not bbe.empty and "bb_type" in bbe.columns:
            bip_bbe = bbe[bbe["bb_type"].notna()]
            gb_pct = float((bip_bbe["bb_type"]=="ground_ball").sum()/len(bip_bbe)*100) if len(bip_bbe) else np.nan
            ld_pct = float((bip_bbe["bb_type"]=="line_drive").sum()/len(bip_bbe)*100) if len(bip_bbe) else np.nan
            fb_pct = float((bip_bbe["bb_type"]=="fly_ball").sum()/len(bip_bbe)*100) if len(bip_bbe) else np.nan
            la_vals = safe_num(bip_bbe["launch_angle"])
            sweet = float(((la_vals >= 8) & (la_vals <= 32)).mean()*100) if len(la_vals) else np.nan
        else:
            gb_pct = ld_pct = fb_pct = sweet = np.nan

        return {
            "Swing%":       (swings/pitches*100, 8.0),
            "Whiff%":       (whiffs/swings*100 if swings else np.nan, 8.0),
            "Chase%":       (chases/oz_pitches*100 if oz_pitches else np.nan, 6.0),
            "Z-Swing%":     (z_swings/z_pitches*100 if z_pitches else np.nan, 7.0),
            "Z-Contact%":   (z_contacts/z_swings*100 if z_swings else np.nan, 7.0),
            "Avg EV":       (avg_ev, 3.0),
            "HardHit%":     (hh, 7.0),
            "Barrel%":      (barrel_pct, 3.0),
            "xwOBA":        (xwoba_v, 0.040),
            "Bat Speed":    (bat_spd, 2.0),
            "Bat Spd (BIP)": (bat_spd_bip, 2.0),
            "Swing Length": (swing_len, 0.3),
            "Fast Swing%":  (fast_swing, 5.0),
            "SweetSpot%":   (sweet, 5.0),
            "GB%":          (gb_pct, 5.0),
            "LD%":          (ld_pct, 4.0),
            "FB%":          (fb_pct, 4.0),
            "90th EV":      (np.nan, 3.0),
            "Max EV":       (np.nan, 3.0),
            "Avg LA":       (np.nan, 5.0),
            "AirPull%":     (_compute_airpull(g), 5.0),
        }

    all_rates = rate_block(vp)
    baselines["_ALL_"] = all_rates

    for ptype, g in vp.groupby(vp["pitch_type"].astype(str), dropna=True):
        if len(g) < MIN_BASELINE_PITCHES:
            continue
        baselines[PITCH_NAMES.get(str(ptype), str(ptype))] = rate_block(g)

    for grp in ["Fastballs", "Offspeed", "Breaking"]:
        g = vp[vp["pitch_group"] == grp] if "pitch_group" in vp.columns else pd.DataFrame()
        if len(g) >= MIN_BASELINE_PITCHES:
            baselines[grp] = rate_block(g)

    return baselines

# =========================================================
# Heatmap
# =========================================================
def add_strikezone(ax):
    rect = Rectangle((STRIKEZONE["x0"], STRIKEZONE["z0"]), STRIKEZONE["w"], STRIKEZONE["h"],
                     fill=False, linewidth=2)
    ax.add_patch(rect)

def add_pitcher_illustration(ax, batter_hand):
    # Show pitcher on mound side (opposite of batter)
    pass

def plot_heatmap(sc, pitch_group_filter, metric, title_suffix=""):
    needed = ["plate_x","plate_z","pitch_group"]
    if not require_cols(sc, needed):
        st.info("Heatmap needs plate_x, plate_z, pitch_group.")
        return

    df = sc.dropna(subset=["plate_x","plate_z","pitch_group"]).copy()
    if pitch_group_filter != "All":
        df = df[df["pitch_group"] == pitch_group_filter]
    df = valid_pitch_rows(df)

    fig, ax = plt.subplots(figsize=(3.8, 3.8))

    if df.empty:
        add_strikezone(ax)
        ax.set_title(f"{pitch_group_filter} — {metric}", fontsize=10)
        ax.set_xlabel("Catcher POV"); ax.set_ylabel("Plate Height")
        ax.set_xlim(-2.5, 2.5); ax.set_ylim(-0.5, 5.0)
        ax.text(0, 2.0, "No data", ha="center", va="center", fontsize=11, color="#aaaaaa")
        st.pyplot(fig, clear_figure=True)
        return

    weights = None
    if metric == "Exit Velo" and "launch_speed" in df.columns:
        df = df.dropna(subset=["launch_speed"]).copy()
        weights = safe_num(df["launch_speed"]).fillna(0)

    if len(df) >= 20 and df["plate_x"].nunique() >= 3 and df["plate_z"].nunique() >= 3:
        try:
            sns.kdeplot(data=df, x="plate_x", y="plate_z", fill=True, levels=9,
                       thresh=0.30, weights=weights, cmap="RdBu_r", ax=ax)
        except Exception:
            ax.scatter(df["plate_x"], df["plate_z"], s=14, alpha=0.35)
    else:
        ax.scatter(df["plate_x"], df["plate_z"], s=14, alpha=0.35)

    add_strikezone(ax)
    ax.set_title(f"{pitch_group_filter} — {metric}{title_suffix}", fontsize=10)
    ax.set_xlabel("Catcher POV"); ax.set_ylabel("Plate Height")
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-0.5, 5.0)
    st.pyplot(fig, clear_figure=True)

# =========================================================
# Trends
# =========================================================
def trend_by_game(sc, variables):
    if sc is None or sc.empty or "game_pk" not in sc.columns or "game_date" not in sc.columns:
        return pd.DataFrame()
    df = sc.dropna(subset=["game_pk","game_date"]).copy()
    if df.empty:
        return pd.DataFrame()
    cols = ["game_pk","game_date"]
    out = df[cols].copy()
    for v in variables:
        out[v] = safe_num(df[v]) if v in df.columns else np.nan
    g = out.groupby(["game_pk","game_date"], as_index=False).mean(numeric_only=True)
    return g.sort_values(["game_date","game_pk"]).reset_index(drop=True)

def plot_trends(tr, variable, label):
    if tr is None or tr.empty or variable not in tr.columns:
        st.info("No trend data available.")
        return
    y = safe_num(tr[variable])
    y_min = float(y.dropna().min()) if y.notna().any() else 0
    y_max = float(y.dropna().max()) if y.notna().any() else 1
    span  = max(y_max - y_min, 1e-6)
    pad   = max(span * 2.0, 1.0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tr["game_date"], y=y,
        mode="lines+markers",
        name=label,
        line=dict(color="#1f77b4"),
        hovertext="Date: " + pd.to_datetime(tr["game_date"]).dt.strftime("%Y-%m-%d").astype(str) + f"<br>{label}: " + y.round(3).astype(str),
        hoverinfo="text",
    ))
    fig.update_layout(
        height=380,
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis=dict(title="Date", tickformat="%b %d", showgrid=True, tickangle=-25),
        yaxis=dict(title=label, showgrid=True, range=[y_min-pad, y_max+pad]),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# Main app
# =========================================================
def main():
    st.title("HITTER DASHBOARD")

    hitter_df = load_hitter_dropdown()

    with st.sidebar:
        st.header("Controls")

        search = st.text_input("Search hitter", value="")
        filtered = hitter_df
        if search.strip():
            s = normalize_name(search)
            filtered = hitter_df[hitter_df["display_norm"].str.contains(s, na=False)].copy()
            if filtered.empty:
                try:
                    import requests as _req
                    r = _req.get("https://statsapi.mlb.com/api/v1/sports/1/players?season=2026", timeout=10)
                    mlb_players = r.json().get("people", [])
                    mlb_rows = []
                    for p in mlb_players:
                        pid = p.get("id")
                        name = p.get("fullName", "")
                        if pid and name and s in normalize_name(name):
                            mlb_rows.append({"key_mlbam": pid, "key_fangraphs": None,
                                            "display": name, "display_norm": normalize_name(name)})
                    filtered = pd.DataFrame(mlb_rows) if mlb_rows else hitter_df
                except Exception:
                    filtered = hitter_df

        manual_mlbam = st.text_input("Manual MLBAM ID (optional)", value="")
        manual_id    = pd.to_numeric(manual_mlbam, errors="coerce")
        use_manual   = pd.notna(manual_id)

        manual_fg    = st.text_input("Manual FanGraphs ID (optional)", value="")
        manual_fg_id = pd.to_numeric(manual_fg, errors="coerce")
        use_manual_fg = pd.notna(manual_fg_id)

        today = dt.date.today()
        default_year = today.year

        season_year = st.selectbox("Season year", options=[2026], index=0)

        if use_manual:
            mlbam_id    = int(manual_id)
            display_name = resolve_name_from_mlbam(hitter_df, mlbam_id)
            fg_id = None
        else:
            selected_display = st.selectbox("Hitter", options=filtered["display"].tolist(), index=0)
            row = filtered.loc[filtered["display"] == selected_display].iloc[0]
            mlbam_id    = int(row["key_mlbam"])
            fg_id       = int(row["key_fangraphs"]) if pd.notna(row.get("key_fangraphs")) else None
            if use_manual_fg:
                fg_id = int(manual_fg_id)
            display_name = selected_display

        st.divider()
        include_st   = st.checkbox("Include Spring Training", value=False)
        include_post = st.checkbox("Include Postseason",      value=False)
        allowed_gt   = allowed_game_types(include_st=include_st, include_post=include_post)

        # Auto-set dates to first game of season
        auto_key = f"auto_dates_h::{APP_VERSION}::{mlbam_id}::{season_year}::{include_st}::{include_post}"
        if _ss_get("auto_key_h") != auto_key:
            # Try to find first game of season for this hitter
            try:
                _test = fetch_statcast_batter(mlbam_id, f"{season_year}-03-25", f"{season_year}-04-30", frozenset({"R"}))
                if _test is not None and not _test.empty and "game_date" in _test.columns:
                    _first = pd.to_datetime(_test["game_date"], errors="coerce").dropna().min().date()
                    _ss_set("start_date_h", _first)
                else:
                    _ss_set("start_date_h", dt.date(season_year, 3, 25))
            except Exception:
                _ss_set("start_date_h", dt.date(season_year, 3, 25))
            _ss_set("end_date_h", today)
            _ss_set("auto_key_h", auto_key)

        start_date = st.date_input("Start date", key="start_date_h")
        end_date   = st.date_input("End date",   key="end_date_h")

        st.divider()
        heat_metric = st.selectbox("Heatmap metric", ["Frequency","Exit Velo"])

        st.divider()
        league_compare = st.checkbox("Compare to league", value=True)
        baseline_days = 30  # kept for compat

        st.divider()
        run_btn = st.button("Run / Refresh Data", type="primary")

    if start_date > end_date:
        st.error("Start date must be on or before end date.")
        return

    start_str = start_date.strftime("%Y-%m-%d")
    end_str   = end_date.strftime("%Y-%m-%d")

    params = (APP_VERSION, mlbam_id, fg_id, display_name, start_str, end_str,
              heat_metric, league_compare, baseline_days, include_st, include_post)

    if run_btn or _ss_get("loaded_params_h") != params:
        with st.spinner("Fetching Statcast batter data..."):
            try:
                sc_raw = fetch_statcast_batter(mlbam_id, start_str, end_str, allowed_gt=allowed_gt)
            except _REQ_EXC:
                st.error("Statcast batter pull timed out.")
                return
            except Exception as e:
                st.error(f"Statcast batter pull failed: {e}")
                return

        if sc_raw.empty:
            _ss_set("sc_h", pd.DataFrame())
            _ss_set("baselines_h", {})
            _ss_set("loaded_params_h", params)
            st.warning("No Statcast rows returned. Try a wider date range.")
            return

        sc = add_helpers(sc_raw)
        _ss_set("sc_h", sc)

        baselines = {}
        if league_compare:
            base_start_str = f"{end_date.year}-03-25"
            base_end_str = end_date.strftime("%Y-%m-%d")
            with st.spinner("Fetching league data (season baseline)..."):
                try:
                    lg = fetch_statcast_league(base_start_str, base_end_str, frozenset(allowed_gt))
                    if lg is not None and not lg.empty:
                        baselines = compute_league_baselines(lg)
                except Exception:
                    st.warning("League pull failed. Shading unavailable.")

        _ss_set("baselines_h", baselines)
        _ss_set("loaded_params_h", params)

    sc        = _ss_get("sc_h", pd.DataFrame())
    baselines = _ss_get("baselines_h", {})

    if sc is None or sc.empty:
        st.info("Open the sidebar and click **Run / Refresh Data**.")
        return

    # ── Header ───────────────────────────────────────────────────────────────
    hitter_team = None
    if not sc.empty and "home_team" in sc.columns and "away_team" in sc.columns and "inning_topbot" in sc.columns:
        tb = sc["inning_topbot"].fillna("").astype(str).mode()
        tb = tb.iloc[0] if not tb.empty else ""
        if tb.lower().startswith("bot"):
            hitter_team = sc["home_team"].dropna().astype(str).mode().iloc[0] if not sc["home_team"].dropna().empty else None
        else:
            hitter_team = sc["away_team"].dropna().astype(str).mode().iloc[0] if not sc["away_team"].dropna().empty else None

    headshot_url = f"https://midfield.mlbstatic.com/v1/people/{mlbam_id}/spots/spot"
    team_id  = TEAM_IDS.get(hitter_team) if hitter_team else None
    logo_url = f"https://www.mlbstatic.com/team-logos/{team_id}.svg" if team_id else None

    prod  = compute_production(sc)
    bb    = compute_batted_ball(sc)

    top_left, top_right = st.columns([2.2, 1])

    with top_left:
        name_cols = st.columns([0.13, 0.6, 0.13, 1.0])
        with name_cols[0]:
            st.image(headshot_url, width=60)
        with name_cols[1]:
            st.subheader(display_name.upper())
        if logo_url:
            with name_cols[2]:
                st.image(logo_url, width=45)

        st.caption(f"{start_str} → {end_str}   ·   game_types={','.join(sorted(list(allowed_gt)))}   ·   app={APP_VERSION}")

        # Season summary
        st.markdown("### SEASON SUMMARY")
        with st.spinner("Building season summary..."):
            ssdf = build_season_summary(fg_id, mlbam_id, display_name, int(end_date.year), allowed_gt)
        if ssdf.empty:
            st.info("No season summary available.")
        else:
            def _safe_fmt(fmt_str):
                def _f(x):
                    try: return fmt_str.format(float(x))
                    except: return str(x) if x not in [None, ""] else "—"
                return _f
            fmt = {"PA":_safe_fmt("{:.0f}"),"AVG":_safe_fmt("{:.3f}"),"OBP":_safe_fmt("{:.3f}"),
                   "SLG":_safe_fmt("{:.3f}"),"OPS":_safe_fmt("{:.3f}"),"K%":_safe_fmt("{:.1f}"),
                   "BB%":_safe_fmt("{:.1f}"),"wOBA":_safe_fmt("{:.3f}"),"xwOBA":_safe_fmt("{:.3f}")}
            fmt = {k: v for k, v in fmt.items() if k in ssdf.columns}
            st.dataframe(ssdf.style.format(fmt, na_rep="—"), use_container_width=True, hide_index=True)

        st.caption("FanGraphs + Statcast (xwOBA).")

        # Platoon splits
        st.markdown("#### Platoon Splits")
        def compute_platoon_splits_h(sc, hand):
            df = sc[sc["stand"] == hand].copy() if "stand" in sc.columns else pd.DataFrame()
            if df.empty:
                return None
            pa_end = df.groupby(["game_pk","at_bat_number"])["pitch_number"].idxmax() if require_cols(df,["game_pk","at_bat_number","pitch_number"]) else pd.Index([])
            pa_df = df.loc[pa_end] if len(pa_end) else pd.DataFrame()
            evs = pa_df["events"].fillna("").astype(str) if not pa_df.empty and "events" in pa_df.columns else pd.Series(dtype=str)
            h = int(evs.isin(["single","double","triple","home_run"]).sum())
            hr = int((evs=="home_run").sum())
            so = int(evs.isin(["strikeout","strikeout_double_play"]).sum())
            bb = int(evs.isin(["walk","intent_walk"]).sum())
            hbp = int((evs=="hit_by_pitch").sum())
            non_ab = {"walk","intent_walk","hit_by_pitch","sac_fly","sac_bunt","catcher_interf"}
            ab = int((~evs.isin(list(non_ab)) & evs.ne("")).sum())
            pa = ab + bb + hbp
            doubles = int((evs=="double").sum())
            triples = int((evs=="triple").sum())
            tb = h - hr - doubles - triples + 2*doubles + 3*triples + 4*hr
            avg = h/ab if ab else None
            obp = (h+bb+hbp)/pa if pa else None
            slg = tb/ab if ab else None
            ops = (obp or 0)+(slg or 0) if obp is not None and slg is not None else None
            k_pct = so/pa*100 if pa else None
            bb_pct = bb/pa*100 if pa else None
            ev = float(safe_num(df["launch_speed"]).dropna().mean()) if "launch_speed" in df.columns else None
            xwoba = float(safe_num(df["estimated_woba_using_speedangle"]).dropna().mean()) if "estimated_woba_using_speedangle" in df.columns else None
            return {"Split": f"vs {hand}HP", "PA": pa, "K%": round(k_pct,1) if k_pct else np.nan,
                    "BB%": round(bb_pct,1) if bb_pct else np.nan, "AVG": round(avg,3) if avg else np.nan,
                    "OBP": round(obp,3) if obp else np.nan, "SLG": round(slg,3) if slg else np.nan,
                    "OPS": round(ops,3) if ops else np.nan, "EV": round(ev,1) if ev else np.nan,
                    "xwOBA": round(xwoba,3) if xwoba else np.nan}
        # Split by pitcher hand (p_throws), not batter hand
        def compute_platoon_splits_p(sc, p_hand):
            df = sc[sc["p_throws"] == p_hand].copy() if "p_throws" in sc.columns else pd.DataFrame()
            if df.empty:
                return None
            pa_end = df.groupby(["game_pk","at_bat_number"])["pitch_number"].idxmax() if require_cols(df,["game_pk","at_bat_number","pitch_number"]) else pd.Index([])
            pa_df = df.loc[pa_end] if len(pa_end) else pd.DataFrame()
            evs = pa_df["events"].fillna("").astype(str) if not pa_df.empty and "events" in pa_df.columns else pd.Series(dtype=str)
            h = int(evs.isin(["single","double","triple","home_run"]).sum())
            hr = int((evs=="home_run").sum())
            so = int(evs.isin(["strikeout","strikeout_double_play"]).sum())
            bb = int(evs.isin(["walk","intent_walk"]).sum())
            hbp = int((evs=="hit_by_pitch").sum())
            non_ab = {"walk","intent_walk","hit_by_pitch","sac_fly","sac_bunt","catcher_interf"}
            ab = int((~evs.isin(list(non_ab)) & evs.ne("")).sum())
            pa = ab + bb + hbp
            doubles = int((evs=="double").sum())
            triples = int((evs=="triple").sum())
            tb = h - hr - doubles - triples + 2*doubles + 3*triples + 4*hr
            avg = h/ab if ab else None
            obp = (h+bb+hbp)/pa if pa else None
            slg = tb/ab if ab else None
            ops = (obp or 0)+(slg or 0) if obp is not None and slg is not None else None
            k_pct = so/pa*100 if pa else None
            bb_pct = bb/pa*100 if pa else None
            ev = float(safe_num(df["launch_speed"]).dropna().mean()) if "launch_speed" in df.columns else None
            xwoba = float(safe_num(df["estimated_woba_using_speedangle"]).dropna().mean()) if "estimated_woba_using_speedangle" in df.columns else None
            label = "vs LHP" if p_hand == "L" else "vs RHP"
            return {"Split": label, "PA": pa, "K%": round(k_pct,1) if k_pct else np.nan,
                    "BB%": round(bb_pct,1) if bb_pct else np.nan, "AVG": round(avg,3) if avg else np.nan,
                    "OBP": round(obp,3) if obp else np.nan, "SLG": round(slg,3) if slg else np.nan,
                    "OPS": round(ops,3) if ops else np.nan, "EV": round(ev,1) if ev else np.nan,
                    "xwOBA": round(xwoba,3) if xwoba else np.nan}
        sl = compute_platoon_splits_p(sc, "L")
        sr = compute_platoon_splits_p(sc, "R")
        sp_rows = [s for s in [sl, sr] if s is not None]
        if sp_rows:
            sp_df = pd.DataFrame(sp_rows)
            sp_fmt = {"PA":"{:.0f}","K%":"{:.1f}","BB%":"{:.1f}","AVG":"{:.3f}","OBP":"{:.3f}","SLG":"{:.3f}","OPS":"{:.3f}","EV":"{:.1f}","xwOBA":"{:.3f}"}
            st.dataframe(sp_df.style.format(sp_fmt, na_rep="—"), use_container_width=True, hide_index=True)

    with top_right:
        st.markdown("### QUICK TOTALS")
        c1, c2 = st.columns(2)
        c1.metric("Pitches Seen", f"{len(valid_pitch_rows(sc)):,}")
        c2.metric("Games", f"{sc['game_date'].nunique():,}" if "game_date" in sc.columns else "—")

    st.divider()

    # ── Switch hitter toggle ──────────────────────────────────────────────────
    batter_hand = None
    if "stand" in sc.columns:
        hands = sc["stand"].dropna().unique().tolist()
        if len(hands) > 1:
            batter_hand = st.radio("Batter side", ["All", "vs LHP", "vs RHP"], horizontal=True, key="batter_hand_toggle")

    def filter_by_hand(df):
        if batter_hand is None or batter_hand == "All" or "p_throws" not in df.columns:
            return df
        pt = "L" if batter_hand == "vs LHP" else "R"
        return df[df["p_throws"] == pt].copy()

    sc_hand = filter_by_hand(sc)

    # ── Bat Tracking ─────────────────────────────────────────────────────────
    st.markdown("## BAT TRACKING + BATTED BALL")

    # Compute bat tracking stats
    def compute_bat_tracking(df):
        out = {}
        if "bat_speed" in df.columns:
            bs = safe_num(df["bat_speed"]).dropna()
            out["Bat Speed"] = round(float(bs.mean()), 1) if len(bs) else None
            # Bat speed on BIP only
            if "bb_type" in df.columns:
                bip_mask = df["bb_type"].notna()
                bs_bip = safe_num(df.loc[bip_mask, "bat_speed"]).dropna() if bip_mask.any() else pd.Series(dtype=float)
                out["Bat Spd (BIP)"] = round(float(bs_bip.mean()), 1) if len(bs_bip) else None
            # Fast swing rate: % of swings >= 75 mph
            swing_mask = df["is_swing"] if "is_swing" in df.columns else pd.Series([False]*len(df))
            bs_swings = safe_num(df.loc[swing_mask, "bat_speed"]).dropna() if swing_mask.any() else pd.Series(dtype=float)
            out["Fast Swing%"] = round(float((bs_swings >= 75).mean() * 100), 1) if len(bs_swings) else None
        if "swing_length" in df.columns:
            sl = safe_num(df["swing_length"]).dropna()
            out["Swing Length"] = round(float(sl.mean()), 1) if len(sl) else None
        return out

    bt = compute_bat_tracking(sc_hand)
    bb_hand = compute_batted_ball(sc_hand)

    # Build single Savant-style table
    bat_row = {
        "Bat Speed": bt.get("Bat Speed"),
        "Bat Spd (BIP)": bt.get("Bat Spd (BIP)"),
        "Swing Length": bt.get("Swing Length"),
        "Fast Swing%": bt.get("Fast Swing%"),
        "Avg EV": bb_hand.get("Avg EV"),
        "90th EV": bb_hand.get("90th EV"),
        "Max EV": bb_hand.get("Max EV"),
        "Avg LA": bb_hand.get("Avg LA"),
        "HardHit%": bb_hand.get("HardHit%"),
        "Barrel%": bb_hand.get("Barrel%"),
        "SweetSpot%": bb_hand.get("SweetSpot%"),
        "AirPull%": bb_hand.get("AirPull%"),
        "GB%": bb_hand.get("GB%"),
        "LD%": bb_hand.get("LD%"),
        "FB%": bb_hand.get("FB%"),
    }
    bat_df = pd.DataFrame([bat_row])
    bat_fmt = {k: "{:.1f}" for k in bat_row if k not in ["Swing Length"]}
    bat_fmt["Swing Length"] = "{:.1f}"
    bat_directions = {
        "Bat Speed": "high_good", "Bat Spd (BIP)": "high_good",
        "Fast Swing%": "high_good", "Swing Length": "low_good",
        "Avg EV": "high_good", "90th EV": "high_good", "Max EV": "high_good",
        "Avg LA": "high_good",
        "HardHit%": "high_good", "Barrel%": "high_good", "SweetSpot%": "high_good",
        "AirPull%": "high_good", "GB%": "low_good", "LD%": "high_good", "FB%": "high_good",
    }
    if league_compare and baselines and "_ALL_" in baselines:
        all_bl = baselines["_ALL_"]
        sty = bat_df.style.format(bat_fmt, na_rep="—")
        def _irp(a, b, t): return tuple(int(a[i]+t*(b[i]-a[i])) for i in range(3))
        green = (64,160,92); red = (210,78,78); white = (255,255,255)
        for col, dirn in bat_directions.items():
            if col not in bat_df.columns: continue
            mu_sd = all_bl.get(col, (None, None))
            mu, sd = mu_sd if isinstance(mu_sd, tuple) else (None, None)
            if mu is None or sd is None or pd.isna(mu) or pd.isna(sd) or sd == 0: continue
            def _sf(val, mu=float(mu), sd=float(sd), dirn=dirn):
                v = pd.to_numeric(val, errors="coerce")
                if pd.isna(v): return ""
                z = (float(v)-mu)/sd
                if dirn == "low_good": z = -z
                if abs(z) <= 0.35: return "background-color:rgb(255,255,255);color:black;"
                z = float(np.clip(z,-2,2))
                t = (z+2)/4
                rgb = _irp(red,white,t/0.5) if t < 0.5 else _irp(white,green,(t-0.5)/0.5)
                return f"background-color:rgb({rgb[0]},{rgb[1]},{rgb[2]});color:black;"
            sty = sty.map(_sf, subset=[col])
        st.dataframe(sty, use_container_width=True, hide_index=True)
    else:
        st.dataframe(bat_df.style.format(bat_fmt, na_rep="—"), use_container_width=True, hide_index=True)

    # Hard fastball splits
    vp = valid_pitch_rows(sc_hand)
    if "pitch_group" in vp.columns and "release_speed" in vp.columns:
        hard_fb = vp[(vp["pitch_group"]=="Fastballs") & (safe_num(vp["release_speed"]) >= 95)]
        high_ivb_fb = vp[(vp["pitch_group"]=="Fastballs") & (safe_num(vp.get("iVB_in", pd.Series(dtype=float))) >= 17)] if "iVB_in" in vp.columns else pd.DataFrame()
        if not hard_fb.empty or not high_ivb_fb.empty:
            st.markdown("#### vs Hard Fastballs")
            hf_rows = []
            if not hard_fb.empty:
                hf_bb = compute_batted_ball(hard_fb)
                hf_pd = compute_plate_discipline(hard_fb, "95+ FB")
                xwoba_hf = float(safe_num(hard_fb["estimated_woba_using_speedangle"]).dropna().mean()) if "estimated_woba_using_speedangle" in hard_fb.columns else None
                hf_rows.append({"Split": "vs 95+ MPH FB", "Pitches": len(hard_fb),
                    "Whiff%": round(hf_pd.get("Whiff%",0),1) if hf_pd and hf_pd.get("Whiff%") else None,
                    "Avg EV": hf_bb.get("Avg EV"), "HardHit%": hf_bb.get("HardHit%"),
                    "Barrel%": hf_bb.get("Barrel%"), "xwOBA": round(xwoba_hf,3) if xwoba_hf else None})
            if not high_ivb_fb.empty:
                hi_bb = compute_batted_ball(high_ivb_fb)
                hi_pd = compute_plate_discipline(high_ivb_fb, "17+ iVB FB")
                xwoba_hi = float(safe_num(high_ivb_fb["estimated_woba_using_speedangle"]).dropna().mean()) if "estimated_woba_using_speedangle" in high_ivb_fb.columns else None
                hf_rows.append({"Split": "vs 17+ iVB FB", "Pitches": len(high_ivb_fb),
                    "Whiff%": round(hi_pd.get("Whiff%",0),1) if hi_pd and hi_pd.get("Whiff%") else None,
                    "Avg EV": hi_bb.get("Avg EV"), "HardHit%": hi_bb.get("HardHit%"),
                    "Barrel%": hi_bb.get("Barrel%"), "xwOBA": round(xwoba_hi,3) if xwoba_hi else None})
            if hf_rows:
                hf_df = pd.DataFrame(hf_rows)
                hf_fmt = {"Whiff%":"{:.1f}","Avg EV":"{:.1f}","HardHit%":"{:.1f}","Barrel%":"{:.1f}","xwOBA":"{:.3f}"}
                st.dataframe(hf_df.style.format(hf_fmt, na_rep="—"), use_container_width=True, hide_index=True)

    st.divider()

    # ── Stats by Pitch Type + Group ───────────────────────────────────────────
    pt_stats = compute_pitch_type_stats(sc_hand)
    pg_stats = compute_pitch_group_stats(sc_hand)

    pt_fmt = {
        "Pitch%": "{:.1f}", "Pitches": "{:.0f}",
        "Swing%": "{:.1f}", "Whiff%": "{:.1f}", "Chase%": "{:.1f}",
        "Z-Swing%": "{:.1f}", "Z-Contact%": "{:.1f}",
        "Avg EV": "{:.1f}", "HardHit%": "{:.1f}", "Barrel%": "{:.1f}",
        "xwOBA": "{:.3f}",
        "Heart Swing%": "{:.1f}", "Heart Contact%": "{:.1f}",
    }
    pt_directions = {
        "Swing%": "high_good", "Whiff%": "low_good", "Chase%": "low_good",
        "Z-Swing%": "high_good", "Z-Contact%": "high_good",
        "Avg EV": "high_good", "HardHit%": "high_good", "Barrel%": "high_good",
        "xwOBA": "high_good",
        "Heart Swing%": "high_good", "Heart Contact%": "high_good",
    }

    # Also add plate discipline (with heart stats) by group
    pd_overall = compute_plate_discipline(sc_hand, "Overall")
    pd_fb = compute_plate_discipline(valid_pitch_rows(sc_hand)[valid_pitch_rows(sc_hand)["pitch_group"]=="Fastballs"] if "pitch_group" in valid_pitch_rows(sc_hand).columns else pd.DataFrame(), "Fastballs")
    pd_br = compute_plate_discipline(valid_pitch_rows(sc_hand)[valid_pitch_rows(sc_hand)["pitch_group"]=="Breaking"] if "pitch_group" in valid_pitch_rows(sc_hand).columns else pd.DataFrame(), "Breaking")
    pd_os = compute_plate_discipline(valid_pitch_rows(sc_hand)[valid_pitch_rows(sc_hand)["pitch_group"]=="Offspeed"] if "pitch_group" in valid_pitch_rows(sc_hand).columns else pd.DataFrame(), "Offspeed")
    pd_rows = [r for r in [pd_fb, pd_br, pd_os, pd_overall] if r is not None]
    pd_directions = {
        "Zone%": "high_good", "Swing%": "low_good", "Z-Swing%": "high_good",
        "Z-Contact%": "high_good", "Chase%": "low_good", "Whiff%": "low_good",
        "Heart Swing%": "high_good", "Heart Contact%": "high_good",
    }

    st.markdown("## STATS BY PITCH TYPE")
    st.markdown("### By Pitch Type")
    if not pt_stats.empty:
        st.dataframe(pt_stats.style.format(pt_fmt, na_rep="—"), use_container_width=True, hide_index=True)

    st.markdown("### By Pitch Group")
    if not pg_stats.empty:
        pg_fmt = {k: v for k, v in pt_fmt.items() if k != "Pitch%"}
        if league_compare and baselines:
            st.dataframe(style_red_green(pg_stats, pt_directions, fmt_map=pg_fmt, group_col="Group", baselines=baselines),
                         use_container_width=True, hide_index=True)
        else:
            st.dataframe(pg_stats.style.format(pg_fmt, na_rep="—"), use_container_width=True, hide_index=True)

    st.caption("Shading = z-score vs league baseline. For a batter: green = better performance for the hitter.")
    st.divider()

    # ── Heatmaps ──────────────────────────────────────────────────────────────
    st.markdown("## HEATMAPS")
    st.caption("From the catcher's POV. Red = more frequent / harder contact.")

    hm_groups = ["All", "Fastballs", "Offspeed", "Breaking"]
    hm_cols = st.columns(len(hm_groups))
    for i, grp in enumerate(hm_groups):
        with hm_cols[i]:
            plot_heatmap(sc, grp, heat_metric)

    st.divider()

    # ── Trends ────────────────────────────────────────────────────────────────
    st.markdown("## TRENDS")

    all_trend_options = list(TREND_LABELS.keys())
    all_trend_labels  = dict(TREND_LABELS)

    tc1, tc2 = st.columns(2)
    with tc1:
        trend_var = st.selectbox("Metric", all_trend_options,
                                 format_func=lambda k: all_trend_labels.get(k, k),
                                 key="trend_var_h")
    with tc2:
        trend_pitch_filter = st.selectbox(
            "Pitch filter",
            options=["(All)"] + sorted(valid_pitch_rows(sc)["pitch_type"].dropna().astype(str).unique().tolist()) if "pitch_type" in sc.columns else ["(All)"],
            key="trend_pitch_h"
        )

    tr_df = sc.copy()
    if trend_pitch_filter != "(All)" and "pitch_type" in tr_df.columns:
        tr_df = tr_df[tr_df["pitch_type"] == trend_pitch_filter]

    tr = trend_by_game(tr_df, [trend_var])
    if tr.empty:
        st.info("No trend data for this selection.")
    else:
        plot_trends(tr, trend_var, all_trend_labels.get(trend_var, trend_var))

if __name__ == "__main__":
    main()
