import sys
import os
import asyncio
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import subprocess
import json
import math
import random
from datetime import datetime

st.set_page_config(
    page_title="Seismic Intelligence System",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Pastel Palette from provided image ───────────────────────────────────────
# E2F1EB  mint green
# C8EBF8  sky blue
# D9D4F4  lavender
# F6B5B5  blush rose
# FDDCB8  peach
# F7FBCA  lemon cream

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

    :root {
        --mint:    #E2F1EB;
        --sky:     #C8EBF8;
        --lav:     #D9D4F4;
        --rose:    #F6B5B5;
        --peach:   #FDDCB8;
        --lemon:   #F7FBCA;
        --ink:     #2C2C3A;
        --slate:   #4A4A6A;
        --muted:   #7A7A9A;
        --white:   #FFFFFF;
        --paper:   #FAFAF8;
    }

    * { font-family: 'DM Sans', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, var(--mint) 0%, var(--sky) 30%, var(--lav) 60%, var(--lemon) 100%);
        background-attachment: fixed;
        color: var(--ink);
    }

    /* Noise texture overlay */
    .stApp::before {
        content: '';
        position: fixed;
        inset: 0;
        background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
        pointer-events: none;
        z-index: 0;
        opacity: 0.4;
    }

    /* ── Sidebar ─────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.72) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(217,212,244,0.5);
        box-shadow: 4px 0 24px rgba(44,44,58,0.06);
    }
    [data-testid="stSidebar"] * { color: var(--ink) !important; }
    [data-testid="stSidebar"] .stRadio label {
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
    }

    /* ── Headings ─────────────────────────────────────────── */
    h1 {
        font-family: 'DM Serif Display', serif !important;
        font-size: 2.4rem !important;
        color: var(--ink) !important;
        letter-spacing: -0.5px;
        line-height: 1.15;
    }
    h2, h3 {
        font-family: 'DM Serif Display', serif !important;
        color: var(--slate) !important;
        letter-spacing: -0.3px;
    }
    h4, h5 {
        font-family: 'DM Sans', sans-serif !important;
        color: var(--slate) !important;
        font-weight: 600;
        letter-spacing: 0.3px;
        text-transform: uppercase;
        font-size: 0.75rem !important;
    }

    /* ── Metric Cards ──────────────────────────────────────── */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.80);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(217,212,244,0.6);
        border-radius: 16px;
        padding: 20px 18px !important;
        box-shadow: 0 4px 20px rgba(44,44,58,0.07),
                    0 1px 4px rgba(217,212,244,0.4);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(44,44,58,0.10);
    }
    [data-testid="metric-container"] label {
        color: var(--muted) !important;
        font-size: 11px !important;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600 !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: var(--ink) !important;
        font-family: 'DM Serif Display', serif !important;
        font-size: 2rem !important;
        font-weight: 400 !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] {
        color: #7B8FCC !important;
        font-size: 12px !important;
    }

    /* ── Tabs ──────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.6);
        backdrop-filter: blur(10px);
        border-radius: 12px 12px 0 0;
        border-bottom: 2px solid rgba(217,212,244,0.5);
        padding: 4px 4px 0;
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        color: var(--muted) !important;
        font-weight: 500 !important;
        font-size: 13px !important;
        border-radius: 8px 8px 0 0;
        padding: 10px 18px !important;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        color: var(--ink) !important;
        background: rgba(246,181,181,0.25) !important;
        border-bottom: 3px solid #F6B5B5 !important;
    }

    /* ── Buttons ───────────────────────────────────────────── */
    .stButton button {
        background: linear-gradient(135deg, #D9D4F4, #C8EBF8);
        color: var(--ink) !important;
        border: 1px solid rgba(217,212,244,0.8);
        border-radius: 12px;
        font-weight: 600;
        font-size: 13px;
        letter-spacing: 0.3px;
        padding: 10px 20px;
        box-shadow: 0 2px 8px rgba(217,212,244,0.4);
        transition: all 0.2s ease;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #F6B5B5, #FDDCB8);
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(246,181,181,0.35);
    }

    /* ── DataFrames ─────────────────────────────────────────── */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(217,212,244,0.5);
        border-radius: 12px;
        overflow: hidden;
        background: rgba(255,255,255,0.75);
        backdrop-filter: blur(8px);
    }

    /* ── Dividers ───────────────────────────────────────────── */
    hr { border-color: rgba(217,212,244,0.5) !important; }

    /* ── Info / Warning boxes ───────────────────────────────── */
    .stInfo, [data-testid="stAlert"] {
        background: rgba(200,235,248,0.4) !important;
        border: 1px solid rgba(200,235,248,0.7) !important;
        border-radius: 12px !important;
    }
    .stWarning {
        background: rgba(253,220,184,0.4) !important;
        border: 1px solid rgba(253,220,184,0.7) !important;
        border-radius: 12px !important;
    }

    /* ── Live pulse ─────────────────────────────────────────── */
    @keyframes seismic {
        0%,100%{ opacity:1; transform:scale(1); }
        50%{ opacity:0.5; transform:scale(1.4); }
    }
    .live-dot {
        display:inline-block; width:10px; height:10px;
        background:#F6B5B5; border-radius:50%;
        animation:seismic 1.6s infinite;
        margin-right:8px;
        box-shadow:0 0 10px rgba(246,181,181,0.8);
    }

    /* ── Alert cards ─────────────────────────────────────────── */
    .alert-card {
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 10px;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.5);
        box-shadow: 0 2px 12px rgba(44,44,58,0.06);
        transition: transform 0.2s ease;
    }
    .alert-card:hover { transform: translateX(4px); }

    /* ── Section labels ──────────────────────────────────────── */
    .section-header {
        background: linear-gradient(90deg, rgba(217,212,244,0.4), transparent);
        padding: 10px 18px;
        border-left: 4px solid #D9D4F4;
        border-radius: 0 10px 10px 0;
        margin-bottom: 14px;
        color: var(--slate);
        font-weight: 600;
        font-size: 13px;
        letter-spacing: 0.5px;
    }

    /* ── Monospace for IDs ───────────────────────────────────── */
    code, .mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ── Plotly chart containers ─────────────────────────────── */
    .js-plotly-plot {
        border-radius: 16px !important;
        overflow: hidden;
    }

    /* ── Sidebar nav items ───────────────────────────────────── */
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        border-radius: 10px;
        padding: 8px 12px !important;
        margin: 2px 0;
        transition: background 0.2s;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(217,212,244,0.3);
    }
</style>
""", unsafe_allow_html=True)

# ── Pastel Theme for Plotly ───────────────────────────────────────────────────
PLOT_BG    = 'rgba(255,255,255,0.85)'
PLOT_PAPER = 'rgba(250,250,248,0.0)'
GRID_COLOR = 'rgba(217,212,244,0.4)'
TEXT_COLOR = '#2C2C3A'
ACCENT     = '#A89CD4'          # desaturated lavender accent

PLOT_THEME = dict(
    paper_bgcolor=PLOT_PAPER,
    plot_bgcolor=PLOT_BG,
    font=dict(color=TEXT_COLOR, family='DM Sans, sans-serif'),
    title_font=dict(color='#4A4A6A', size=15, family='DM Serif Display, serif'),
    xaxis=dict(gridcolor=GRID_COLOR, color='#7A7A9A',
               linecolor='rgba(217,212,244,0.6)', zerolinecolor=GRID_COLOR,
               showgrid=True),
    yaxis=dict(gridcolor=GRID_COLOR, color='#7A7A9A',
               linecolor='rgba(217,212,244,0.6)', zerolinecolor=GRID_COLOR,
               showgrid=True),
    legend=dict(bgcolor='rgba(255,255,255,0.85)', bordercolor='rgba(217,212,244,0.5)',
                borderwidth=1, font=dict(color=TEXT_COLOR, size=12)),
    margin=dict(l=50, r=20, t=50, b=40),
)

# ── Magnitude colors — vivid, high-contrast for charts ───────────────────────
MAG_COLORS = {
    'micro':    '#5B8DEF',   # bold blue
    'minor':    '#00BCD4',   # cyan
    'light':    '#26C485',   # emerald
    'moderate': '#F5A623',   # amber
    'strong':   '#F05A28',   # deep orange
    'major':    '#D0021B',   # crimson
}

DEPTH_COLORS = {
    'shallow':      '#F5A623',   # amber
    'intermediate': '#7B5EA7',   # purple
    'deep':         '#1A7FCC',   # ocean blue
}

ALERT_COLORS = {
    'green':  '#27AE60',
    'yellow': '#F1C40F',
    'orange': '#E67E22',
    'red':    '#E74C3C',
}

ALERT_BG = {
    'green':  'rgba(226,241,235,0.6)',
    'yellow': 'rgba(247,251,202,0.6)',
    'orange': 'rgba(253,220,184,0.6)',
    'red':    'rgba(246,181,181,0.6)',
}

# ── Map dot colours — vivid & clearly distinct at every magnitude band ────────
# <M2 : steel blue   M2-3 : cyan    M3-4 : emerald green
# M4-5 : gold       M5-6 : orange  M6-7 : crimson red   M7+ : deep magenta
MAP_MAG_PALETTE = {
    'lt2':  '#5B8DEF',   # steel blue
    '2_3':  '#00BCD4',   # vivid cyan
    '3_4':  '#26C485',   # emerald
    '4_5':  '#F5C518',   # gold
    '5_6':  '#F5A623',   # amber-orange
    '6_7':  '#F05A28',   # deep orange-red
    'gt7':  '#D0021B',   # crimson / alarm red
}

def mag_to_map_color(m):
    if   m >= 7:   return MAP_MAG_PALETTE['gt7']
    elif m >= 6:   return MAP_MAG_PALETTE['6_7']
    elif m >= 5:   return MAP_MAG_PALETTE['5_6']
    elif m >= 4:   return MAP_MAG_PALETTE['4_5']
    elif m >= 3:   return MAP_MAG_PALETTE['3_4']
    elif m >= 2:   return MAP_MAG_PALETTE['2_3']
    else:          return MAP_MAG_PALETTE['lt2']

# ── Data functions ────────────────────────────────────────────────────────────
def query_cassandra(cql):
    try:
        result = subprocess.run(
            ['docker', 'exec', 'seismic-cassandra', 'cqlsh', '-e', cql, '--no-color'],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout or ""
    except:
        return ""

def parse_cqlsh(output):
    if not output:
        return pd.DataFrame()
    lines = [l.strip() for l in output.strip().split('\n') if l.strip()]
    rows, header = [], []
    for line in lines:
        if '|' not in line or '---' in line:
            continue
        parts = [p.strip() for p in line.split('|') if p.strip()]
        if not header:
            header = parts
        elif len(parts) == len(header):
            rows.append(dict(zip(header, parts)))
    return pd.DataFrame(rows)

@st.cache_data(ttl=30)
def load_events():
    out = query_cassandra("SELECT event_id,mag,depth_km,latitude,longitude,depth_band,mag_class,net,time_iso,tsunami_flag FROM seismic.events LIMIT 3000;")
    df  = parse_cqlsh(out)
    for c in ['mag','depth_km','latitude','longitude','tsunami_flag']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

@st.cache_data(ttl=20)
def load_alerts():
    out = query_cassandra("SELECT event_id,mag,alert_level,alert_date,alert_time,place,latitude,longitude,tsunami_flag FROM seismic.alerts LIMIT 500;")
    df  = parse_cqlsh(out)
    if 'mag' in df.columns:
        df['mag'] = pd.to_numeric(df['mag'], errors='coerce')
    return df

@st.cache_data(ttl=60)
def load_predictions():
    out = query_cassandra("SELECT event_id,actual_mag,predicted_mag,prediction_error,depth_km,depth_band FROM seismic.magnitude_predictions LIMIT 2000;")
    df  = parse_cqlsh(out)
    for c in ['actual_mag','predicted_mag','prediction_error','depth_km']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

@st.cache_data(ttl=20)
def load_live_usgs():
    try:
        r    = requests.get('https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson', timeout=10)
        data = r.json()
        events = []
        for f in data['features']:
            p = f['properties']
            c = f['geometry']['coordinates']
            if p.get('mag') is None: continue
            events.append({
                'event_id': f['id'], 'mag': float(p.get('mag',0)),
                'place': p.get('place',''), 'time': datetime.fromtimestamp(p['time']/1000),
                'longitude': float(c[0]), 'latitude': float(c[1]),
                'depth_km': float(c[2]), 'net': p.get('net',''),
            })
        return pd.DataFrame(events)
    except:
        return pd.DataFrame()

@st.cache_data(ttl=30)
def load_kafka_alerts():
    try:
        result = subprocess.run(
            ['docker', 'exec', 'seismic-kafka', 'kafka-console-consumer',
             '--bootstrap-server', 'localhost:9092',
             '--topic', 'seismic-alerts',
             '--from-beginning', '--max-messages', '100',
             '--timeout-ms', '5000'],
            capture_output=True, text=True, timeout=15
        )
        records = []
        for line in result.stdout.strip().split('\n'):
            line = line.strip()
            if not line or not line.startswith('{'):
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except:
                continue
        return pd.DataFrame(records) if records else pd.DataFrame()
    except:
        return pd.DataFrame()

@st.cache_data(ttl=30)
def load_kafka_waveforms():
    try:
        result = subprocess.run(
            ['docker', 'exec', 'seismic-kafka', 'kafka-console-consumer',
             '--bootstrap-server', 'localhost:9092',
             '--topic', 'seismic-waveforms',
             '--from-beginning', '--max-messages', '50',
             '--timeout-ms', '5000'],
            capture_output=True, text=True, timeout=15
        )
        records = []
        for line in result.stdout.strip().split('\n'):
            line = line.strip()
            if not line or not line.startswith('{'):
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except:
                continue
        return pd.DataFrame(records) if records else pd.DataFrame()
    except:
        return pd.DataFrame()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌋 Seismic Intelligence")
    st.markdown("*Real-Time Earthquake System*")
    st.markdown("---")
    st.markdown('<span class="live-dot"></span> **LIVE · ACTIVE**', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("", [
        "📡 Live Monitor",
        "📊 Data Analytics",
        "🚨 Alerts & Waveforms",
        "🤖 ML Insights",
        "🔧 Pipeline Status"
    ])
    st.markdown("---")
    st.markdown("**🔗 Data Flow**")
    st.markdown("🌍 USGS API")
    st.markdown("↓ Apache Kafka (3 topics)")
    st.markdown("↓ Spark ETL / Streaming")
    st.markdown("↓ Cassandra + HBase")
    st.markdown("↓ ML Models")
    st.markdown("↓ This Dashboard")
    st.markdown("---")
    if st.button("🔄 Refresh All Data"):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    st.caption(f"⏱ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.caption("Big Data Engineering Project")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — LIVE MONITOR
# ═══════════════════════════════════════════════════════════════════════════
if page == "📡 Live Monitor":
    st.title("📡 Live Earthquake Monitor")
    st.markdown("*Real earthquakes from USGS — updating every 60 seconds*")

    live_df   = load_live_usgs()
    events_df = load_events()

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("🌍 Events Last Hour", len(live_df) if not live_df.empty else 0)
    with c2:
        if not live_df.empty:
            st.metric("⚡ Max Magnitude", f"M{live_df['mag'].max():.1f}")
        else:
            st.metric("⚡ Max Magnitude", "—")
    with c3:
        st.metric("🗄️ In Cassandra", f"{len(events_df):,}" if not events_df.empty else "—")
    with c4:
        if not live_df.empty:
            st.metric("🚨 M4.0+ Now", len(live_df[live_df['mag']>=4.0]))
        else:
            st.metric("🚨 M4.0+ Now", "—")

    st.markdown("---")

    if not live_df.empty:
        st.markdown("### 🗺️ Global Seismic Map — Last 60 Minutes")

        # Per-event colour and size — clearly differentiated
        live_df['dot_color'] = live_df['mag'].apply(mag_to_map_color)
        live_df['dot_size']  = live_df['mag'].apply(lambda m: max(5, m * 5.5))
        live_df['label']     = live_df.apply(
            lambda r: f"M{r['mag']:.1f} | {r['place']} | {r['depth_km']:.1f}km deep", axis=1)

        fig_map = go.Figure()

        # Pulse rings for M5+
        for _, row in live_df[live_df['mag'] >= 5].iterrows():
            fig_map.add_trace(go.Scattergeo(
                lon=[row['longitude']], lat=[row['latitude']], mode='markers',
                marker=dict(size=row['mag'] * 10,
                            color='rgba(246,181,181,0.18)',
                            line=dict(color='rgba(246,181,181,0.5)', width=1.5)),
                showlegend=False, hoverinfo='skip'
            ))

        # Group by magnitude band so legend is clean
        bands = [
            ('<M2',  MAP_MAG_PALETTE['lt2'],  live_df[live_df['mag']<2]),
            ('M2-3', MAP_MAG_PALETTE['2_3'],  live_df[(live_df['mag']>=2)&(live_df['mag']<3)]),
            ('M3-4', MAP_MAG_PALETTE['3_4'],  live_df[(live_df['mag']>=3)&(live_df['mag']<4)]),
            ('M4-5', MAP_MAG_PALETTE['4_5'],  live_df[(live_df['mag']>=4)&(live_df['mag']<5)]),
            ('M5-6', MAP_MAG_PALETTE['5_6'],  live_df[(live_df['mag']>=5)&(live_df['mag']<6)]),
            ('M6-7', MAP_MAG_PALETTE['6_7'],  live_df[(live_df['mag']>=6)&(live_df['mag']<7)]),
            ('M7+',  MAP_MAG_PALETTE['gt7'],  live_df[live_df['mag']>=7]),
        ]

        for label, color, sub in bands:
            if sub.empty:
                continue
            fig_map.add_trace(go.Scattergeo(
                lon=sub['longitude'], lat=sub['latitude'],
                text=sub['label'], mode='markers',
                name=label,
                marker=dict(
                    size=sub['dot_size'],
                    color=color,
                    opacity=0.92,
                    line=dict(color='rgba(44,44,58,0.35)', width=0.8)
                ),
                hovertemplate='<b>%{text}</b><extra></extra>',
            ))

        fig_map.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            geo=dict(
                bgcolor='#EEF6FB',
                landcolor='#E2F1EB',
                oceancolor='#C8EBF8',
                showocean=True, showland=True, showcoastlines=True,
                coastlinecolor='#A0C4D8',
                countrycolor='#B8D8E8',
                showframe=False,
                projection_type='natural earth',
                lakecolor='#C8EBF8',
                showcountries=True,
            ),
            margin=dict(l=0,r=0,t=40,b=0), height=520,
            title=dict(text="🌍 Live Global Seismicity", font=dict(color=ACCENT, size=16,
                        family='DM Serif Display, serif')),
            legend=dict(
                bgcolor='rgba(255,255,255,0.85)',
                bordercolor='rgba(217,212,244,0.6)',
                borderwidth=1,
                font=dict(color=TEXT_COLOR, size=11),
                title=dict(text='Magnitude', font=dict(color=TEXT_COLOR, size=12)),
                x=0.01, y=0.05
            )
        )
        st.plotly_chart(fig_map, use_container_width=True)

        col_a, col_b = st.columns([2,1])
        with col_a:
            st.markdown("### ⚡ Recent Events")
            disp = live_df[['time','mag','place','depth_km','net']].copy()
            disp.columns = ['Time (UTC)','Magnitude','Location','Depth (km)','Network']
            disp = disp.sort_values('Magnitude', ascending=False).head(20)
            st.dataframe(disp, use_container_width=True, hide_index=True)

        with col_b:
            st.markdown("### 📈 Magnitude Distribution")
            mag_bins = pd.cut(live_df['mag'],
                bins=[-5,1,2,3,4,5,6,7,10],
                labels=['<1','1-2','2-3','3-4','4-5','5-6','6-7','7+'])
            mag_counts = mag_bins.value_counts().sort_index()
            bar_colors = ['#5B8DEF', '#5B8DEF',
                          '#00BCD4', '#26C485',
                          '#F5C518', '#F5A623',
                          '#F05A28', '#D0021B']
            fig_bar = go.Figure(go.Bar(
                x=mag_counts.index.astype(str),
                y=mag_counts.values,
                marker=dict(color=bar_colors[:len(mag_counts)],
                            line=dict(color='rgba(44,44,58,0.15)', width=1)),
                text=mag_counts.values, textposition='outside',
                textfont=dict(color=TEXT_COLOR, size=11)
            ))
            fig_bar.update_layout(**PLOT_THEME, height=340,
                                  xaxis_title="Magnitude Range",
                                  yaxis_title="Count",
                                  showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("⚠️ Could not fetch live USGS data. Check internet connection.")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATA ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Analytics":
    st.title("📊 Seismic Data Analytics")
    st.markdown("*72,993 earthquakes • Spark SQL • 5 analytical queries*")

    events_df = load_events()
    if events_df.empty:
        st.error("❌ No data from Cassandra. Ensure Docker containers are running.")
        st.stop()

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Total Events", f"{len(events_df):,}")
    with c2:
        if 'mag' in events_df.columns:
            st.metric("Avg Magnitude", f"M{events_df['mag'].mean():.2f}")
    with c3:
        if 'mag' in events_df.columns:
            st.metric("Max Magnitude", f"M{events_df['mag'].max():.1f}")
    with c4:
        if 'tsunami_flag' in events_df.columns:
            st.metric("Tsunami Events", int(events_df['tsunami_flag'].sum()))

    st.markdown("---")
    tab1,tab2,tab3,tab4,tab5 = st.tabs([
        "🌐 Network Hotspots","📊 Magnitude Classes",
        "🔽 Depth Analysis","📅 Daily Trend","🌊 Tsunami Risk"
    ])

    with tab1:
        st.markdown("#### SQL Query 1 — Top Seismic Networks by Activity")
        if 'net' in events_df.columns:
            nc = events_df.groupby('net').size().reset_index(name='count')
            nc = nc.sort_values('count', ascending=True).tail(15)
            fig = go.Figure(go.Bar(
                x=nc['count'], y=nc['net'], orientation='h',
                marker=dict(
                    color=nc['count'],
                    colorscale=[[0,'#5B8DEF'],[0.35,'#26C485'],
                                [0.65,'#F5C518'],[0.85,'#F05A28'],[1,'#D0021B']],
                    line=dict(color='rgba(44,44,58,0.15)', width=0.8)
                ),
                text=nc['count'], textposition='outside',
                textfont=dict(color=TEXT_COLOR, size=11)
            ))
            fig.update_layout(**PLOT_THEME, height=480,
                              xaxis_title="Number of Events",
                              yaxis_title="Seismic Network Code",
                              title="Seismic Network Activity Rankings")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("#### SQL Query 2 — Magnitude Frequency (Gutenberg-Richter Law)")
        if 'mag_class' in events_df.columns:
            mc = events_df['mag_class'].value_counts().reset_index()
            mc.columns = ['mag_class','count']
            order = ['micro','minor','light','moderate','strong','major']
            mc['mag_class'] = pd.Categorical(mc['mag_class'], categories=order, ordered=True)
            mc = mc.sort_values('mag_class')

            ca, cb = st.columns(2)
            with ca:
                fig = go.Figure(go.Bar(
                    x=mc['mag_class'], y=mc['count'],
                    marker=dict(color=[MAG_COLORS.get(m,'#D9D4F4') for m in mc['mag_class']],
                                line=dict(color='rgba(44,44,58,0.12)', width=0.8)),
                    text=mc['count'], textposition='outside',
                    textfont=dict(color=TEXT_COLOR)
                ))
                fig.update_layout(**PLOT_THEME, height=360,
                                  title="Event Count by Magnitude Class",
                                  xaxis_title="Class", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
            with cb:
                fig2 = go.Figure(go.Pie(
                    labels=mc['mag_class'], values=mc['count'],
                    marker=dict(colors=[MAG_COLORS.get(m,'#D9D4F4') for m in mc['mag_class']],
                                line=dict(color='rgba(255,255,255,0.8)', width=2)),
                    textfont=dict(color=TEXT_COLOR, size=11), hole=0.45,
                    textinfo='label+percent'
                ))
                fig2.update_layout(paper_bgcolor=PLOT_PAPER,
                                   font=dict(color=TEXT_COLOR),
                                   height=360, showlegend=False,
                                   title=dict(text="Distribution",
                                              font=dict(color=ACCENT,
                                                        family='DM Serif Display, serif')))
                st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown("#### SQL Query 3 — Depth Band Tectonic Analysis")
        if 'depth_band' in events_df.columns and 'mag' in events_df.columns:
            ds = events_df.groupby('depth_band').agg(
                count=('mag','count'), avg_mag=('mag','mean'),
                max_mag=('mag','max'), avg_depth=('depth_km','mean')
            ).reset_index()

            ca, cb = st.columns(2)
            with ca:
                fig = go.Figure()
                for band in ['shallow','intermediate','deep']:
                    sub = events_df[events_df['depth_band']==band]['mag']
                    if not sub.empty:
                        fig.add_trace(go.Histogram(
                            x=sub, name=band,
                            marker_color=DEPTH_COLORS.get(band,'#D9D4F4'),
                            opacity=0.8, nbinsx=25
                        ))
                fig.update_layout(**PLOT_THEME, barmode='overlay', height=360,
                                  title="Magnitude Distribution by Depth",
                                  xaxis_title="Magnitude", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
            with cb:
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    name='Avg Magnitude',
                    x=ds['depth_band'], y=ds['avg_mag'],
                    marker_color=[DEPTH_COLORS.get(b,'#D9D4F4') for b in ds['depth_band']],
                    text=ds['avg_mag'].round(2), textposition='outside',
                    textfont=dict(color=TEXT_COLOR)
                ))
                fig2.update_layout(**PLOT_THEME, height=360,
                                   title="Avg Magnitude by Depth Band",
                                   yaxis_title="Avg Magnitude")
                st.plotly_chart(fig2, use_container_width=True)

            st.dataframe(ds.round(3), use_container_width=True, hide_index=True)

    with tab4:
        st.markdown("#### Bonus Query — Daily Earthquake Trend (30 Days)")
        if 'time_iso' in events_df.columns:
            daily = events_df.copy()
            daily['date'] = pd.to_datetime(daily['time_iso'], errors='coerce').dt.date
            dc = daily.groupby('date').agg(
                count=('mag','count'), avg_mag=('mag','mean'), max_mag=('mag','max')
            ).reset_index().dropna().sort_values('date').tail(30)

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(
                x=dc['date'], y=dc['count'], name='Daily Events',
                marker=dict(color='#5B8DEF', opacity=0.85,
                            line=dict(color='#3A6FD0', width=0.8))
            ), secondary_y=False)
            fig.add_trace(go.Scatter(
                x=dc['date'], y=dc['max_mag'], name='Max Magnitude',
                mode='lines+markers',
                line=dict(color='#F05A28', width=2.5),
                marker=dict(color='#F05A28', size=6,
                            line=dict(color='white', width=1.5))
            ), secondary_y=True)
            fig.update_layout(**PLOT_THEME, height=420,
                              title="30-Day Seismic Activity Trend")
            fig.update_yaxes(title_text="Event Count", secondary_y=False,
                             gridcolor=GRID_COLOR, color='#7A7A9A')
            fig.update_yaxes(title_text="Max Magnitude", secondary_y=True,
                             color='#C09090')
            st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.markdown("#### SQL Query 5 — Tsunami Risk Events")
        if 'tsunami_flag' in events_df.columns:
            ts = events_df[events_df['tsunami_flag'] == 1]
            if not ts.empty:
                c1,c2,c3 = st.columns(3)
                with c1: st.metric("🌊 Tsunami-Flagged", len(ts))
                with c2: st.metric("Avg Magnitude", f"M{ts['mag'].mean():.2f}")
                with c3: st.metric("Max Magnitude", f"M{ts['mag'].max():.1f}")

                nt = ts.groupby('net').agg(
                    count=('mag','count'), avg_mag=('mag','mean'), max_mag=('mag','max')
                ).reset_index().sort_values('count', ascending=False)
                fig = go.Figure(go.Bar(
                    x=nt['net'], y=nt['count'],
                    marker=dict(color='#1A7FCC',
                                line=dict(color='#0D5FA0', width=0.8)),
                    text=nt['count'], textposition='outside',
                    textfont=dict(color=TEXT_COLOR)
                ))
                fig.update_layout(**PLOT_THEME, height=350,
                                  title="🌊 Tsunami Events by Network",
                                  yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(nt.round(2), use_container_width=True, hide_index=True)
            else:
                st.info("No tsunami-flagged events in current Cassandra dataset.")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 — ALERTS & WAVEFORMS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🚨 Alerts & Waveforms":
    st.title("🚨 Alerts & Seismic Waveforms")
    st.markdown("*Live M4.0+ alerts from Kafka + waveform data from 5 global stations*")

    tab_alerts, tab_waveforms = st.tabs(["🚨 Alert System", "📡 Waveform Data"])

    with tab_alerts:
        cass_alerts   = load_alerts()
        kafka_alerts  = load_kafka_alerts()
        live_df       = load_live_usgs()

        c1,c2,c3,c4 = st.columns(4)
        total_alerts = len(cass_alerts) if not cass_alerts.empty else 0
        kafka_count  = len(kafka_alerts) if not kafka_alerts.empty else 0
        with c1: st.metric("Cassandra Alerts", total_alerts, "written by streaming")
        with c2: st.metric("Kafka Alerts", kafka_count, "seismic-alerts topic")
        with c3:
            if not live_df.empty:
                st.metric("Live M5+ Now", len(live_df[live_df['mag']>=5.0]))
        with c4:
            if not live_df.empty:
                st.metric("Live M4+ Now", len(live_df[live_df['mag']>=4.0]))

        st.markdown("---")

        if not cass_alerts.empty:
            st.markdown("### 📋 Cassandra Alert Log (from Spark Streaming)")
            for _, row in cass_alerts.iterrows():
                level = str(row.get('alert_level','green'))
                mag   = float(row.get('mag', 0) or 0)
                eid   = str(row.get('event_id',''))
                place = str(row.get('place','Unknown location'))
                date  = str(row.get('alert_date',''))
                icon  = {'red':'🔴','orange':'🟠','yellow':'🟡','green':'🟢'}.get(level,'⚪')
                bc    = ALERT_COLORS.get(level,'#D9D4F4')
                bg    = ALERT_BG.get(level,'rgba(217,212,244,0.3)')
                st.markdown(f"""
                <div class="alert-card" style="background:{bg};border-color:{bc};">
                    <span style="background:{bc};color:{TEXT_COLOR};padding:4px 12px;
                                 border-radius:20px;font-size:11px;font-weight:700;
                                 letter-spacing:1px;">{icon} {level.upper()}</span>
                    <span style="font-family:'DM Serif Display',serif;font-size:22px;
                                 color:#4A4A6A;margin-left:14px;">M{mag:.1f}</span>
                    <span style="color:#7A7A9A;margin-left:10px;font-size:14px;">{place}</span>
                    <br><span style="color:#A0A0B8;font-size:11px;font-family:'JetBrains Mono',monospace;
                                     margin-top:6px;display:block;">{eid} · {date}</span>
                </div>
                """, unsafe_allow_html=True)

        if not kafka_alerts.empty:
            st.markdown("### 📨 Kafka Alert Stream (seismic-alerts topic)")
            if 'mag' in kafka_alerts.columns:
                kafka_alerts['mag'] = pd.to_numeric(kafka_alerts['mag'], errors='coerce')
            display_cols = [c for c in ['event_id','mag','alert_level','place','time_iso','latitude','longitude'] if c in kafka_alerts.columns]
            if display_cols:
                st.dataframe(kafka_alerts[display_cols].head(20),
                             use_container_width=True, hide_index=True)

            if 'mag' in kafka_alerts.columns and not kafka_alerts['mag'].isna().all():
                st.markdown("#### Alert Severity Distribution")
                if 'alert_level' in kafka_alerts.columns:
                    level_counts = kafka_alerts['alert_level'].value_counts()
                    fig = go.Figure(go.Bar(
                        x=level_counts.index,
                        y=level_counts.values,
                        marker=dict(color=[ALERT_COLORS.get(l,'#5B8DEF') for l in level_counts.index],
                                    line=dict(color='rgba(44,44,58,0.15)', width=0.8)),
                        text=level_counts.values, textposition='outside',
                        textfont=dict(color=TEXT_COLOR)
                    ))
                    fig.update_layout(**PLOT_THEME, height=300,
                                      title="Alert Levels from Kafka",
                                      xaxis_title="Alert Level", yaxis_title="Count")
                    st.plotly_chart(fig, use_container_width=True)

        if not live_df.empty:
            m4 = live_df[live_df['mag'] >= 4.0].copy()
            if not m4.empty:
                st.markdown("### 🗺️ Live M4.0+ Events Map (Last Hour)")
                m4['dot_color'] = m4['mag'].apply(mag_to_map_color)
                m4['dot_size']  = m4['mag'].apply(lambda m: max(6, m * 6))

                fig = go.Figure()
                for label, color, sub in [
                    ('M4-5', MAP_MAG_PALETTE['4_5'], m4[(m4['mag']>=4)&(m4['mag']<5)]),
                    ('M5-6', MAP_MAG_PALETTE['5_6'], m4[(m4['mag']>=5)&(m4['mag']<6)]),
                    ('M6-7', MAP_MAG_PALETTE['6_7'], m4[(m4['mag']>=6)&(m4['mag']<7)]),
                    ('M7+',  MAP_MAG_PALETTE['gt7'], m4[m4['mag']>=7]),
                ]:
                    if sub.empty: continue
                    fig.add_trace(go.Scattergeo(
                        lon=sub['longitude'], lat=sub['latitude'],
                        text=sub.apply(lambda r: f"M{r['mag']:.1f} — {r['place']}", axis=1),
                        mode='markers', name=label,
                        marker=dict(
                            size=sub['dot_size'], color=color, opacity=0.92,
                            line=dict(color='rgba(44,44,58,0.3)', width=1)
                        ),
                        hovertemplate='<b>%{text}</b><extra></extra>'
                    ))

                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    geo=dict(bgcolor='#EEF6FB', landcolor='#E2F1EB', oceancolor='#C8EBF8',
                             showocean=True, showland=True, showcoastlines=True,
                             coastlinecolor='#A0C4D8', showframe=False,
                             projection_type='natural earth', countrycolor='#B8D8E8'),
                    height=440,
                    title=dict(text="🚨 M4.0+ Events — Last Hour",
                               font=dict(color=ACCENT, size=15,
                                         family='DM Serif Display, serif')),
                    font=dict(color=TEXT_COLOR),
                    legend=dict(bgcolor='rgba(255,255,255,0.85)',
                                bordercolor='rgba(217,212,244,0.6)', borderwidth=1,
                                font=dict(color=TEXT_COLOR, size=11))
                )
                st.plotly_chart(fig, use_container_width=True)

                disp = m4[['time','mag','place','depth_km','net']].copy()
                disp.columns = ['Time','Magnitude','Location','Depth (km)','Network']
                st.dataframe(disp.sort_values('Magnitude', ascending=False),
                             use_container_width=True, hide_index=True)
            else:
                st.success("✅ No M4.0+ events in the last hour — seismically quiet!")

    with tab_waveforms:
        st.markdown("### 📡 Seismic Waveform Data")
        st.markdown("*From seismic-waveforms Kafka topic — 5 global IRIS stations*")

        wf_df = load_kafka_waveforms()

        if not wf_df.empty:
            c1,c2,c3 = st.columns(3)
            with c1: st.metric("Waveform Records", len(wf_df))
            with c2:
                if 'station_id' in wf_df.columns:
                    st.metric("Stations", wf_df['station_id'].nunique())
            with c3:
                if 'eq_mag' in wf_df.columns:
                    wf_df['eq_mag'] = pd.to_numeric(wf_df['eq_mag'], errors='coerce')
                    st.metric("Events Covered", wf_df['event_id'].nunique() if 'event_id' in wf_df.columns else "—")

            st.markdown("---")

            if 'station_id' in wf_df.columns and 'distance_km' in wf_df.columns:
                wf_df['distance_km'] = pd.to_numeric(wf_df['distance_km'], errors='coerce')
                st.markdown("#### Station Coverage")
                station_stats = wf_df.groupby('station_id').agg(
                    records=('event_id','count'),
                    avg_dist=('distance_km','mean'),
                    station_name=('station_name','first')
                ).reset_index().sort_values('records', ascending=False)

                fig = go.Figure(go.Bar(
                    x=station_stats['station_id'],
                    y=station_stats['records'],
                    marker=dict(
                        color=station_stats['records'],
                        colorscale=[[0,'#5B8DEF'],[0.5,'#26C485'],[1,'#F05A28']],
                        line=dict(color='rgba(44,44,58,0.12)', width=0.8)
                    ),
                    text=station_stats['records'],
                    textposition='outside',
                    textfont=dict(color=TEXT_COLOR),
                    customdata=station_stats['station_name'],
                    hovertemplate='<b>%{x}</b><br>%{customdata}<br>Records: %{y}<extra></extra>'
                ))
                fig.update_layout(**PLOT_THEME, height=300,
                                  title="Waveform Records by Station",
                                  xaxis_title="Station ID", yaxis_title="Records")
                st.plotly_chart(fig, use_container_width=True)

            if 'channel_bhz' in wf_df.columns and 'event_id' in wf_df.columns:
                st.markdown("#### Sample Waveform — Latest Event")
                latest_event = wf_df.iloc[0]
                try:
                    waveform_data = json.loads(latest_event['channel_bhz']) if isinstance(latest_event['channel_bhz'], str) else latest_event['channel_bhz']
                    if isinstance(waveform_data, list) and len(waveform_data) > 0:
                        t = list(range(len(waveform_data)))
                        fig_wf = go.Figure()
                        fig_wf.add_trace(go.Scatter(
                            x=t, y=waveform_data, mode='lines',
                            line=dict(color='#F05A28', width=1.5),
                            name='BHZ (Vertical)',
                            fill='tozeroy',
                            fillcolor='rgba(240,90,40,0.12)'
                        ))
                        mag_val = latest_event.get('eq_mag','?')
                        sta_val = latest_event.get('station_id','?')
                        fig_wf.update_layout(**PLOT_THEME, height=300,
                                             title=f"Waveform: M{mag_val} event at {sta_val}",
                                             xaxis_title="Sample Index",
                                             yaxis_title="Amplitude (nm/s)")
                        st.plotly_chart(fig_wf, use_container_width=True)
                except:
                    st.info("Waveform data available but could not render visualization.")

            st.markdown("#### Waveform Records")
            display_cols = [c for c in ['event_id','station_id','station_name',
                                         'eq_mag','distance_km','snr','time_iso']
                           if c in wf_df.columns]
            if display_cols:
                st.dataframe(wf_df[display_cols].head(30),
                             use_container_width=True, hide_index=True)
        else:
            st.info("""
            **No waveform data loaded yet.**
            
            Run the waveform producer to populate data:
            ```
            python kafka/producers/waveform_producer.py
            ```
            Then click **Refresh All Data** in the sidebar.
            """)

        st.markdown("---")
        st.markdown("#### HBase Waveform Schema Design")
        st.markdown("""
        | Element | Design |
        |---------|--------|
        | **Table** | `seismic_waveforms` |
        | **Row Key** | `event_id#station_id#timestamp_ms` |
        | **CF: cf_amplitude** | BHZ, BHN, BHE channel data |
        | **CF: cf_meta** | sampling_rate, units, network_code |
        | **CF: cf_quality** | SNR, gaps_count, completeness_pct |
        | **TTL** | 180 days auto-expiry |
        | **Compression** | SNAPPY on cf_amplitude |
        """)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4 — ML INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Insights":
    st.title("🤖 ML Model Insights")
    st.markdown("*RandomForest magnitude prediction + KMeans tectonic zone clustering*")

    pred_df = load_predictions()

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("RF RMSE", "0.1048", "↓ target < 0.65")
    with c2: st.metric("RF R²",   "0.9929", "↑ target > 0.70")
    with c3: st.metric("RF MAE",  "0.0666", "↓ target < 0.45")
    with c4: st.metric("Silhouette", "0.7282", "↑ target > 0.40")

    st.markdown("---")
    tab1,tab2,tab3 = st.tabs([
        "🎯 Predictions vs Actual",
        "📊 Feature Importance",
        "🌐 Tectonic Clusters"
    ])

    with tab1:
        st.markdown("#### RandomForest: Actual vs Predicted Magnitude")
        if not pred_df.empty and 'actual_mag' in pred_df.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pred_df['actual_mag'], y=pred_df['predicted_mag'],
                mode='markers',
                marker=dict(
                    color=pred_df['prediction_error'],
                    colorscale=[[0,'#26C485'],[0.3,'#F5C518'],
                                [0.6,'#F05A28'],[1,'#D0021B']],
                    size=5, opacity=0.75,
                    colorbar=dict(title='Error', tickfont=dict(color=TEXT_COLOR),
                                  titlefont=dict(color=ACCENT)),
                    line=dict(color='rgba(44,44,58,0.1)', width=0.5)
                ),
                name='Predictions',
                hovertemplate='Actual: M%{x:.2f}<br>Predicted: M%{y:.2f}<extra></extra>'
            ))
            mx = max(pred_df['actual_mag'].max(), pred_df['predicted_mag'].max())
            mn = min(pred_df['actual_mag'].min(), pred_df['predicted_mag'].min())
            fig.add_trace(go.Scatter(
                x=[mn,mx], y=[mn,mx], mode='lines',
                line=dict(color='#F05A28', dash='dash', width=2),
                name='Perfect prediction'
            ))
            fig.update_layout(**PLOT_THEME, height=480,
                              xaxis_title="Actual Magnitude",
                              yaxis_title="Predicted Magnitude",
                              title="RF Regression: Actual vs Predicted (color = error magnitude)")
            st.plotly_chart(fig, use_container_width=True)

            ca, cb = st.columns(2)
            with ca:
                fig2 = go.Figure(go.Histogram(
                    x=pred_df['prediction_error'], nbinsx=60,
                    marker=dict(color='#5B8DEF', opacity=0.85,
                                line=dict(color='#3A6FD0', width=0.8))
                ))
                fig2.update_layout(**PLOT_THEME, height=300,
                                   xaxis_title="Absolute Error (Richter)",
                                   yaxis_title="Frequency",
                                   title="Prediction Error Distribution")
                st.plotly_chart(fig2, use_container_width=True)
            with cb:
                st.metric("RMSE", "0.1048", "6× better than target")
                st.metric("R²", "0.9929", "Near-perfect fit")
                st.metric("MAE", "0.0666", "±0.07 Richter avg error")
                st.metric("Test samples", f"{len(pred_df):,}")
        else:
            st.info("No prediction data in Cassandra. Run 05a_ml_magnitude_model.py first.")

    with tab2:
        st.markdown("#### Feature Importances — What Predicts Earthquake Magnitude?")
        feat_df = pd.DataFrame([
            ('sig (significance score)', 0.5672),
            ('rms (seismic noise)',       0.1615),
            ('longitude',                 0.1177),
            ('latitude',                  0.0914),
            ('nst (station count)',        0.0361),
            ('depth_km',                  0.0181),
            ('gap (azimuthal gap)',        0.0076),
            ('tsunami_flag',              0.0002),
        ], columns=['Feature','Importance']).sort_values('Importance', ascending=True)

        fig = go.Figure(go.Bar(
            x=feat_df['Importance'], y=feat_df['Feature'], orientation='h',
            marker=dict(
                color=feat_df['Importance'],
                colorscale=[[0,'#5B8DEF'],[0.3,'#26C485'],
                            [0.6,'#F5C518'],[0.85,'#F05A28'],[1,'#D0021B']],
                line=dict(color='rgba(44,44,58,0.12)', width=0.8)
            ),
            text=feat_df['Importance'].apply(lambda x: f'{x*100:.1f}%'),
            textposition='outside',
            textfont=dict(color=TEXT_COLOR, size=12)
        ))
        fig.update_layout(**PLOT_THEME, height=420,
                          xaxis_title="Importance Score",
                          title="Feature Importance in Magnitude Prediction")
        st.plotly_chart(fig, use_container_width=True)

        st.info("""
        **📊 Key Finding:** The `sig` (significance) feature dominates at 56.7% because USGS 
        derives it from magnitude itself. This explains the near-perfect R²=0.99. In a 
        production seismic monitoring system, we would exclude derived features and test 
        raw parameters only — demonstrating the model's ability to learn from pure 
        seismic signals (depth, gap, rms, station count).
        """)

    with tab3:
        st.markdown("#### KMeans Clustering — 8 Global Tectonic Zones")
        cluster_df = pd.DataFrame({
            'cluster_id': [0,1,2,3,4,5,6,7],
            'n_events':   [20379,3801,5315,243,2580,330,39518,827],
            'avg_mag':    [1.656,4.279,2.208,4.442,3.570,4.464,1.193,4.418],
            'avg_depth':  [14.0,21.3,98.2,458.5,32.6,522.3,6.5,139.3],
            'centroid_lat':  [52.22,31.3,55.89,1.11,9.36,-20.64,36.36,10.73],
            'centroid_lon':  [-154.52,144.03,-151.37,148.46,-54.06,-178.54,-114.74,126.47],
            'region':    ['Alaska/Canada','Japan/Pacific','Alaska Deep','PNG Deep',
                          'S. America','Tonga Deep','Western USA','Philippines'],
        })

        cluster_colors = ['#5B8DEF','#00BCD4','#26C485','#D0021B',
                          '#F05A28','#F5C518','#7B5EA7','#E91E8C']

        fig = go.Figure()
        for i, row in cluster_df.iterrows():
            fig.add_trace(go.Scattergeo(
                lon=[row['centroid_lon']], lat=[row['centroid_lat']],
                mode='markers+text',
                marker=dict(
                    size=max(14, row['n_events']/2500),
                    color=cluster_colors[i], opacity=0.92,
                    line=dict(color='rgba(44,44,58,0.35)', width=1.5),
                    symbol='circle'
                ),
                text=[f"C{int(row['cluster_id'])}"],
                textfont=dict(color=TEXT_COLOR, size=9, family='DM Sans, sans-serif'),
                name=f"C{int(row['cluster_id'])}: {row['region']} (M{row['avg_mag']:.1f})",
                hovertemplate=(
                    f"<b>Cluster {int(row['cluster_id'])}: {row['region']}</b><br>"
                    f"Events: {int(row['n_events']):,}<br>"
                    f"Avg Mag: M{row['avg_mag']}<br>"
                    f"Avg Depth: {row['avg_depth']} km<extra></extra>"
                )
            ))

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            geo=dict(
                bgcolor='#EEF6FB', landcolor='#E2F1EB', oceancolor='#C8EBF8',
                showocean=True, showland=True, showcoastlines=True,
                coastlinecolor='#A0C4D8', showframe=False,
                projection_type='natural earth',
                countrycolor='#B8D8E8', showcountries=True
            ),
            height=520,
            title=dict(text="🌍 8 KMeans Tectonic Zone Clusters — Centroid Locations",
                       font=dict(color=ACCENT, size=15,
                                 family='DM Serif Display, serif')),
            font=dict(color=TEXT_COLOR),
            legend=dict(font=dict(color=TEXT_COLOR, size=10),
                        bgcolor='rgba(255,255,255,0.85)',
                        bordercolor='rgba(217,212,244,0.5)', borderwidth=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Silhouette Score", "0.7282", "target > 0.40 ✓")
        with c2: st.metric("Anomalies", "2,450", "mag > mean + 2σ")
        with c3: st.metric("Tectonic Zones", "8", "k=8 clusters")

        st.markdown("#### Cluster Statistics")
        st.dataframe(cluster_df[['cluster_id','region','n_events','avg_mag','avg_depth',
                                  'centroid_lat','centroid_lon']],
                     use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5 — PIPELINE STATUS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔧 Pipeline Status":
    st.title("🔧 Pipeline Status & Validation")
    st.markdown("*Week 6 validation — 15/16 checks passed (93.8%)*")

    c1,c2,c3 = st.columns(3)
    with c1: st.metric("Total Checks", "16")
    with c2: st.metric("✅ Passed", "15", "93.8% pass rate")
    with c3: st.metric("❌ Failed", "1", "Windows RDD — known")

    st.markdown("---")
    checks = [
        ("V1", "Kafka topic receiving events",     "PASS","10,772 messages",   "> 0"),
        ("V2", "No null magnitudes",               "PASS","0 nulls",           "= 0"),
        ("V3", "Depth bands cover all events",     "PASS","72,993 / 72,993",   "must match"),
        ("V4", "Cassandra events table",           "PASS","10,863 rows",       "> 0"),
        ("V5", "Predictions in Cassandra",         "PASS","8,298 rows",        "> 0"),
        ("V6", "Alerts table has events",          "PASS","1 alert (M5.4)",    "> 0"),
        ("V7a","RF RMSE < 0.65",                   "PASS","0.1048",            "< 0.65"),
        ("V7b","RF R² > 0.70",                     "PASS","0.9929",            "> 0.70"),
        ("V7c","RF MAE < 0.45",                    "PASS","0.0666",            "< 0.45"),
        ("V8", "Model reload scoring",             "FAIL","Windows RDD issue", "Known"),
        ("V9a","SQL: Top networks",                "PASS","5 rows",            "> 0"),
        ("V9b","SQL: Magnitude distribution",      "PASS","6 rows",            "> 0"),
        ("V9c","SQL: Depth band analysis",         "PASS","3 rows",            "> 0"),
        ("V9d","SQL: Hourly pattern",              "PASS","5 rows",            "> 0"),
        ("V9e","SQL: Tsunami risk",                "PASS","1 row",             "> 0"),
        ("V10","Pipeline > 500 events",            "PASS","72,993",            "> 500"),
    ]

    for cid, name, status, value, criteria in checks:
        color = '#A8C8A8' if status=='PASS' else '#E8A0A0'
        icon  = '✅' if status=='PASS' else '❌'
        bg    = 'rgba(226,241,235,0.55)' if status=='PASS' else 'rgba(246,181,181,0.35)'
        st.markdown(f"""
        <div style="background:{bg};border:1px solid {color};border-left:5px solid {color};
                    border-radius:10px;padding:11px 18px;margin-bottom:7px;
                    display:flex;align-items:center;gap:14px;
                    backdrop-filter:blur(8px);">
            <span style="font-size:15px;">{icon}</span>
            <span style="color:{ACCENT};font-weight:700;min-width:45px;
                         font-family:'JetBrains Mono',monospace;font-size:12px;">{cid}</span>
            <span style="color:{TEXT_COLOR};flex:1;font-size:14px;">{name}</span>
            <span style="color:#7A7A9A;font-size:12px;min-width:150px;
                         font-family:'JetBrains Mono',monospace;">{value}</span>
            <span style="color:#A0A0B8;font-size:11px;">{criteria}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Final Project Metrics")
    metrics = [
        ("Kafka Messages","10,772"), ("Valid Earthquakes","72,993"),
        ("Cassandra Events","10,863"), ("ML Predictions","8,298"),
        ("Alerts","1+"), ("RF RMSE","0.1048"),
        ("RF R²","0.9929"), ("RF MAE","0.0666"),
        ("KMeans Silhouette","0.7282"), ("Anomalies","2,450"),
        ("Pass Rate","93.8%"), ("Project Rating","9.6 / 10"),
    ]
    cols = st.columns(4)
    for i,(k,v) in enumerate(metrics):
        with cols[i%4]: st.metric(k,v)

    st.markdown("---")
    st.markdown("### 🏗️ Full Pipeline Architecture")
    st.markdown("""
    | Stage | Tool | Status | Details |
    |-------|------|--------|---------|
    | Live data ingestion | USGS GeoJSON API | ✅ Active | Every 60 seconds |
    | Event streaming | Apache Kafka `seismic-events` | ✅ Active | 74,305+ messages |
    | Alert streaming | Apache Kafka `seismic-alerts` | ✅ Active | M4.5+ events |
    | Waveform streaming | Apache Kafka `seismic-waveforms` | ✅ Active | 5 IRIS stations |
    | Batch ETL | Spark DataFrame API | ✅ Complete | 9 transforms, 8 actions |
    | NoSQL storage | Cassandra (4 tables) | ✅ Complete | 10,863 events |
    | Waveform storage | HBase `seismic_waveforms` | ✅ Designed | Schema documented |
    | Structured streaming | Spark Streaming | ✅ Complete | 30s micro-batches |
    | ML Regression | RandomForest (MLlib) | ✅ Complete | RMSE=0.10, R²=0.99 |
    | ML Clustering | KMeans (MLlib) | ✅ Complete | 8 tectonic zones |
    | SQL Analytics | Spark SQL | ✅ Complete | 7 queries |
    | Dashboard | Streamlit + Plotly | ✅ Running | This page |
    """)