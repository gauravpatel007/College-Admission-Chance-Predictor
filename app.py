# to run this app: streamlit run app.py
"""
Gujarat Engineering Admission Predictor - Streamlit App (Enhanced)
====================================================================
A web app where a student inputs their GUJCET score, 12th marks,
category, preferred branch, and college - and gets a predicted
admission probability with insights.

Based on real ACPC cutoff data scraped from acpc.gujarat.gov.in

FEATURES:
  Tab 1 - Predict My Chances  (gauge, radar chart, smart tips)
  Tab 2 - What-If Simulator   (interactive scenario builder)
  Tab 3 - Branch Recommender  (best branches for your rank)
  Tab 4 - College Comparison   (bar charts, head-to-head)
  Tab 5 - College Rankings     (tier heatmap, leaderboard)
  Tab 6 - Cutoff Trends        (year-over-year analytics)
  Tab 7 - Model Info & SHAP    (explainability)
"""

import sys
import io
import os
import pickle
import warnings
import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
MODEL_PATH = ROOT_DIR / "model.pkl"
ENCODERS_PATH = ROOT_DIR / "encoders.pkl"
SHAP_PATH = ROOT_DIR / "shap_data.pkl"

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gujarat Engineering Admission Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    /* ─── Header ─── */
    .main-header {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 40%, #2c5364 70%, #4ecdc4 100%);
        padding: 2.2rem 2rem;
        border-radius: 18px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.25);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute; top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.04) 0%, transparent 70%);
        animation: shimmer 8s ease-in-out infinite;
    }
    @keyframes shimmer { 0%,100% { transform: rotate(0deg); } 50% { transform: rotate(180deg); } }
    .main-header h1 { color: white; font-size: 2.2rem; font-weight: 700; margin: 0; position: relative; }
    .main-header p  { color: rgba(255,255,255,0.85); font-size: 1rem; margin-top: 0.4rem; position: relative; }
    /* ─── Result cards ─── */
    .result-card {
        border-radius: 16px; padding: 2rem; text-align: center;
        box-shadow: 0 6px 24px rgba(0,0,0,0.1);
        border: 1px solid rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    .result-card:hover { transform: translateY(-2px); }
    .result-card-success { background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); border: 2px solid #28a745; }
    .result-card-warning { background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%); border: 2px solid #ffc107; }
    .result-card-danger  { background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); border: 2px solid #dc3545; }
    .probability-number { font-size: 3.5rem; font-weight: 700; line-height: 1; }
    .probability-label  { font-size: 1rem; color: #555; margin-top: 0.3rem; }
    /* ─── Stat pills ─── */
    .stat-pill {
        display: inline-flex; align-items: center; gap: 6px;
        background: linear-gradient(135deg, #2d6a9f, #4ecdc4);
        color: white; padding: 6px 16px; border-radius: 24px;
        font-size: 0.82rem; font-weight: 600; margin: 3px;
    }
    /* ─── Metric mini-card ─── */
    .mini-card {
        background: white; border-radius: 12px; padding: 1rem;
        box-shadow: 0 2px 14px rgba(0,0,0,0.06);
        border-left: 4px solid #2d6a9f; margin-bottom: 0.6rem;
    }
    .mini-card .val { font-size: 1.4rem; font-weight: 700; color: #1e3a5f; }
    .mini-card .lbl { font-size: 0.78rem; color: #999; }
    /* ─── Tip box ─── */
    .tip-box {
        background: linear-gradient(135deg, #eef7ff, #e0f0ff);
        border-left: 4px solid #2d6a9f; border-radius: 10px;
        padding: 1rem 1.2rem; margin: 0.6rem 0;
        font-size: 0.92rem; color: #1a3d5c;
    }
    /* ─── Footer ─── */
    .footer {
        text-align: center; padding: 1.5rem; color: #888;
        font-size: 0.82rem; border-top: 1px solid #eee; margin-top: 2rem;
    }
    /* ─── Section header ─── */
    .section-header {
        background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
        color: white; padding: 0.7rem 1.2rem; border-radius: 10px;
        margin-bottom: 1rem; font-weight: 600; font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Load Model & Data ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)
    return model_data, encoders


@st.cache_data
def load_cutoff_data():
    cutoff_df = pd.read_csv(DATA_DIR / "acpc_cutoffs.csv")
    student_df = pd.read_csv(DATA_DIR / "student_data.csv")
    return cutoff_df, student_df


# ── Helper Functions ──────────────────────────────────────────────────────────
def classify_tier(college_name):
    name_lower = str(college_name).lower()
    tier1 = ["daiict", "nirma", "pdeu", "svnit", "ldce", "l.d.", "dhirubhai", "iit", "nit", "iiit"]
    tier2 = ["bvm", "vgec", "gcet", "msu", "marwadi", "silver oak", "charotar", "birla", "vishwakarma", "institute of tech"]
    if any(kw in name_lower for kw in tier1): return 1
    if any(kw in name_lower for kw in tier2): return 2
    return 3


def _predict_single(model_data, encoders, cutoff_df,
                     gujcet, pcm, category, branch, college, rank):
    """Core prediction logic — returns (probability, avg_closing)."""
    model = model_data["model"]
    le_cat = encoders["category"]
    le_br  = encoders["branch"]

    try: cat_enc = le_cat.transform([category])[0]
    except ValueError: cat_enc = 0
    try: br_enc = le_br.transform([branch])[0]
    except ValueError: br_enc = 0

    tier = classify_tier(college)
    mask = cutoff_df["branch"].str.lower().str.contains(branch.split()[0].lower(), na=False)
    avg_close = cutoff_df.loc[mask, "closing_rank"].median() if mask.any() else cutoff_df["closing_rank"].median()
    ratio = rank / max(1, avg_close)

    X = pd.DataFrame([{
        "gujcet_score": gujcet,
        "twelfth_pcm_percent": pcm,
        "category_encoded": cat_enc,
        "branch_encoded": br_enc,
        "college_tier": tier,
        "closing_rank": avg_close,
        "rank_to_cutoff_ratio": ratio,
        "year_num": 2024,
    }])
    prob = model.predict_proba(X)[0][1]
    return prob, avg_close


def get_alternatives(cutoff_df, student_rank, branch, top_n=8):
    mask = cutoff_df["branch"].str.lower().str.contains(branch.split()[0].lower(), na=False)
    df = cutoff_df[mask].copy() if mask.any() else cutoff_df.copy()
    latest = df["year"].max()
    df = df[df["year"] == latest]
    df["margin"] = df["closing_rank"] - student_rank
    df = df.sort_values("margin", ascending=False)
    return df.head(top_n)[["college", "branch", "closing_rank", "margin"]]


def rank_percentile(student_rank, cutoff_df):
    """What percentile is this rank among all closing ranks?"""
    all_ranks = cutoff_df["closing_rank"].dropna()
    pct = (all_ranks >= student_rank).mean() * 100
    return round(pct, 1)


def generate_report_csv(prob, gujcet, pcm, rank, category, branch, college, alts):
    """Generate a downloadable CSV report."""
    rows = [
        ["Gujarat Engineering Admission Report", ""],
        ["Generated", datetime.datetime.now().strftime("%Y-%m-%d %H:%M")],
        ["", ""],
        ["--- Student Profile ---", ""],
        ["GUJCET Score", gujcet],
        ["12th PCM %", pcm],
        ["Merit Rank", rank],
        ["Category", category],
        ["Branch", branch],
        ["College", college],
        ["", ""],
        ["--- Prediction ---", ""],
        ["Admission Probability", f"{prob*100:.1f}%"],
        ["Status", "Strong" if prob > 0.7 else ("Moderate" if prob > 0.4 else "Low")],
        ["", ""],
        ["--- Alternative Colleges ---", ""],
        ["College", "Closing Rank"],
    ]
    for _, r in alts.iterrows():
        rows.append([r["college"], int(r["closing_rank"])])
    df = pd.DataFrame(rows, columns=["Field", "Value"])
    return df.to_csv(index=False)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
def main():
    try:
        model_data, encoders = load_model()
        cutoff_df, student_df = load_cutoff_data()
    except Exception as e:
        st.error(f"Error loading model/data: {e}")
        st.info("Run `python notebooks/model_training.py` first.")
        return

    branches   = sorted(student_df["branch"].dropna().unique())
    colleges   = sorted(student_df["college"].dropna().unique())
    categories = ["General", "OBC", "SC", "ST", "EWS"]

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1>🎓 Gujarat Engineering Admission Predictor</h1>
        <p>AI-powered predictions based on 5 years of real ACPC cutoff data</p>
        <div style="position:relative;margin-top:0.6rem;">
            <span class="stat-pill">📄 1,498 cutoff records</span>
            <span class="stat-pill">🏛️ 323 colleges</span>
            <span class="stat-pill">📅 2020-2025</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 📝 Enter Your Details")
        st.markdown("---")
        gujcet = st.slider("🎯 GUJCET Score", 0, 120, 80)
        twelfth = st.slider("📚 12th PCM %", 35.0, 99.9, 75.0, 0.1)
        student_rank = st.number_input("🏅 Merit Rank", 1, 100000, 5000, step=100)
        category = st.selectbox("📋 Category", categories)
        branch = st.selectbox(
            "🔧 Preferred Branch", branches,
            index=branches.index("Computer Engineering") if "Computer Engineering" in branches else 0,
        )
        college = st.selectbox("🏛️ Preferred College", colleges)
        st.markdown("---")
        predict_btn = st.button("🔮 Predict My Chances", type="primary", use_container_width=True)

        # Rank Percentile (always visible)
        pct = rank_percentile(student_rank, cutoff_df)
        st.markdown("---")
        st.markdown(f"""
        <div class="mini-card">
            <div class="lbl">Your Rank Percentile</div>
            <div class="val">{pct}%</div>
            <div class="lbl">of colleges have a closing rank ≥ your rank</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🎯 Prediction",
        "🔄 What-If Simulator",
        "💡 Branch Recommender",
        "🏛️ College Comparison",
        "🏆 College Rankings",
        "📊 Cutoff Trends",
        "🧠 Model & SHAP",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — PREDICTION
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        if predict_btn:
            prob, avg_closing = _predict_single(
                model_data, encoders, cutoff_df,
                gujcet, twelfth, category, branch, college, student_rank
            )
            if prob > 0.7:
                cls, emoji, txt, clr = "result-card-success", "✅", "Strong Chance!", "#28a745"
            elif prob > 0.4:
                cls, emoji, txt, clr = "result-card-warning", "⚠️", "Moderate Chance", "#e6a817"
            else:
                cls, emoji, txt, clr = "result-card-danger", "❌", "Low Chance", "#dc3545"

            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown(f"""
                <div class="result-card {cls}">
                    <div class="probability-number" style="color:{clr}">{prob*100:.1f}%</div>
                    <div class="probability-label">Admission Probability</div>
                    <div style="font-size:1.4rem;margin-top:.4rem">{emoji} {txt}</div>
                </div>""", unsafe_allow_html=True)

                # Gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=prob*100,
                    number={"suffix": "%", "font": {"size": 30}},
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": clr},
                           "steps": [{"range": [0,40], "color": "#f8d7da"},
                                     {"range": [40,70], "color": "#fff3cd"},
                                     {"range": [70,100], "color": "#d4edda"}]}
                ))
                fig.update_layout(height=260, margin=dict(l=20,r=20,t=30,b=10))
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.markdown("### 📊 Profile Summary")
                mc = st.columns(2)
                for col_obj, label, val in [
                    (mc[0], "GUJCET", f"{gujcet}/120"), (mc[1], "12th PCM", f"{twelfth}%"),
                    (mc[0], "Rank", f"#{student_rank:,}"), (mc[1], "Category", category),
                    (mc[0], "Tier", f"Tier {classify_tier(college)}"), (mc[1], "Percentile", f"{pct}%"),
                ]:
                    col_obj.markdown(f'<div class="mini-card"><div class="lbl">{label}</div><div class="val">{val}</div></div>', unsafe_allow_html=True)

                st.markdown(f"**Branch:** {branch}")
                st.markdown(f"**College:** {college[:60]}{'...' if len(college)>60 else ''}")
                st.markdown(f"**Avg Closing Rank:** #{int(avg_closing):,}")

                # Radar chart — profile strength
                norms = {
                    "GUJCET": gujcet/120*100,
                    "12th %": twelfth,
                    "Rank Score": max(0, 100 - student_rank/600),
                    "Category\nBoost": {"General":30,"EWS":50,"OBC":60,"SC":80,"ST":90}.get(category,40),
                    "Tier Match": {1:90,2:60,3:40}[classify_tier(college)],
                }
                fig_r = go.Figure(go.Scatterpolar(
                    r=list(norms.values()), theta=list(norms.keys()),
                    fill="toself", marker_color="#2d6a9f",
                    fillcolor="rgba(45,106,159,0.25)",
                ))
                fig_r.update_layout(
                    polar=dict(radialaxis=dict(range=[0,100], showticklabels=False)),
                    height=280, margin=dict(l=40,r=40,t=30,b=30),
                    title="Profile Strength Radar",
                )
                st.plotly_chart(fig_r, use_container_width=True)

            # ── Smart Tips ──
            st.markdown('<div class="section-header">💡 Smart Tips</div>', unsafe_allow_html=True)
            tips = []
            if prob < 0.5:
                tips.append("Your rank is above the median cutoff for this branch. Consider Tier 2/3 colleges or less competitive branches.")
            if gujcet < 60:
                tips.append("A GUJCET score below 60 limits your options. Focus on colleges where cutoff is rank-based rather than score-based.")
            if twelfth < 65:
                tips.append("12th percentage below 65% may restrict certain NRI/management seats. Strengthen your rank to compensate.")
            if classify_tier(college) == 1 and prob < 0.6:
                tips.append("Tier 1 colleges are extremely competitive. Consider applying to Tier 2 colleges as safe options.")
            if category in ["SC", "ST"]:
                tips.append("Reserved category seats typically have 1.5–2.2× higher closing ranks — leverage this advantage.")
            if not tips:
                tips.append("Your profile looks strong! Keep this college as your top choice and apply early.")
            for t in tips:
                st.markdown(f'<div class="tip-box">{t}</div>', unsafe_allow_html=True)

            # ── Alternatives ──
            st.markdown('<div class="section-header">🏆 Best Alternatives For You</div>', unsafe_allow_html=True)
            alts = get_alternatives(cutoff_df, student_rank, branch)
            if not alts.empty:
                alts_disp = alts.copy()
                alts_disp["Status"] = alts_disp["margin"].apply(
                    lambda x: "✅ Strong" if x > 2000 else ("🟡 Possible" if x > 0 else "🔴 Difficult"))
                alts_disp = alts_disp.rename(columns={"college":"College","branch":"Branch","closing_rank":"Closing Rank"})
                st.dataframe(alts_disp[["College","Branch","Closing Rank","Status"]], hide_index=True, use_container_width=True)

            # ── Download ──
            csv = generate_report_csv(prob, gujcet, twelfth, student_rank, category, branch, college, alts)
            st.download_button("📥 Download Admission Report (CSV)", csv, "admission_report.csv", "text/csv",
                               use_container_width=True)

        else:
            st.markdown("### 👈 Enter your details in the sidebar and click **Predict My Chances**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("📊 Records", f"{len(cutoff_df):,}")
            c2.metric("🏛️ Colleges", cutoff_df['college'].nunique())
            c3.metric("🔧 Branches", cutoff_df['branch'].nunique())
            c4.metric("📅 Years", "2020–2025")

            results = model_data.get("results", {})
            if results:
                st.markdown("### 🤖 Model Performance")
                perf = pd.DataFrame([
                    {"Model": n, "Accuracy": f"{r['accuracy']:.2%}", "AUC-ROC": f"{r['auc']:.4f}",
                     "CV Mean": f"{r['cv_mean']:.4f}"}
                    for n, r in results.items()
                ])
                st.dataframe(perf, hide_index=True, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — WHAT-IF SIMULATOR
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown('<div class="section-header">🔄 What-If Scenario Simulator</div>', unsafe_allow_html=True)
        st.markdown("See how changes to your profile affect your admission probability **in real-time**.")

        sim_col1, sim_col2 = st.columns([1, 1])

        with sim_col1:
            st.markdown("#### Adjust Your Scenario")
            sim_gujcet = st.slider("GUJCET Score", 0, 120, gujcet, key="sim_gujcet")
            sim_pcm    = st.slider("12th PCM %", 35.0, 99.9, twelfth, 0.5, key="sim_pcm")
            sim_rank   = st.slider("Merit Rank", 100, 80000, student_rank, 100, key="sim_rank")
            sim_cat    = st.selectbox("Category", categories, index=categories.index(category), key="sim_cat")
            sim_branch = st.selectbox("Branch", branches,
                                       index=branches.index(branch) if branch in branches else 0,
                                       key="sim_br")

        with sim_col2:
            st.markdown("#### Probability Comparison")

            # Original prediction
            prob_orig, _ = _predict_single(model_data, encoders, cutoff_df,
                                           gujcet, twelfth, category, branch, college, student_rank)
            # Simulated prediction
            prob_sim, _ = _predict_single(model_data, encoders, cutoff_df,
                                          sim_gujcet, sim_pcm, sim_cat, sim_branch, college, sim_rank)

            delta = prob_sim - prob_orig
            delta_str = f"+{delta*100:.1f}%" if delta >= 0 else f"{delta*100:.1f}%"
            delta_color = "#28a745" if delta >= 0 else "#dc3545"

            st.markdown(f"""
            <div style="display:flex;gap:1rem;margin-bottom:1rem;">
                <div class="mini-card" style="flex:1;text-align:center;">
                    <div class="lbl">Original</div>
                    <div class="val">{prob_orig*100:.1f}%</div>
                </div>
                <div class="mini-card" style="flex:1;text-align:center;">
                    <div class="lbl">Simulated</div>
                    <div class="val" style="color:{delta_color}">{prob_sim*100:.1f}%</div>
                </div>
                <div class="mini-card" style="flex:1;text-align:center;">
                    <div class="lbl">Change</div>
                    <div class="val" style="color:{delta_color}">{delta_str}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Comparison bar chart
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Original", x=["Probability"], y=[prob_orig*100],
                                  marker_color="#6c757d", text=[f"{prob_orig*100:.1f}%"], textposition="outside"))
            fig.add_trace(go.Bar(name="Simulated", x=["Probability"], y=[prob_sim*100],
                                  marker_color="#2d6a9f", text=[f"{prob_sim*100:.1f}%"], textposition="outside"))
            fig.update_layout(barmode="group", height=300, yaxis_range=[0, 110],
                              margin=dict(l=20,r=20,t=30,b=20), showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

        # Multi-rank sweep chart
        st.markdown("#### 📈 Probability vs Rank Sweep")
        sweep_ranks = list(range(500, 50001, 500))
        sweep_probs = []
        for r in sweep_ranks:
            p, _ = _predict_single(model_data, encoders, cutoff_df,
                                    sim_gujcet, sim_pcm, sim_cat, sim_branch, college, r)
            sweep_probs.append(p * 100)
        fig_sweep = go.Figure()
        fig_sweep.add_trace(go.Scatter(x=sweep_ranks, y=sweep_probs, mode="lines",
                                        fill="tozeroy", fillcolor="rgba(78,205,196,0.15)",
                                        line=dict(color="#2d6a9f", width=3), name="Probability"))
        fig_sweep.add_vline(x=sim_rank, line_dash="dash", line_color="#dc3545",
                             annotation_text=f"Your Rank: {sim_rank}")
        fig_sweep.update_layout(
            xaxis_title="Merit Rank", yaxis_title="Admission Probability (%)",
            height=400, margin=dict(l=20,r=20,t=30,b=20),
            yaxis_range=[0, 105],
        )
        st.plotly_chart(fig_sweep, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — BRANCH RECOMMENDER
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown('<div class="section-header">💡 Branch Recommender — Best Branches For Your Rank</div>',
                    unsafe_allow_html=True)
        st.markdown(f"Based on your **Merit Rank #{student_rank:,}**, here are the branches ranked by your admission probability.")

        branch_probs = []
        for br in branches:
            p, ac = _predict_single(model_data, encoders, cutoff_df,
                                     gujcet, twelfth, category, br, college, student_rank)
            branch_probs.append({"Branch": br, "Probability": p*100, "Avg Closing Rank": int(ac)})
        bp_df = pd.DataFrame(branch_probs).sort_values("Probability", ascending=False).reset_index(drop=True)
        bp_df["Rank"] = range(1, len(bp_df)+1)
        bp_df["Status"] = bp_df["Probability"].apply(
            lambda x: "✅ Strong" if x > 70 else ("🟡 Moderate" if x > 40 else "🔴 Difficult"))

        # Horizontal bar chart
        fig = px.bar(bp_df.head(15), x="Probability", y="Branch", orientation="h",
                     color="Probability", color_continuous_scale="RdYlGn",
                     title=f"Top 15 Branches by Admission Probability (Rank #{student_rank:,})",
                     text="Probability")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(height=500, yaxis=dict(autorange="reversed"),
                          margin=dict(l=20,r=20,t=50,b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Table
        st.dataframe(bp_df[["Rank","Branch","Probability","Avg Closing Rank","Status"]].head(15),
                     hide_index=True, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — COLLEGE COMPARISON
    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown('<div class="section-header">🏛️ College Cutoff Comparison</div>', unsafe_allow_html=True)

        sel_branch = st.selectbox("Select Branch", cutoff_df["branch"].dropna().unique(), key="cmp_br")
        b_data = cutoff_df[cutoff_df["branch"].str.lower().str.contains(sel_branch.split()[0].lower(), na=False)]

        if not b_data.empty:
            latest_yr = b_data["year"].max()
            latest = b_data[b_data["year"] == latest_yr]
            top15 = latest.nsmallest(15, "closing_rank")

            fig = px.bar(top15, x="college", y="closing_rank", color="closing_rank",
                         color_continuous_scale="RdYlGn_r",
                         title=f"Top 15 Most Competitive — {sel_branch} ({latest_yr})")
            fig.update_layout(xaxis_tickangle=-45, height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Lowest Cutoff", f"#{int(latest['closing_rank'].min()):,}")
            sc2.metric("Median Cutoff", f"#{int(latest['closing_rank'].median()):,}")
            sc3.metric("Highest Cutoff", f"#{int(latest['closing_rank'].max()):,}")

        # ── Head-to-Head ──
        st.markdown('<div class="section-header">⚔️ Head-to-Head College Comparison</div>', unsafe_allow_html=True)
        h2h_cols = st.columns(2)
        with h2h_cols[0]:
            college_a = st.selectbox("College A", colleges, key="h2h_a")
        with h2h_cols[1]:
            college_b = st.selectbox("College B", colleges, index=min(1, len(colleges)-1), key="h2h_b")

        if college_a and college_b:
            data_a = cutoff_df[cutoff_df["college"] == college_a]
            data_b = cutoff_df[cutoff_df["college"] == college_b]

            ma = {"College": college_a, "Tier": classify_tier(college_a),
                  "Branches": data_a["branch"].nunique(),
                  "Avg Closing Rank": int(data_a["closing_rank"].mean()) if not data_a.empty else "N/A",
                  "Min Closing Rank": int(data_a["closing_rank"].min()) if not data_a.empty else "N/A",
                  "Years of Data": data_a["year"].nunique() if not data_a.empty else 0}
            mb = {"College": college_b, "Tier": classify_tier(college_b),
                  "Branches": data_b["branch"].nunique(),
                  "Avg Closing Rank": int(data_b["closing_rank"].mean()) if not data_b.empty else "N/A",
                  "Min Closing Rank": int(data_b["closing_rank"].min()) if not data_b.empty else "N/A",
                  "Years of Data": data_b["year"].nunique() if not data_b.empty else 0}

            cmp_df = pd.DataFrame([ma, mb]).set_index("College").T
            st.dataframe(cmp_df, use_container_width=True)

            # Both probabilities
            prob_a, _ = _predict_single(model_data, encoders, cutoff_df,
                                        gujcet, twelfth, category, branch, college_a, student_rank)
            prob_b, _ = _predict_single(model_data, encoders, cutoff_df,
                                        gujcet, twelfth, category, branch, college_b, student_rank)

            fig_h2h = go.Figure()
            fig_h2h.add_trace(go.Bar(name=college_a[:30], x=["Admission Prob"], y=[prob_a*100],
                                      marker_color="#2d6a9f", text=[f"{prob_a*100:.1f}%"], textposition="outside"))
            fig_h2h.add_trace(go.Bar(name=college_b[:30], x=["Admission Prob"], y=[prob_b*100],
                                      marker_color="#4ecdc4", text=[f"{prob_b*100:.1f}%"], textposition="outside"))
            fig_h2h.update_layout(barmode="group", height=300, yaxis_range=[0,110])
            st.plotly_chart(fig_h2h, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5 — COLLEGE RANKINGS
    # ══════════════════════════════════════════════════════════════════════════
    with tab5:
        st.markdown('<div class="section-header">🏆 College Rankings & Tier Analysis</div>', unsafe_allow_html=True)

        cutoff_df["tier"] = cutoff_df["college"].apply(classify_tier)
        latest_yr = cutoff_df["year"].max()
        latest_all = cutoff_df[cutoff_df["year"] == latest_yr]

        # Tier distribution pie
        tier_counts = latest_all.groupby("tier")["college"].nunique().reset_index()
        tier_counts.columns = ["Tier", "Count"]
        tier_counts["Label"] = tier_counts["Tier"].map({1: "Tier 1 (Elite)", 2: "Tier 2 (Good)", 3: "Tier 3 (Other)"})
        fig_pie = px.pie(tier_counts, values="Count", names="Label",
                         color_discrete_sequence=["#1e3a5f", "#2d6a9f", "#b0c4de"],
                         title="College Distribution by Tier", hole=0.45)
        fig_pie.update_layout(height=350)

        # Most competitive colleges
        college_rank = latest_all.groupby("college").agg(
            min_cutoff=("closing_rank", "min"),
            avg_cutoff=("closing_rank", "mean"),
            branches=("branch", "nunique"),
        ).reset_index().sort_values("min_cutoff").head(20)
        college_rank["Tier"] = college_rank["college"].apply(classify_tier)
        college_rank["Rank"] = range(1, len(college_rank)+1)

        cr1, cr2 = st.columns([1, 1])
        with cr1:
            st.plotly_chart(fig_pie, use_container_width=True)
        with cr2:
            st.markdown("#### 🥇 Top 20 Most Competitive Colleges")
            st.dataframe(
                college_rank[["Rank", "college", "Tier", "min_cutoff", "avg_cutoff", "branches"]].rename(
                    columns={"college": "College", "min_cutoff": "Best Cutoff",
                             "avg_cutoff": "Avg Cutoff", "branches": "Branches"}
                ), hide_index=True, use_container_width=True, height=400
            )

        # Heatmap — branch x tier
        st.markdown("#### 🗺️ Branch × Tier Heatmap (Median Closing Rank)")
        hm_data = latest_all.groupby(["branch", "tier"])["closing_rank"].median().reset_index()
        hm_pivot = hm_data.pivot(index="branch", columns="tier", values="closing_rank")
        hm_pivot.columns = [f"Tier {c}" for c in hm_pivot.columns]
        top_hm = hm_pivot.dropna().sort_values("Tier 1" if "Tier 1" in hm_pivot.columns else hm_pivot.columns[0]).head(15)

        if not top_hm.empty:
            fig_hm = px.imshow(
                top_hm.values, x=top_hm.columns.tolist(), y=top_hm.index.tolist(),
                color_continuous_scale="RdYlGn_r", aspect="auto",
                title="Median Closing Rank by Branch × Tier (lower = harder)",
                labels=dict(color="Closing Rank"),
            )
            fig_hm.update_layout(height=500)
            st.plotly_chart(fig_hm, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 6 — CUTOFF TRENDS
    # ══════════════════════════════════════════════════════════════════════════
    with tab6:
        st.markdown('<div class="section-header">📊 Cutoff Trends (2020-2025)</div>', unsafe_allow_html=True)

        trend = cutoff_df.copy()
        branch_avg = trend.groupby(["year", "branch"]).agg(med=("closing_rank", "median")).reset_index()
        top_br = branch_avg.groupby("branch")["med"].mean().nsmallest(10).index.tolist()
        filt = branch_avg[branch_avg["branch"].isin(top_br)]

        fig = px.line(filt, x="year", y="med", color="branch", markers=True,
                      title="Branch-wise Median Cutoff Trends",
                      labels={"med": "Median Closing Rank", "year": "Year", "branch": "Branch"})
        fig.update_layout(height=500, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # Category distribution
        st.markdown("#### Category-wise Admission Rate")
        cat_data = student_df.groupby("category").agg(
            admitted=("admitted", "sum"), total=("admitted", "count")).reset_index()
        cat_data["rate"] = cat_data["admitted"] / cat_data["total"]
        fig2 = px.bar(cat_data, x="category", y="rate", color="category",
                      color_discrete_sequence=px.colors.qualitative.Set2,
                      title="Admission Rate by Category")
        fig2.update_yaxes(tickformat=".0%")
        fig2.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

        # Year stats
        st.markdown("#### Year-over-Year Competitiveness")
        ys = cutoff_df.groupby("year").agg(
            avg=("closing_rank", "mean"), mn=("closing_rank", "min"),
            mx=("closing_rank", "max"), n_col=("college", "nunique")).reset_index()
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=ys["year"], y=ys["avg"], mode="lines+markers",
                                   name="Average", line=dict(color="#2d6a9f", width=3)))
        fig3.add_trace(go.Scatter(x=ys["year"], y=ys["mn"], mode="lines+markers",
                                   name="Most Competitive", line=dict(color="#dc3545", width=2, dash="dash")))
        fig3.update_layout(title="Overall Trend", height=400, xaxis_title="Year", yaxis_title="Closing Rank")
        st.plotly_chart(fig3, use_container_width=True)

        # Sunburst — drill-down
        st.markdown("#### 🌐 Sunburst: Tier → Branch breakdown")
        sun_data = cutoff_df.copy()
        sun_data["tier_label"] = sun_data["college"].apply(
            lambda x: {1: "Tier 1", 2: "Tier 2", 3: "Tier 3"}[classify_tier(x)])
        sun_agg = sun_data.groupby(["tier_label", "branch"]).size().reset_index(name="count")
        sun_agg = sun_agg[sun_agg["count"] >= 3]
        fig_sun = px.sunburst(sun_agg, path=["tier_label", "branch"], values="count",
                               color="count", color_continuous_scale="blues",
                               title="College Tier → Branch Data Volume")
        fig_sun.update_layout(height=550)
        st.plotly_chart(fig_sun, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 7 — MODEL & SHAP
    # ══════════════════════════════════════════════════════════════════════════
    with tab7:
        st.markdown('<div class="section-header">🧠 Model & SHAP Explainability</div>', unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        with m1:
            st.markdown("#### Model Details")
            model_name = model_data.get("model_name", "Unknown")
            st.markdown(f"- **Best Model:** {model_name}")
            st.markdown(f"- **Training Records:** {len(student_df):,}")
            st.markdown(f"- **Features:** {len(encoders.get('feature_cols', []))}")
            st.markdown(f"- **Source:** acpc.gujarat.gov.in (scraped)")

            results = model_data.get("results", {})
            if results:
                st.markdown("#### All Models")
                for n, r in results.items():
                    with st.expander(n):
                        st.write(f"Accuracy: {r['accuracy']:.4f}")
                        st.write(f"AUC-ROC: {r['auc']:.4f}")
                        st.write(f"CV: {r['cv_mean']:.4f} ± {r['cv_std']:.4f}")

        with m2:
            st.markdown("#### Feature Importance")
            if hasattr(model_data["model"], "feature_importances_"):
                imps = model_data["model"].feature_importances_
                fdf = pd.DataFrame({"Feature": encoders.get("feature_cols",[]),
                                     "Importance": imps}).sort_values("Importance", ascending=True)
                fig = px.bar(fdf, x="Importance", y="Feature", orientation="h",
                             color="Importance", color_continuous_scale="viridis")
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        # SHAP
        st.markdown("#### 🔍 SHAP Explainability")
        if SHAP_PATH.exists():
            try:
                with open(SHAP_PATH, "rb") as f:
                    shap_data = pickle.load(f)
                shap_vals = shap_data["shap_values"]
                if isinstance(shap_vals, list): shap_vals = shap_vals[1]
                mean_shap = np.abs(shap_vals).mean(axis=0)
                sdf = pd.DataFrame({"Feature": shap_data["feature_names"],
                                     "Mean |SHAP|": mean_shap}).sort_values("Mean |SHAP|", ascending=True)
                fig = px.bar(sdf, x="Mean |SHAP|", y="Feature", orientation="h",
                             color="Mean |SHAP|", color_continuous_scale="reds",
                             title="SHAP — Feature Impact on Admission")
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                st.info("SHAP shows **which factors** pushed the prediction toward admission (positive) "
                        "or rejection (negative). Higher bars = more influential features.")
            except Exception as e:
                st.warning(f"SHAP unavailable: {e}")
        else:
            st.info("SHAP data not found. Run model training to generate.")

        # Dataset summary
        st.markdown("#### 📋 Dataset Summary")
        st.markdown(f"""
        | Metric | Value |
        |---|---|
        | Cutoff Records | {len(cutoff_df):,} |
        | Training Records | {len(student_df):,} |
        | Colleges | {cutoff_df['college'].nunique()} |
        | Branches | {cutoff_df['branch'].nunique()} |
        | Years | {cutoff_df['year'].min()} — {cutoff_df['year'].max()} |
        | Source | ACPC Gujarat (scraped) |
        """)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="footer">
        🎓 Gujarat Engineering Admission Predictor | Built with Streamlit + scikit-learn + XGBoost<br>
        Data scraped from <a href="https://acpc.gujarat.gov.in">acpc.gujarat.gov.in</a> |
        Real ACPC cutoff data 2020-2025
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
