import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="World Football Intelligence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    /* ── Metric cards ── */
    .metric-card {
        border-radius: 12px;
        padding: 18px 20px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        height: 90px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        background: rgba(128,128,128,0.06);
    }
    .metric-value {
        font-size: 26px;
        font-weight: 800;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 11px;
        opacity: 0.6;
        margin-top: 4px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* ── Insight boxes ── */
    .insight-box {
        background: rgba(245, 158, 11, 0.1);
        border-left: 4px solid #f59e0b;
        padding: 14px 16px;
        border-radius: 8px;
        margin: 6px 0;
        font-size: 13.5px;
        line-height: 1.6;
    }

    /* ── Prediction cards ── */
    .prediction-card {
        border-radius: 14px;
        padding: 22px 16px;
        border: 1px solid rgba(128,128,128,0.2);
        margin: 8px 0;
        text-align: center;
        background: rgba(128,128,128,0.05);
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] .stRadio label {
        padding: 5px 0;
        font-size: 14px;
    }

    /* ── Misc ── */
    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        padding: 10px 36px;
        font-size: 15px;
    }
    hr { border-color: rgba(128,128,128,0.2) !important; margin: 1.2rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    matches    = pd.read_csv("clean_matches.csv",        parse_dates=["date"])
    elo        = pd.read_csv("elo_matches.csv",          parse_dates=["date"])
    ratings    = pd.read_csv("elo_final_ratings.csv")
    history    = pd.read_csv("elo_history.csv",          parse_dates=["date"])
    importance = pd.read_csv("ml_feature_importance.csv")

    try:
        predictions = pd.read_csv("ml_predictions_2026.csv", parse_dates=["date"])
    except Exception:
        predictions = pd.DataFrame()

    try:
        ml_data = pd.read_csv("ml_dataset.csv", parse_dates=["date"])
    except Exception:
        ml_data = pd.DataFrame()

    if "rank" in ratings.columns:
        ratings = ratings.sort_values("rank").reset_index(drop=True)

    return matches, elo, ratings, history, importance, predictions, ml_data


matches, elo, ratings, history, importance, predictions, ml_data = load_data()
ALL_TEAMS = sorted(pd.concat([matches["home_team"], matches["away_team"]]).unique().tolist())


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def metric_card(value, label, accent="#1f77b4"):
    return f"""
    <div class="metric-card" style="border-left-color:{accent};">
        <div class="metric-value" style="color:{accent};">{value}</div>
        <div class="metric-label">{label}</div>
    </div>"""


# Shared Plotly layout — transparent, no hardcoded text colors
PLOTLY_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="sans-serif", size=12),
    margin=dict(t=40, b=20, l=10, r=10),
)

GRID_COLOR  = "rgba(128,128,128,0.15)"
BAR_COLORS  = {"Win": "#2563eb", "Draw": "#6b7280", "Loss": "#dc2626"}
LINE_COLORS = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2"]


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:10px 0 18px;">
        <span style="font-size:34px;">⚽</span>
        <div>
            <div style="font-size:15px;font-weight:800;color:#f1f5f9;line-height:1.2;">World Football</div>
            <div style="font-size:11px;color:#94a3b8;letter-spacing:0.05em;">INTELLIGENCE PLATFORM</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "🏠 Overview",
            "🔍 Team Explorer",
            "⚔️  Head-to-Head",
            "📈 Elo Rankings",
            "🤖 Match Predictor",
            "🏆 2026 World Cup",
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:12px;color:#94a3b8;line-height:1.8;">
        <b style="color:#cbd5e1;">About</b><br>
        49,215 international matches · 1872–2026<br><br>
        <b style="color:#cbd5e1;">Stack</b><br>
        pandas · SQLite · XGBoost<br>
        Custom Elo Engine · Streamlit
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ═══════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<script>window.scrollTo(0, 0);</script>', unsafe_allow_html=True)

    st.markdown('<script>window.scrollTo(0, 0);</script>', unsafe_allow_html=True)
    st.title("🌍 World Football Intelligence Platform")
    st.markdown("*149 years of international football — analysed, modelled, and visualised*")
    st.markdown("---")

    total_goals = int(matches["total_goals"].sum())
    avg_goals   = matches["total_goals"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(metric_card("49,215",          "Total Matches",     "#2563eb"), unsafe_allow_html=True)
    c2.markdown(metric_card(len(ALL_TEAMS),    "Unique Teams",      "#7c3aed"), unsafe_allow_html=True)
    c3.markdown(metric_card(f"{total_goals:,}","Total Goals",       "#16a34a"), unsafe_allow_html=True)
    c4.markdown(metric_card(matches["tournament"].nunique(), "Tournaments", "#d97706"), unsafe_allow_html=True)
    c5.markdown(metric_card(f"{avg_goals:.2f}","Avg Goals / Match", "#dc2626"), unsafe_allow_html=True)

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📊 Average Goals per Match by Decade")
        decade_goals = matches.groupby("decade")["total_goals"].mean().reset_index()
        decade_goals.columns = ["decade", "avg_goals"]
        fig = px.bar(
            decade_goals, x="decade", y="avg_goals",
            color="avg_goals", color_continuous_scale="Blues",
            labels={"decade": "Decade", "avg_goals": "Avg Goals"},
        )
        fig.update_traces(marker_line_width=0)
        fig.update_layout(**PLOTLY_LAYOUT, showlegend=False,
                          coloraxis_showscale=False, height=320, bargap=0.2)
        fig.update_xaxes(
            tickmode="array",
            tickvals=decade_goals["decade"].tolist(),
            ticktext=[str(d) for d in decade_goals["decade"].tolist()],
            tickangle=-45, gridcolor=GRID_COLOR
        )
        fig.update_yaxes(gridcolor=GRID_COLOR)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("🏠 Home Advantage Decay Over Time")
        home_only = matches[matches["neutral"] == False].copy()
        home_only["decade"] = (home_only["year"] // 10) * 10
        ha = home_only.groupby("decade").agg(
            home_wr=("result", lambda x: (x == "home_win").mean() * 100),
            away_wr=("result", lambda x: (x == "away_win").mean() * 100),
        ).reset_index()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=ha["decade"], y=ha["home_wr"], name="Home Win %",
            fill="tozeroy", line=dict(color="#2563eb", width=2.5),
            fillcolor="rgba(37,99,235,0.12)"
        ))
        fig2.add_trace(go.Scatter(
            x=ha["decade"], y=ha["away_wr"], name="Away Win %",
            fill="tozeroy", line=dict(color="#dc2626", width=2.5),
            fillcolor="rgba(220,38,38,0.12)"
        ))
        fig2.update_layout(**PLOTLY_LAYOUT, height=320,
                           yaxis_title="%", xaxis_title="Decade",
                           legend=dict(orientation="h", y=1.08, x=0))
        fig2.update_xaxes(gridcolor=GRID_COLOR)
        fig2.update_yaxes(gridcolor=GRID_COLOR)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📅 International Football Match Volume by Year")
    yearly_vol = matches.groupby(["year", "tournament_tier"]).size().reset_index(name="matches")
    tier_names = {1: "Major Finals", 2: "Qualifications", 3: "Regional Cups", 4: "Friendlies"}
    yearly_vol["tier_name"] = yearly_vol["tournament_tier"].map(tier_names)
    fig3 = px.area(
        yearly_vol, x="year", y="matches", color="tier_name",
        color_discrete_map={
            "Major Finals":   "#dc2626",
            "Qualifications": "#2563eb",
            "Regional Cups":  "#16a34a",
            "Friendlies":     "#93c5fd"
        }
    )
    fig3.update_layout(**PLOTLY_LAYOUT, height=300,
                       legend=dict(orientation="h", y=1.08, x=0, title=""))
    fig3.update_xaxes(gridcolor=GRID_COLOR)
    fig3.update_yaxes(gridcolor=GRID_COLOR)
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("💡 Key Findings")
    c_a, c_b, c_c = st.columns(3)
    with c_a:
        st.markdown('<div class="insight-box">🏠 <b>Home advantage has halved</b> — from ~35% gap in the 1890s to ~12% today</div>', unsafe_allow_html=True)
    with c_b:
        st.markdown('<div class="insight-box">⚽ <b>Goals peaked pre-1940</b> — tactical evolution in the 1960s reduced scoring significantly</div>', unsafe_allow_html=True)
    with c_c:
        st.markdown('<div class="insight-box">📈 <b>Matches grew 50×</b> — from ~5/year in the 1880s to 800+ per year today</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# PAGE 2: TEAM EXPLORER
# ═══════════════════════════════════════════════════════
elif page == "🔍 Team Explorer":
    st.markdown('<script>window.scrollTo(0, 0);</script>', unsafe_allow_html=True)

    st.title("🔍 Team Explorer")
    st.markdown("Deep-dive into any national team's full history, Elo trajectory, and performance stats.")

    selected_team = st.selectbox(
        "Select a team", ALL_TEAMS,
        index=ALL_TEAMS.index("Brazil") if "Brazil" in ALL_TEAMS else 0
    )

    team_matches = matches[
        (matches["home_team"] == selected_team) | (matches["away_team"] == selected_team)
    ].copy()

    team_matches["team_score"] = np.where(
        team_matches["home_team"] == selected_team,
        team_matches["home_score"], team_matches["away_score"]
    )
    team_matches["opp_score"] = np.where(
        team_matches["home_team"] == selected_team,
        team_matches["away_score"], team_matches["home_score"]
    )
    team_matches["opponent"] = np.where(
        team_matches["home_team"] == selected_team,
        team_matches["away_team"], team_matches["home_team"]
    )
    team_matches["team_result"] = np.where(
        team_matches["team_score"] > team_matches["opp_score"], "Win",
        np.where(team_matches["team_score"] == team_matches["opp_score"], "Draw", "Loss")
    )

    total  = len(team_matches)
    wins   = int((team_matches["team_result"] == "Win").sum())
    draws  = int((team_matches["team_result"] == "Draw").sum())
    losses = int((team_matches["team_result"] == "Loss").sum())
    elo_r  = ratings[ratings["team"] == selected_team]["elo_rating"].values
    rank_v = ratings[ratings["team"] == selected_team]["rank"].values

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Matches", f"{total:,}")
    c2.metric("Wins",   f"{wins:,}",   f"{wins/total*100:.1f}%" if total else "—")
    c3.metric("Draws",  f"{draws:,}",  f"{draws/total*100:.1f}%" if total else "—")
    c4.metric("Losses", f"{losses:,}", f"{losses/total*100:.1f}%" if total else "—")
    if len(elo_r):
        c5.metric("Elo Rating", f"{elo_r[0]:.0f}", f"Rank #{int(rank_v[0])}")

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📈 Elo Rating History")
        team_elo = history[history["team"] == selected_team].copy()
        if len(team_elo) > 0:
            team_elo["year"] = team_elo["date"].dt.year
            yearly_elo = team_elo.groupby("year")["elo_rating"].mean().reset_index()
            fig = px.line(
                yearly_elo, x="year", y="elo_rating",
                labels={"year": "Year", "elo_rating": "Elo Rating"},
                color_discrete_sequence=["#2563eb"]
            )
            fig.update_traces(line_width=2.5)
            fig.update_layout(**PLOTLY_LAYOUT, height=300)
            fig.update_xaxes(gridcolor=GRID_COLOR)
            fig.update_yaxes(gridcolor=GRID_COLOR)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No Elo history available for this team.")

    with col_right:
        st.subheader("📊 Results by Decade")
        team_matches["decade"] = (team_matches["date"].dt.year // 10) * 10
        decade_res = (
            team_matches.groupby(["decade", "team_result"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        fig2 = go.Figure()
        rank_map = {"Win": 1, "Draw": 2, "Loss": 3}
        for r in ["Win", "Draw", "Loss"]:
            if r in decade_res.columns:
                fig2.add_trace(go.Bar(
                    name=r, x=decade_res["decade"], y=decade_res[r],
                    marker_color=BAR_COLORS[r], marker_line_width=0,
                    legendrank=rank_map[r]
                ))
        fig2.update_layout(
            **PLOTLY_LAYOUT, barmode="stack", height=300,
            bargap=0.28, bargroupgap=0.1,
            xaxis_title="Decade", yaxis_title="Matches",
            legend=dict(orientation="h", y=1.08, x=0)
        )
        fig2.update_xaxes(
            tickmode="array",
            tickvals=decade_res["decade"].tolist(),
            ticktext=[str(d) for d in decade_res["decade"].tolist()],
            tickangle=0, gridcolor=GRID_COLOR
        )
        fig2.update_yaxes(gridcolor=GRID_COLOR)
        st.plotly_chart(fig2, use_container_width=True)

    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.subheader("🥇 Most Played Opponents")
        top_opp = (
            team_matches.groupby("opponent").size()
            .sort_values(ascending=False).head(10).reset_index()
        )
        top_opp.columns = ["Opponent", "Matches"]
        st.dataframe(top_opp, hide_index=True, use_container_width=True)

    with col_r2:
        st.subheader("🏆 Biggest Wins")
        big_wins = team_matches[team_matches["team_result"] == "Win"].copy()
        big_wins["margin"] = big_wins["team_score"] - big_wins["opp_score"]
        big_wins = big_wins.nlargest(8, "margin")[
            ["date", "opponent", "team_score", "opp_score", "tournament"]
        ].copy()
        big_wins["date"] = big_wins["date"].dt.strftime("%Y-%m-%d")
        big_wins.columns = ["Date", "Opponent", "For", "Against", "Tournament"]
        st.dataframe(big_wins, hide_index=True, use_container_width=True)

    st.subheader("⚽ Goals Scored vs Conceded per Year")
    yearly_goals = team_matches.groupby(team_matches["date"].dt.year).agg(
        scored   =("team_score", "mean"),
        conceded =("opp_score",  "mean")
    ).reset_index()
    yearly_goals.columns = ["year", "scored", "conceded"]
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=yearly_goals["year"], y=yearly_goals["scored"],
        name="Avg Scored", line=dict(color="#16a34a", width=2.5)
    ))
    fig3.add_trace(go.Scatter(
        x=yearly_goals["year"], y=yearly_goals["conceded"],
        name="Avg Conceded", line=dict(color="#dc2626", width=2.5)
    ))
    fig3.update_layout(
        **PLOTLY_LAYOUT, height=280,
        xaxis_title="Year", yaxis_title="Goals per match",
        legend=dict(orientation="h", y=1.08, x=0)
    )
    fig3.update_xaxes(gridcolor=GRID_COLOR)
    fig3.update_yaxes(gridcolor=GRID_COLOR)
    st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════
# PAGE 3: HEAD-TO-HEAD
# ═══════════════════════════════════════════════════════
elif page == "⚔️  Head-to-Head":
    st.markdown('<script>window.scrollTo(0, 0);</script>', unsafe_allow_html=True)

    st.title("⚔️ Head-to-Head Analyzer")

    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox(
            "Team A", ALL_TEAMS,
            index=ALL_TEAMS.index("Brazil") if "Brazil" in ALL_TEAMS else 0
        )
    with col2:
        team_b = st.selectbox(
            "Team B", ALL_TEAMS,
            index=ALL_TEAMS.index("Argentina") if "Argentina" in ALL_TEAMS else 1
        )

    if team_a == team_b:
        st.warning("Please select two different teams.")
        st.stop()

    h2h = matches[
        ((matches["home_team"] == team_a) & (matches["away_team"] == team_b)) |
        ((matches["home_team"] == team_b) & (matches["away_team"] == team_a))
    ].copy().sort_values("date")

    if len(h2h) == 0:
        st.warning(f"No matches found between **{team_a}** and **{team_b}**.")
        st.stop()

    def result_for_a(row):
        if row["home_team"] == team_a:
            return "Win" if row["result"] == "home_win" else ("Draw" if row["result"] == "draw" else "Loss")
        else:
            return "Win" if row["result"] == "away_win" else ("Draw" if row["result"] == "draw" else "Loss")

    h2h["result_a"] = h2h.apply(result_for_a, axis=1)
    a_wins = int((h2h["result_a"] == "Win").sum())
    b_wins = int((h2h["result_a"] == "Loss").sum())
    draws  = int((h2h["result_a"] == "Draw").sum())
    total  = len(h2h)

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{team_a} Wins", a_wins, f"{a_wins/total*100:.1f}%")
    c2.metric("Draws",          draws,  f"{draws/total*100:.1f}%")
    c3.metric(f"{team_b} Wins", b_wins, f"{b_wins/total*100:.1f}%")
    c4.metric("Total Matches",  total,  f"Since {h2h['date'].min().year}")

    a_pct = a_wins / total * 100
    d_pct = draws  / total * 100
    b_pct = b_wins / total * 100
    st.markdown(f"""
    <div style="margin:14px 0 4px;font-size:12px;font-weight:700;opacity:0.5;letter-spacing:0.06em;text-transform:uppercase;">
        Overall Dominance
    </div>
    <div style="display:flex;height:34px;border-radius:10px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.1);">
        <div style="width:{a_pct:.1f}%;background:#2563eb;display:flex;align-items:center;justify-content:center;color:white;font-weight:700;font-size:13px;">{a_pct:.0f}%</div>
        <div style="width:{d_pct:.1f}%;background:#6b7280;display:flex;align-items:center;justify-content:center;color:white;font-size:13px;">{d_pct:.0f}%</div>
        <div style="width:{b_pct:.1f}%;background:#dc2626;display:flex;align-items:center;justify-content:center;color:white;font-weight:700;font-size:13px;">{b_pct:.0f}%</div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:12px;opacity:0.5;margin:6px 0 20px;">
        <span>🔵 {team_a}</span><span>⬜ Draw</span><span>🔴 {team_b}</span>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Results by Decade")
        h2h["decade"] = (h2h["date"].dt.year // 10) * 10
        decade_h2h = (
            h2h.groupby(["decade", "result_a"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        fig = go.Figure()
        rank_map = {"Win": 1, "Draw": 2, "Loss": 3}
        for r in ["Win", "Draw", "Loss"]:
            if r in decade_h2h.columns:
                fig.add_trace(go.Bar(
                    name=r, x=decade_h2h["decade"], y=decade_h2h[r],
                    marker_color=BAR_COLORS[r], marker_line_width=0,
                    legendrank=rank_map[r]
                ))
        fig.update_layout(
            **PLOTLY_LAYOUT, barmode="stack", height=310,
            bargap=0.28, bargroupgap=0.1,
            title=dict(text=f"{team_a} record vs {team_b} by decade", font=dict(size=13)),
            legend=dict(orientation="h", y=1.12, x=0)
        )
        fig.update_xaxes(
            tickmode="array",
            tickvals=decade_h2h["decade"].tolist(),
            ticktext=[str(d) for d in decade_h2h["decade"].tolist()],
            tickangle=0, gridcolor=GRID_COLOR
        )
        fig.update_yaxes(gridcolor=GRID_COLOR)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Goal Distribution")
        h2h["a_goals"] = np.where(h2h["home_team"] == team_a, h2h["home_score"], h2h["away_score"])
        h2h["b_goals"] = np.where(h2h["home_team"] == team_b, h2h["home_score"], h2h["away_score"])
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=h2h["a_goals"], name=team_a, opacity=0.75,
            marker_color="#2563eb", nbinsx=10
        ))
        fig2.add_trace(go.Histogram(
            x=h2h["b_goals"], name=team_b, opacity=0.75,
            marker_color="#dc2626", nbinsx=10
        ))
        fig2.update_layout(
            **PLOTLY_LAYOUT, barmode="overlay", height=310,
            xaxis_title="Goals scored", yaxis_title="Matches",
            legend=dict(orientation="h", y=1.08, x=0),
            bargap=0.05
        )
        fig2.update_xaxes(gridcolor=GRID_COLOR)
        fig2.update_yaxes(gridcolor=GRID_COLOR)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📋 Full Match History")
    display_h2h = h2h[[
        "date", "home_team", "home_score", "away_score",
        "away_team", "tournament", "result_a"
    ]].copy()
    display_h2h["date"] = display_h2h["date"].dt.strftime("%Y-%m-%d")
    display_h2h = display_h2h.sort_values("date", ascending=False)
    display_h2h.columns = [
        "Date", "Home", "Home Score", "Away Score",
        "Away", "Tournament", f"Result ({team_a})"
    ]
    st.dataframe(display_h2h, hide_index=True, use_container_width=True, height=380)


# ═══════════════════════════════════════════════════════
# PAGE 4: ELO RANKINGS
# ═══════════════════════════════════════════════════════
elif page == "📈 Elo Rankings":
    st.markdown('<script>window.scrollTo(0, 0);</script>', unsafe_allow_html=True)

    st.title("📈 Elo Rankings & Timeline")

    tab1, tab2 = st.tabs(["🏆 Current Rankings", "📈 Rating Timeline"])

    with tab1:
        st.subheader("Current World Rankings by Elo")
        n_teams = st.slider("Show top N teams", 10, 50, 25)
        top_n   = ratings.head(n_teams).copy()

        fig = go.Figure(go.Bar(
            x=top_n["elo_rating"],
            y=top_n["team"],
            orientation="h",
            marker=dict(
                color=top_n["elo_rating"],
                colorscale="Blues",
                showscale=False
            ),
            text=top_n["elo_rating"].round(0).astype(int),
            textposition="outside"
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=max(440, n_teams * 24),
            bargap=0.22,
            yaxis=dict(autorange="reversed"),
            xaxis_title="Elo Rating",
        )
        fig.update_xaxes(gridcolor=GRID_COLOR)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Elo vs Win Rate (Top 50 Teams)")
        fig2 = px.scatter(
            ratings.head(50), x="elo_rating", y="win_rate",
            size="total_matches", color="elo_rating",
            color_continuous_scale="Blues",
            hover_name="team", text="team",
            labels={"elo_rating": "Elo Rating", "win_rate": "Win Rate"}
        )
        fig2.update_traces(textposition="top center", textfont_size=9)
        fig2.update_layout(
            **PLOTLY_LAYOUT, height=460,
            showlegend=False, coloraxis_showscale=False
        )
        fig2.update_xaxes(gridcolor=GRID_COLOR)
        fig2.update_yaxes(gridcolor=GRID_COLOR)
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.subheader("Elo Rating Over Time")
        default_teams   = [t for t in ["Brazil", "Germany", "Spain", "France", "Argentina", "England"]
                           if t in ALL_TEAMS]
        selected_teams  = st.multiselect("Select teams to compare", ALL_TEAMS, default=default_teams)

        if selected_teams:
            team_hist = history[history["team"].isin(selected_teams)].copy()
            team_hist["year"] = team_hist["date"].dt.year
            yearly = team_hist.groupby(["team", "year"])["elo_rating"].mean().reset_index()
            fig3 = px.line(
                yearly, x="year", y="elo_rating", color="team",
                color_discrete_sequence=LINE_COLORS,
                labels={"year": "Year", "elo_rating": "Elo Rating", "team": "Team"}
            )
            fig3.update_traces(line_width=2.2)
            fig3.update_layout(
                **PLOTLY_LAYOUT, height=460,
                legend=dict(orientation="h", y=1.06, x=0, title="")
            )
            fig3.update_xaxes(gridcolor=GRID_COLOR)
            fig3.update_yaxes(gridcolor=GRID_COLOR)
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Select at least one team to display the timeline.")

        st.subheader("⚡ Biggest Single-Match Elo Swings")
        big_swings = elo.reindex(
            elo["home_elo_change"].abs().sort_values(ascending=False).index
        ).head(10)
        big_swings = big_swings[[
            "date", "home_team", "away_team",
            "home_score", "away_score", "home_elo_change", "tournament"
        ]].copy()
        big_swings["date"]           = big_swings["date"].dt.strftime("%Y-%m-%d")
        big_swings["home_elo_change"] = big_swings["home_elo_change"].round(1)
        big_swings.columns = ["Date", "Home", "Away", "HG", "AG", "Home Elo Δ", "Tournament"]
        st.dataframe(big_swings, hide_index=True, use_container_width=True)


# ═══════════════════════════════════════════════════════
# PAGE 5: MATCH PREDICTOR
# ═══════════════════════════════════════════════════════
elif page == "🤖 Match Predictor":
    st.markdown('<script>window.scrollTo(0, 0);</script>', unsafe_allow_html=True)

    st.title("🤖 Live Match Predictor")
    st.markdown("Uses the trained XGBoost model to predict match outcomes based on current team strengths.")

    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        home_team = st.selectbox(
            "🏠 Home Team", ALL_TEAMS,
            index=ALL_TEAMS.index("Brazil") if "Brazil" in ALL_TEAMS else 0
        )
    with col2:
        st.markdown(
            "<br><div style='text-align:center;font-size:26px;font-weight:800;opacity:0.4;'>vs</div>",
            unsafe_allow_html=True
        )
    with col3:
        away_team = st.selectbox(
            "✈️ Away Team", ALL_TEAMS,
            index=ALL_TEAMS.index("Argentina") if "Argentina" in ALL_TEAMS else 1
        )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        is_neutral  = st.checkbox("Neutral venue", value=False)
    with col_b:
        tier_map    = {1: "World Cup / Continental Final", 2: "Qualification",
                       3: "Regional Cup", 4: "Friendly"}
        tier_choice = st.selectbox("Tournament type", list(tier_map.values()))
        tier_val    = [k for k, v in tier_map.items() if v == tier_choice][0]
    with col_c:
        match_month = st.slider("Match month", 1, 12, 6)

    predict_clicked = st.button("🔮 Predict Outcome", type="primary")

    if predict_clicked:
        if home_team == away_team:
            st.error("Please select two different teams.")
            st.stop()

        # ── Elo ratings ──
        h_elo_r = ratings[ratings["team"] == home_team]["elo_rating"]
        a_elo_r = ratings[ratings["team"] == away_team]["elo_rating"]
        h_elo   = float(h_elo_r.values[0]) if len(h_elo_r) else 1000.0
        a_elo   = float(a_elo_r.values[0]) if len(a_elo_r) else 1000.0
        elo_diff = h_elo - a_elo
        exp_prob = 1 / (1 + 10 ** ((a_elo - h_elo) / 400))

        # ── Recent form ──
        def get_form(team, window=10):
            tm = matches[
                (matches["home_team"] == team) | (matches["away_team"] == team)
            ].tail(window)
            w = d = l = 0
            for _, row in tm.iterrows():
                is_home = row["home_team"] == team
                if (is_home and row["result"] == "home_win") or \
                   (not is_home and row["result"] == "away_win"):
                    w += 1
                elif row["result"] == "draw":
                    d += 1
                else:
                    l += 1
            n = w + d + l
            return (w/n if n > 0 else 0.33), (d/n if n > 0 else 0.33)

        h_wr5,  h_dr5 = get_form(home_team, 5)
        a_wr5,  a_dr5 = get_form(away_team, 5)
        h_wr10, _     = get_form(home_team, 10)
        a_wr10, _     = get_form(away_team, 10)

        # ── Goals ──
        def get_goals(team, window=10):
            h  = matches[matches["home_team"] == team].tail(window)
            a  = matches[matches["away_team"] == team].tail(window)
            sc = pd.concat([h["home_score"], a["away_score"]]).mean()
            cn = pd.concat([h["away_score"], a["home_score"]]).mean()
            return (sc if not np.isnan(sc) else 1.5,
                    cn if not np.isnan(cn) else 1.2)

        h_sc, h_cn = get_goals(home_team)
        a_sc, a_cn = get_goals(away_team)

        # ── H2H ──
        h2h_sub = matches[
            ((matches["home_team"] == home_team) & (matches["away_team"] == away_team)) |
            ((matches["home_team"] == away_team) & (matches["away_team"] == home_team))
        ]
        h2h_total = len(h2h_sub)
        if h2h_total > 0:
            hw = (
                ((h2h_sub["home_team"] == home_team) & (h2h_sub["result"] == "home_win")).sum() +
                ((h2h_sub["away_team"] == home_team) & (h2h_sub["result"] == "away_win")).sum()
            )
            h2h_home_wr = hw / h2h_total
        else:
            h2h_home_wr = 0.4
            h2h_total   = 5

        FEATURES = [
            "home_elo_pre", "away_elo_pre", "elo_diff", "exp_home_win_prob",
            "home_win_rate_5", "home_win_rate_10", "away_win_rate_5", "away_win_rate_10",
            "home_draw_rate_5", "away_draw_rate_5",
            "home_avg_scored_10", "home_avg_conceded_10",
            "away_avg_scored_10", "away_avg_conceded_10",
            "h2h_home_win_rate", "h2h_total_matches",
            "tournament_tier", "neutral", "rest_advantage", "month",
        ]

        if len(ml_data) > 0:
            from xgboost import XGBClassifier

            @st.cache_resource
            def train_model():
                _ml = pd.read_csv("ml_dataset.csv", parse_dates=["date"])
                X   = _ml[FEATURES].fillna(0)
                y   = _ml["target"]
                mdl = XGBClassifier(
                    n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    objective="multi:softprob", num_class=3,
                    eval_metric="mlogloss", use_label_encoder=False,
                    random_state=42, n_jobs=-1
                )
                mdl.fit(X, y)
                return mdl

            with st.spinner("Training model on first run — please wait…"):
                model = train_model()

            feat_row = pd.DataFrame([[
                h_elo, a_elo, elo_diff, exp_prob,
                h_wr5, h_wr10, a_wr5, a_wr10,
                h_dr5, a_dr5,
                h_sc, h_cn, a_sc, a_cn,
                h2h_home_wr, h2h_total,
                tier_val, int(is_neutral), 0, match_month
            ]], columns=FEATURES)

            proba  = model.predict_proba(feat_row)[0]
            # target: 0=away_win, 1=draw, 2=home_win
            p_away, p_draw, p_home = float(proba[0]), float(proba[1]), float(proba[2])
        else:
            p_home = exp_prob * 0.72
            p_draw = 0.24
            p_away = max(0.0, 1 - p_home - p_draw)

        st.markdown("---")
        st.subheader("🔮 Prediction Results")

        col_h, col_d, col_a = st.columns(3)
        with col_h:
            st.markdown(f"""
            <div class="prediction-card" style="border-top:4px solid #2563eb;">
                <div style="font-size:12px;font-weight:700;opacity:0.5;text-transform:uppercase;letter-spacing:0.06em;">🏠 {home_team} Win</div>
                <div style="font-size:50px;font-weight:800;color:#2563eb;line-height:1.1;margin:8px 0;">{p_home*100:.1f}%</div>
                <div style="font-size:12px;opacity:0.45;">Elo: {h_elo:.0f}</div>
            </div>""", unsafe_allow_html=True)
        with col_d:
            st.markdown(f"""
            <div class="prediction-card" style="border-top:4px solid #6b7280;">
                <div style="font-size:12px;font-weight:700;opacity:0.5;text-transform:uppercase;letter-spacing:0.06em;">🤝 Draw</div>
                <div style="font-size:50px;font-weight:800;color:#6b7280;line-height:1.1;margin:8px 0;">{p_draw*100:.1f}%</div>
                <div style="font-size:12px;opacity:0.45;">Elo diff: {elo_diff:+.0f}</div>
            </div>""", unsafe_allow_html=True)
        with col_a:
            st.markdown(f"""
            <div class="prediction-card" style="border-top:4px solid #dc2626;">
                <div style="font-size:12px;font-weight:700;opacity:0.5;text-transform:uppercase;letter-spacing:0.06em;">✈️ {away_team} Win</div>
                <div style="font-size:50px;font-weight:800;color:#dc2626;line-height:1.1;margin:8px 0;">{p_away*100:.1f}%</div>
                <div style="font-size:12px;opacity:0.45;">Elo: {a_elo:.0f}</div>
            </div>""", unsafe_allow_html=True)

        # Probability bar
        st.markdown(f"""
        <div style="margin:20px 0 6px;display:flex;height:36px;border-radius:10px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.1);">
            <div style="width:{p_home*100:.1f}%;background:#2563eb;display:flex;align-items:center;justify-content:center;color:white;font-weight:700;font-size:13px;">{p_home*100:.0f}%</div>
            <div style="width:{p_draw*100:.1f}%;background:#6b7280;display:flex;align-items:center;justify-content:center;color:white;font-size:13px;">{p_draw*100:.0f}%</div>
            <div style="width:{p_away*100:.1f}%;background:#dc2626;display:flex;align-items:center;justify-content:center;color:white;font-weight:700;font-size:13px;">{p_away*100:.0f}%</div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:12px;opacity:0.45;margin-bottom:20px;">
            <span>🔵 {home_team}</span><span>⬜ Draw</span><span>🔴 {away_team}</span>
        </div>
        """, unsafe_allow_html=True)

        max_p = max(p_home, p_draw, p_away)
        if p_home == max_p:
            verdict = f"🏠 **{home_team}** are favourites"
        elif p_away == max_p:
            verdict = f"✈️ **{away_team}** are favourites"
        else:
            verdict = "🤝 This match is too close to call — a draw is most likely"
        st.markdown(f"### {verdict}")

        st.subheader("📊 Feature Importance (model trained on 40,000+ matches)")
        fig = px.bar(
            importance.head(12), x="importance_pct", y="feature",
            orientation="h", color="importance_pct",
            color_continuous_scale="Blues",
            labels={"importance_pct": "Importance %", "feature": "Feature"}
        )
        fig.update_traces(marker_line_width=0)
        fig.update_layout(
            **PLOTLY_LAYOUT, height=360, showlegend=False,
            coloraxis_showscale=False, bargap=0.22,
            yaxis=dict(autorange="reversed")
        )
        fig.update_xaxes(gridcolor=GRID_COLOR)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════
# PAGE 6: 2026 WORLD CUP
# ═══════════════════════════════════════════════════════
elif page == "🏆 2026 World Cup":
    st.markdown('<script>window.scrollTo(0, 0);</script>', unsafe_allow_html=True)

    st.title("🏆 2026 FIFA World Cup Predictions")
    st.markdown("Match outcome probabilities generated by the XGBoost model trained on 149 years of football history.")

    if len(predictions) == 0:
        st.warning("Run `Match_Prediction.ipynb` first to generate `ml_predictions_2026.csv`.")
        st.stop()

    home_fav = int((predictions["prob_home_win"] > predictions["prob_away_win"]).sum())
    away_fav = int((predictions["prob_away_win"] > predictions["prob_home_win"]).sum())
    close    = int((abs(predictions["prob_home_win"] - predictions["prob_away_win"]) < 10).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fixtures Predicted",       len(predictions))
    c2.metric("Home Team Favoured",       home_fav)
    c3.metric("Away Team Favoured",       away_fav)
    c4.metric("Too Close to Call (<10%)", close)

    st.markdown("---")
    st.subheader("📋 All Fixtures with Win Probabilities")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        all_wc_teams = sorted(pd.concat([
            predictions["home_team"], predictions["away_team"]
        ]).unique().tolist())
        filter_team = st.selectbox("Filter by team (optional)", ["All"] + all_wc_teams)
    with col_f2:
        sort_by = st.selectbox("Sort by", ["Date", "Home Win %", "Away Win %", "Home Elo"])

    display_pred = predictions.copy()
    if filter_team != "All":
        display_pred = display_pred[
            (display_pred["home_team"] == filter_team) |
            (display_pred["away_team"] == filter_team)
        ]

    sort_map = {
        "Date":       ("date",          True),
        "Home Win %": ("prob_home_win", False),
        "Away Win %": ("prob_away_win", False),
        "Home Elo":   ("home_elo",      False),
    }
    sort_col, sort_asc = sort_map[sort_by]
    display_pred = display_pred.sort_values(sort_col, ascending=sort_asc)

    # Format for display
    disp = display_pred.copy()
    disp["date"] = pd.to_datetime(disp["date"]).dt.strftime("%Y-%m-%d")
    for col in ["prob_home_win", "prob_draw", "prob_away_win"]:
        if col in disp.columns:
            disp[col] = disp[col].round(1).astype(str) + "%"
    for col in ["home_elo", "away_elo"]:
        if col in disp.columns:
            disp[col] = disp[col].round(0).astype(int)

    rename_map = {
        "date":             "Date",
        "home_team":        "Home Team",
        "away_team":        "Away Team",
        "home_elo":         "Home Elo",
        "away_elo":         "Away Elo",
        "prob_home_win":    "Home Win %",
        "prob_draw":        "Draw %",
        "prob_away_win":    "Away Win %",
        "predicted_result": "Predicted Result",
        "favourite":        "Favourite",
    }
    disp = disp.rename(columns={k: v for k, v in rename_map.items() if k in disp.columns})
    st.dataframe(disp, hide_index=True, use_container_width=True, height=500)

    st.subheader("🏅 Predicted Strongest Teams in Tournament")
    team_strength = pd.concat([
        predictions[["home_team", "home_elo"]].rename(columns={"home_team": "team", "home_elo": "elo"}),
        predictions[["away_team", "away_elo"]].rename(columns={"away_team": "team", "away_elo": "elo"}),
    ]).drop_duplicates("team").sort_values("elo", ascending=False).head(16)

    fig = px.bar(
        team_strength, x="team", y="elo",
        color="elo", color_continuous_scale="Reds",
        labels={"team": "Team", "elo": "Elo Rating"},
        text="elo"
    )
    fig.update_traces(
        texttemplate="%{text:.0f}", textposition="outside", marker_line_width=0
    )
    fig.update_layout(
        **PLOTLY_LAYOUT, height=400, showlegend=False,
        coloraxis_showscale=False, bargap=0.3,
        xaxis_tickangle=-30
    )
    fig.update_xaxes(gridcolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR)
    st.plotly_chart(fig, use_container_width=True)