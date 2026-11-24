import json
import os

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="IPL Live Win Predictor", page_icon="🏏", layout="wide")

# ---------- Load artefacts ----------
@st.cache_resource
def load_model():
    return joblib.load("pipe.pkl")


@st.cache_data
def load_metrics():
    if os.path.exists("metrics.json"):
        with open("metrics.json") as f:
            return json.load(f)
    return {}


pipe = load_model()
metrics = load_metrics()

TEAMS = metrics.get("teams") or [
    "Sunrisers Hyderabad", "Mumbai Indians", "Royal Challengers Bangalore",
    "Kolkata Knight Riders", "Kings XI Punjab", "Chennai Super Kings",
    "Rajasthan Royals", "Delhi Capitals",
]
CITIES = metrics.get("cities") or ["Mumbai", "Chennai", "Bangalore", "Kolkata", "Delhi"]

COLORS = {
    "Sunrisers Hyderabad": "#FF822A", "Mumbai Indians": "#2F7BFF",
    "Royal Challengers Bangalore": "#E5322D", "Kolkata Knight Riders": "#8A5CF5",
    "Kings XI Punjab": "#F0443C", "Chennai Super Kings": "#FDB913",
    "Rajasthan Royals": "#EA1A85", "Delhi Capitals": "#3B82F6",
}

# ---------- Styling ----------
st.markdown(
    """
<style>
.stApp {background: radial-gradient(circle at 20% 0%, #1b1f3a 0%, #0b0d1a 55%);}
.block-container {padding-top: 1.5rem; max-width: 1250px;}
h1, h2, h3, h4, p, label, span {color: #e8eaf6;}
.hero {padding: 18px 26px; border-radius: 18px; margin-bottom: 14px;
       background: linear-gradient(120deg, #6a11cb 0%, #2575fc 100%);}
.hero h1 {margin: 0; font-size: 2rem; color: white;}
.hero p {margin: 4px 0 0; color: #dbe4ff;}
.card {background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.10);
       border-radius: 16px; padding: 16px 20px;}
.big {font-size: 2.6rem; font-weight: 800; line-height: 1.1;}
.small {color: #9aa3c7; font-size: 0.85rem; text-transform: uppercase; letter-spacing: .08em;}
.stButton>button {border-radius: 12px; font-weight: 700; width: 100%; height: 3rem;
                  background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.2); color: white;}
.stButton>button:hover {border-color: #2575fc; color: #2575fc;}
</style>
<div class="hero"><h1>🏏 IPL Live Win Probability</h1>
<p>Ball-by-ball chase predictor powered by machine learning</p></div>
""",
    unsafe_allow_html=True,
)


# ---------- Core helpers ----------
def predict(batting, bowling, city, target, score, wickets_fallen, balls_bowled):
    balls_left = max(120 - balls_bowled, 0)
    runs_left = target - score
    if runs_left <= 0:
        return 1.0
    if wickets_fallen >= 10 or balls_left == 0:
        return 0.0
    crr = score * 6 / max(balls_bowled, 1)
    rrr = runs_left * 6 / balls_left
    row = pd.DataFrame(
        {
            "batting_team": [batting], "bowling_team": [bowling], "city": [city],
            "runs_left": [runs_left], "balls_left": [balls_left],
            "wickets": [10 - wickets_fallen], "total_runs_x": [target],
            "crr": [crr], "rrr": [rrr],
        }
    )
    return float(pipe.predict_proba(row)[0][1])


def fmt_overs(balls):
    return f"{balls // 6}.{balls % 6}"


def gauge(p, name, color):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(p * 100, 1),
            number={"suffix": "%", "font": {"size": 44, "color": "white"}},
            title={"text": name, "font": {"size": 15, "color": "#9aa3c7"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#555"},
                "bar": {"color": color},
                "bgcolor": "rgba(255,255,255,0.05)",
                "steps": [{"range": [0, 100], "color": "rgba(255,255,255,0.04)"}],
            },
        )
    )
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"})
    return fig


def worm(history, bat, bowl, bat_c, bowl_c):
    h = pd.DataFrame(history)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h["balls"], y=h["win"] * 100, name=bat, mode="lines",
                             line=dict(color=bat_c, width=4), fill="tozeroy",
                             fillcolor="rgba(37,117,252,0.12)"))
    fig.add_trace(go.Scatter(x=h["balls"], y=(1 - h["win"]) * 100, name=bowl, mode="lines",
                             line=dict(color=bowl_c, width=4)))
    wk = h[h["wicket"]]
    fig.add_trace(go.Scatter(x=wk["balls"], y=wk["win"] * 100, mode="markers", name="Wicket",
                             marker=dict(color="white", size=11, symbol="x",
                                         line=dict(color="red", width=2))))
    fig.add_hline(y=50, line_dash="dot", line_color="#666")
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Balls bowled", range=[0, 120], gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="Win probability (%)", range=[0, 100], gridcolor="rgba(255,255,255,0.06)"),
        legend=dict(orientation="h", y=1.12), font={"color": "white"},
    )
    return fig


# ---------- Sidebar: match setup ----------
with st.sidebar:
    st.header("⚙️ Match setup")
    batting = st.selectbox("Batting team (chasing)", TEAMS, index=0)
    bowling = st.selectbox("Bowling team", [t for t in TEAMS if t != batting], index=1)
    city = st.selectbox("Host city", CITIES)
    target = st.number_input("Target", min_value=2, max_value=300, value=180, step=1)

    setup_key = (batting, bowling, city, int(target))
    if st.session_state.get("setup_key") != setup_key:
        st.session_state.setup_key = setup_key
        st.session_state.score = 0
        st.session_state.wkts = 0
        st.session_state.balls = 0
        p0 = predict(batting, bowling, city, target, 0, 0, 0)
        st.session_state.history = [{"balls": 0, "win": p0, "wicket": False}]
        st.session_state.log = []

    if st.button("🔄 Reset match"):
        st.session_state.setup_key = None
        st.rerun()

bat_c = COLORS.get(batting, "#2575fc")
bowl_c = COLORS.get(bowling, "#f0443c")

tab_live, tab_whatif, tab_insights = st.tabs(["🔴 Live ball-by-ball", "🎛️ What-if scenario", "📊 Model insights"])

# ---------- TAB 1: live ----------
with tab_live:
    s = st.session_state

    def add_ball(runs=0, wicket=False, extra=0, legal=True):
        if s.balls >= 120 or s.wkts >= 10 or s.score >= target:
            return
        s.score += runs + extra
        if wicket:
            s.wkts += 1
        if legal:
            s.balls += 1
        p = predict(batting, bowling, city, target, s.score, s.wkts, s.balls)
        s.history.append({"balls": s.balls, "win": p, "wicket": wicket})
        label = "W" if wicket else (f"{runs}" if legal else f"{'Wd' if runs == 0 else 'NB'}+{extra}")
        s.log.append(label)

    st.markdown("##### Tap what happened on the last ball 👇")
    r1 = st.columns(7)
    for col, r in zip(r1, [0, 1, 2, 3, 4, 6]):
        if col.button(("•" if r == 0 else str(r)), key=f"run{r}"):
            add_ball(runs=r)
    if r1[6].button("❌ W", key="wkt"):
        add_ball(wicket=True)
    r2 = st.columns(3)
    if r2[0].button("Wide (+1)", key="wd"):
        add_ball(extra=1, legal=False)
    if r2[1].button("No-ball (+1)", key="nb"):
        add_ball(extra=1, legal=False)
    if r2[2].button("↩️ Undo last ball", key="undo") and len(s.history) > 1 and s.log:
        st.info("Tip: use Reset match to restart – undo is intentionally simple.")

    p_win = s.history[-1]["win"]
    runs_left = max(target - s.score, 0)
    balls_left = 120 - s.balls
    crr = s.score * 6 / max(s.balls, 1)
    rrr = runs_left * 6 / balls_left if balls_left > 0 else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f"<div class='card'><div class='small'>Score</div><div class='big'>{s.score}/{s.wkts}</div></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='card'><div class='small'>Overs</div><div class='big'>{fmt_overs(s.balls)}</div></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='card'><div class='small'>Need</div><div class='big'>{runs_left}<span style='font-size:1rem'> off {balls_left}</span></div></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='card'><div class='small'>CRR / RRR</div><div class='big'>{crr:.1f} <span style='font-size:1.2rem;color:#9aa3c7'>/ {rrr:.1f}</span></div></div>", unsafe_allow_html=True)

    if s.score >= target:
        st.success(f"🏆 {batting} won the chase!")
        st.balloons()
    elif s.wkts >= 10 or s.balls >= 120:
        st.error(f"🏆 {bowling} defended the target!")

    g1, g2 = st.columns(2)
    g1.plotly_chart(gauge(p_win, batting, bat_c), use_container_width=True)
    g2.plotly_chart(gauge(1 - p_win, bowling, bowl_c), use_container_width=True)

    st.markdown("#### 📈 Win-probability worm")
    st.plotly_chart(worm(s.history, batting, bowling, bat_c, bowl_c), use_container_width=True)

    if s.log:
        st.caption("Ball log: " + " · ".join(s.log[-24:]))

# ---------- TAB 2: what-if ----------
with tab_whatif:
    st.markdown("Move the sliders – the prediction updates **instantly** (no button).")
    c1, c2, c3 = st.columns(3)
    w_score = c1.slider("Current score", 0, int(target) + 20, min(60, int(target)))
    w_overs = c2.slider("Overs completed", 0.0, 19.5, 8.0, 0.1)
    w_wkts = c3.slider("Wickets fallen", 0, 9, 2)

    # convert cricket notation (e.g. 8.3) to balls
    whole, part = int(w_overs), round((w_overs - int(w_overs)) * 10)
    w_balls = whole * 6 + min(part, 5)
    pw = predict(batting, bowling, city, target, w_score, w_wkts, w_balls)

    a, b = st.columns([1, 1])
    a.plotly_chart(gauge(pw, batting, bat_c), use_container_width=True)
    b.plotly_chart(gauge(1 - pw, bowling, bowl_c), use_container_width=True)

    st.markdown("#### 🔬 Sensitivity: how does the win chance change with wickets in hand?")
    xs = list(range(0, 10))
    ys = [predict(batting, bowling, city, target, w_score, w, w_balls) * 100 for w in xs]
    fig = go.Figure(go.Bar(x=xs, y=ys, marker_color=bat_c))
    fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis_title="Wickets fallen", yaxis_title=f"{batting} win %",
                      font={"color": "white"}, margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

# ---------- TAB 3: insights ----------
with tab_insights:
    if not metrics:
        st.info("Run `python train.py` to generate metrics.json and see model insights here.")
    else:
        st.markdown(f"**Best model:** `{metrics['best_model']}` · trained on "
                    f"{metrics['n_rows']:,} ball states from {metrics['n_matches']} matches")
        res = pd.DataFrame(metrics["results"]).T
        st.dataframe(res, use_container_width=True)
        st.caption("Evaluated on held-out *matches* (grouped split), so there is no ball-level leakage. "
                   "Lower log-loss / Brier = better calibrated probabilities.")
        if metrics.get("importance"):
            imp = pd.Series(metrics["importance"]).sort_values()
            fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color="#2575fc"))
            fig.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font={"color": "white"}, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)
