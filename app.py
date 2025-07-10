import streamlit as st
import pickle
import pandas as pd
from PIL import Image

# Load trained pipeline
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Teams and Cities
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
         'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
         'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Pune', 'Rajkot', 'Indore', 'Mumbai', 'Kolkata', 'Delhi',
          'Chandigarh', 'Kanpur', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala', 'Kochi',
          'Visakhapatnam', 'Raipur', 'Ranchi', 'Abu Dhabi', 'Sharjah', 'Mohali', 'Bengaluru']

# Page Config
st.set_page_config(page_title="IPL Win Predictor", layout="centered")

# Title
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🏏 IPL Winning Probability Predictor</h1>", unsafe_allow_html=True)

# Team selection
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('🟢 Select the Batting Team', sorted(teams))
with col2:
    bowling_team = st.selectbox('🔵 Select the Bowling Team', sorted(teams))

selected_city = st.selectbox('📍 Select Host City', sorted(cities))
target = st.number_input('🎯 Target Score', min_value=1)

# Match inputs
col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('🏏 Current Score', min_value=0)
with col4:
    overs = st.number_input('⏱️ Overs Completed', min_value=0.0, max_value=20.0, step=0.1)
with col5:
    wickets_out = st.number_input('❌ Wickets Fallen', min_value=0, max_value=10)

# Predict Button
if st.button('🚀 Predict Probability'):

    # Basic validations
    if batting_team == bowling_team:
        st.warning("⚠️ Batting and Bowling team cannot be the same.")
    elif overs == 0:
        st.warning("⚠️ Overs must be greater than 0 to calculate CRR.")
    elif score >= target:
        st.success("🎉 Target already achieved! You won! 🏆")
    else:
        # Feature engineering
        runs_left = target - score
        balls_left = 120 - int(overs * 6)
        wickets = 10 - wickets_out
        crr = score / overs
        rrr = (runs_left * 6) / balls_left

        # Input dataframe
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        # Predict probabilities
        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]

        batting_win_percent = round(win * 100)
        bowling_win_percent = round(loss * 100)

        st.markdown("### 🧮 Winning Probabilities")

        if win > loss:
            st.success(f"✅ {batting_team} Win Probability: **{batting_win_percent}%**")
            st.error(f"❌ {bowling_team} Win Probability: **{bowling_win_percent}%**")
            st.progress(win)
        else:
            st.success(f"✅ {bowling_team} Win Probability: **{bowling_win_percent}%**")
            st.error(f"❌ {batting_team} Win Probability: **{batting_win_percent}%**")
            st.progress(loss)

        # Optional confetti
        if win > 0.8 or loss > 0.8:
            st.balloons()

