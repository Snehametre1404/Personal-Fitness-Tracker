import streamlit as st
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings("ignore")

# Set Streamlit Page Config
st.set_page_config(page_title="Personal Fitness Tracker", page_icon="💪", layout="wide")

# Custom Styling
st.markdown("""
    <style>
        .stApp {
            background-color: #f7f9fc;
        }
        .title {
            text-align: center;
            font-size: 40px;
            color: #FF5733;
            font-weight: bold;
        }
        .subheader {
            text-align: center;
            font-size: 22px;
            color: #555;
        }
        .sidebar-text {
            font-size: 18px;
            color: #333;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="title">🔥 Personal Fitness Tracker 🔥</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Predict calories burned based on your fitness parameters.</p>', unsafe_allow_html=True)

# Sidebar - User Input Parameters
st.sidebar.header("📌 User Input Parameters")
def user_input_features():
    st.sidebar.markdown('<p class="sidebar-text">Adjust the sliders to input your details:</p>', unsafe_allow_html=True)
    age = st.sidebar.slider("Age", 10, 100, 30)
    bmi = st.sidebar.slider("BMI", 15, 40, 20)
    duration = st.sidebar.slider("Duration (min)", 0, 60, 30)
    heart_rate = st.sidebar.slider("Heart Rate", 60, 180, 90)
    body_temp = st.sidebar.slider("Body Temperature (C)", 36, 42, 37)
    gender = st.sidebar.radio("Gender", ["Male", "Female"], horizontal=True)
    gender_value = 1 if gender == "Male" else 0

    return pd.DataFrame({
        "Age": [age],
        "BMI": [bmi],
        "Duration": [duration],
        "Heart_Rate": [heart_rate],
        "Body_Temp": [body_temp],
        "Gender_male": [gender_value]
    })

df = user_input_features()

st.write("---")
st.header("🎯 Your Selected Parameters")
st.dataframe(df, width=800)

# Load Data
@st.cache_data
def load_data():
    calories = pd.read_csv("calories.csv")
    exercise = pd.read_csv("exercise.csv")
    data = exercise.merge(calories, on="User_ID").drop(columns=["User_ID"])
    data["BMI"] = round(data["Weight"] / ((data["Height"] / 100) ** 2), 2)
    return data

data = load_data()
train_data, test_data = train_test_split(data, test_size=0.2, random_state=1)
train_data = train_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
test_data = test_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
train_data = pd.get_dummies(train_data, drop_first=True)
test_data = pd.get_dummies(test_data, drop_first=True)

# Model Training
X_train, y_train = train_data.drop("Calories", axis=1), train_data["Calories"]
random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
random_reg.fit(X_train, y_train)

df = df.reindex(columns=X_train.columns, fill_value=0)
prediction = random_reg.predict(df)

# Display Prediction
st.write("---")
st.header("🔥 Calories Burned Prediction")
st.success(f"🔥 You are estimated to burn **{round(prediction[0], 2)} kilocalories**.")

# Find Similar Cases
st.write("---")
st.header("🔍 Similar Fitness Profiles")
cal_range = [prediction[0] - 10, prediction[0] + 10]
similar_data = data[(data["Calories"] >= cal_range[0]) & (data["Calories"] <= cal_range[1])]
st.dataframe(similar_data.sample(5) if len(similar_data) > 5 else similar_data)

# Insights
st.write("---")
st.header("📊 Comparison Insights")
st.info(f"🏋️‍♂️ You are older than **{round((data['Age'] < df['Age'].values[0]).mean() * 100, 2)}%** of people.")
st.info(f"⏳ Your workout duration is longer than **{round((data['Duration'] < df['Duration'].values[0]).mean() * 100, 2)}%** of people.")
st.info(f"❤️ Your heart rate is higher than **{round((data['Heart_Rate'] < df['Heart_Rate'].values[0]).mean() * 100, 2)}%** of people.")
st.info(f"🌡️ Your body temperature is higher than **{round((data['Body_Temp'] < df['Body_Temp'].values[0]).mean() * 100, 2)}%** of people.")

# Visualizations
st.write("---")
st.header("📈 Data Visualizations")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(data['Calories'], bins=30, kde=True, color='blue', ax=ax)
ax.set_title("Distribution of Calories Burned", fontsize=14)
st.pyplot(fig)

st.write("---")
st.subheader("💪 Thank you for using the Personal Fitness Tracker! Stay healthy and active! 💥")
