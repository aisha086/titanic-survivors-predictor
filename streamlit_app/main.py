import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("streamlit_app/model/titanic_model.pkl")

# Title
st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to predict survival.")

# Input widgets
sex = st.radio("Sex", ["male", "female"])
pclass = st.selectbox("Passenger Class", [1, 2, 3])
age = st.slider("Age", 0, 80, 25)
family_size = st.slider("Family Size (Siblings + Spouses + Parents + Children)", 0, 10, 1)

# Encode inputs
sex_encoded = 1 if sex == "male" else 0

# Feature vector
features = np.array([[sex_encoded, pclass, age, family_size]])

# Predict
if st.button("Predict"):
    prediction = model.predict(features)
    proba = model.predict_proba(features)
    confidence = proba[0][1]  # Probability of Surviving

    if prediction[0] == 1:
        st.success("🎉 Predicted: Survived")
    else:
        st.error("💀 Predicted: Did Not Survive")

    st.write(f"**Confidence of Survival:** `{confidence:.2%}`")
    st.progress(confidence)

    # Optional bar chart
    proba_df = pd.DataFrame({
        'Class': ['Did not Survive', 'Survived'],
        'Probability': proba[0]
    })
    st.markdown("#### 📊 Prediction Probabilities")
    st.bar_chart(proba_df.set_index('Class'))

