"""
Streamlit App - Wellness Tourism Package Purchase Predictor
--------------------------------------------------------------
Loads the model committed to the repository by the training job and
serves an interactive form. Customer inputs are collected into a
dataframe with the exact schema the model was trained on and passed
to the pipeline for a live prediction.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "best_model.joblib"

st.set_page_config(page_title="Wellness Package Predictor", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def main():
    st.title("Visit with Us: Wellness Tourism Package Predictor")
    st.write(
        "Enter a customer\'s profile and interaction details to predict "
        "whether they are likely to purchase the Wellness Tourism Package."
    )

    model = load_model()
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        num_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
        num_followups = st.number_input("Number of Followups", min_value=0, max_value=10, value=3)
        product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
        duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=60, value=15)

    with col2:
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        num_trips = st.number_input("Number of Trips per Year", min_value=0, max_value=20, value=3)
        passport = st.selectbox("Holds Passport", ["Yes", "No"])
        pitch_satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)
        own_car = st.selectbox("Owns a Car", ["Yes", "No"])
        num_children = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=200000, value=22000)

    if st.button("Predict"):
        input_df = pd.DataFrame([{
            "Age": age, "TypeofContact": type_of_contact, "CityTier": city_tier,
            "DurationOfPitch": duration_of_pitch, "Occupation": occupation, "Gender": gender,
            "NumberOfPersonVisiting": num_person_visiting, "NumberOfFollowups": num_followups,
            "ProductPitched": product_pitched, "PreferredPropertyStar": preferred_property_star,
            "MaritalStatus": marital_status, "NumberOfTrips": num_trips,
            "Passport": 1 if passport == "Yes" else 0, "PitchSatisfactionScore": pitch_satisfaction,
            "OwnCar": 1 if own_car == "Yes" else 0, "NumberOfChildrenVisiting": num_children,
            "Designation": designation, "MonthlyIncome": monthly_income,
        }])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.success(f"Likely to purchase the Wellness Package (probability: {probability:.2%})")
        else:
            st.info(f"Unlikely to purchase the Wellness Package (probability: {probability:.2%})")


if __name__ == "__main__":
    main()
