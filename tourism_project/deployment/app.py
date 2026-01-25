#!/usr/bin/env python3
"""
Streamlit App for Tourism Package Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download

# Page configuration
st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="🏖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    """Load the trained model from HuggingFace Hub"""
    #try:
        model_path = hf_hub_download(
            repo_id="u2jyothibhat/tourism-package-prediction-model",
            filename="best_model.joblib"
        )
        model = joblib.load(model_path)
        return model
    #except Exception as e:
        #st.error(f"Error loading model: {e}")
        #return None

def prepare_input_data(age, gender, marital_status, city_tier, type_of_contact,
                      occupation, designation, monthly_income, num_person_visiting,
                      num_children_visiting, preferred_property_star, num_trips,
                      passport, own_car, duration_of_pitch, product_pitched,
                      num_followups, pitch_satisfaction_score):
    """Prepare input data for model prediction"""

    # Create mapping dictionaries
    gender_map = {"Male": 1, "Female": 0}
    marital_map = {"Single": 2, "Married": 1, "Divorced": 0, "Unmarried": 3}
    contact_map = {"Self Enquiry": 1, "Company Invited": 0}
    occupation_map = {"Salaried": 2, "Small Business": 1, "Free Lancer": 0}
    designation_map = {"Executive": 0, "Manager": 1, "Senior Manager": 2, "AVP": 3, "VP": 4}
    product_map = {"Basic": 0, "Standard": 1, "Deluxe": 2, "Super Deluxe": 3}
    passport_map = {"Yes": 1, "No": 0}
    car_map = {"Yes": 1, "No": 0}

    # Feature engineering (matching training data encoding)
    if monthly_income <= 15000:
        income_category = 0  # Low
    elif monthly_income <= 25000:
        income_category = 1  # Medium
    elif monthly_income <= 35000:
        income_category = 2  # High
    else:
        income_category = 3  # Very High

    if age <= 25:
        age_group = 0  # Young
    elif age <= 35:
        age_group = 1  # Adult
    elif age <= 45:
        age_group = 2  # Middle-aged
    elif age <= 55:
        age_group = 3  # Senior
    else:
        age_group = 4  # Elderly

    # Create input array
    input_array = np.array([[
        age, contact_map[type_of_contact], city_tier, duration_of_pitch,
        occupation_map[occupation], gender_map[gender], num_person_visiting,
        num_followups, product_map[product_pitched], preferred_property_star,
        marital_map[marital_status], num_trips, passport_map[passport],
        pitch_satisfaction_score, car_map[own_car], num_children_visiting,
        designation_map[designation], monthly_income, income_category, age_group
    ]])

    return input_array

def main():
    """Main Streamlit app"""

    st.title("Tourism Package Prediction")
    st.markdown("### Predict Customer Purchase Likelihood for Wellness Tourism Package")
    st.markdown("---")

    # Load model
    model = load_model()
    #if model is None:
    #    st.error("Failed to load the prediction model.")
    #    return

    # Sidebar inputs
    st.sidebar.header("Customer Information")

    # Demographics
    st.sidebar.subheader("Demographics")
    age = st.sidebar.slider("Age", 18, 80, 35)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])

    # Location & Contact
    st.sidebar.subheader("Location & Contact")
    city_tier = st.sidebar.selectbox("City Tier", [1, 2, 3])
    type_of_contact = st.sidebar.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])

    # Professional Info
    st.sidebar.subheader("Professional Info")
    occupation = st.sidebar.selectbox("Occupation", ["Salaried", "Small Business", "Free Lancer"])
    designation = st.sidebar.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.sidebar.number_input("Monthly Income", 10000, 50000, 20000)

    # Travel Preferences
    st.sidebar.subheader("Travel Preferences")
    num_person_visiting = st.sidebar.slider("Number of Persons Visiting", 1, 5, 2)
    num_children_visiting = st.sidebar.slider("Number of Children Visiting", 0, 3, 0)
    preferred_property_star = st.sidebar.slider("Preferred Property Star Rating", 1.0, 5.0, 3.0, 0.5)
    num_trips = st.sidebar.slider("Number of Trips per Year", 0.0, 10.0, 2.0, 0.5)

    # Additional Info
    st.sidebar.subheader("Additional Info")
    passport = st.sidebar.selectbox("Has Passport", ["Yes", "No"])
    own_car = st.sidebar.selectbox("Owns Car", ["Yes", "No"])

    # Sales Interaction
    st.sidebar.subheader("Sales Interaction")
    duration_of_pitch = st.sidebar.slider("Duration of Pitch (minutes)", 5, 60, 15)
    product_pitched = st.sidebar.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe"])
    num_followups = st.sidebar.slider("Number of Followups", 0.0, 6.0, 3.0, 0.5)
    pitch_satisfaction_score = st.sidebar.slider("Pitch Satisfaction Score", 1, 5, 3)

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Customer Profile Summary")
        profile_data = {
            "Age": age,
            "Gender": gender,
            "Marital Status": marital_status,
            "City Tier": city_tier,
            "Occupation": occupation,
            "Monthly Income": f"₹{monthly_income:,}",
            "Number of Persons": num_person_visiting,
            "Preferred Star Rating": preferred_property_star,
            "Annual Trips": num_trips,
            "Has Passport": passport,
            "Owns Car": own_car
        }

        for key, value in profile_data.items():
            st.write(f"**{key}:** {value}")

    with col2:
        st.subheader("Prediction")

        if st.button("Predict Purchase Likelihood", type="primary"):
            input_data = prepare_input_data(
                age, gender, marital_status, city_tier, type_of_contact,
                occupation, designation, monthly_income, num_person_visiting,
                num_children_visiting, preferred_property_star, num_trips,
                passport, own_car, duration_of_pitch, product_pitched,
                num_followups, pitch_satisfaction_score
            )

            try:
                prediction = model.predict(input_data)[0]
                prediction_proba = model.predict_proba(input_data)[0]

                if prediction == 1:
                    st.success("High likelihood of purchase!")
                    st.write(f"**Confidence:** {prediction_proba[1]:.2%}")
                    st.balloons()
                else:
                    st.warning("Low likelihood of purchase")
                    st.write(f"**Confidence:** {prediction_proba[0]:.2%}")

                # Probability breakdown
                st.subheader("Probability Breakdown")
                prob_df = pd.DataFrame({
                    'Outcome': ['Will Not Purchase', 'Will Purchase'],
                    'Probability': [prediction_proba[0], prediction_proba[1]]
                })
                st.bar_chart(prob_df.set_index('Outcome'))

            except Exception as e:
                st.error(f"Prediction error: {e}")

    st.markdown("...")
    st.markdown("### About This Model")
    st.info("""
    This ML model predicts customer purchase likelihood for the Wellness Tourism Package
    based on demographics, travel preferences, and sales interaction data.
    """)

if __name__ == "__main__":
    main()
