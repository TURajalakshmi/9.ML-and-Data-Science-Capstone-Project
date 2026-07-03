import streamlit as st
import pandas as pd
import pickle

# Load deployment model
loaded_model = pickle.load(
    open("house_price_deploy_model.sav", "rb")
)

st.title("🏠 House Price Prediction")

st.write("📋Enter house details below.")

# User inputs
overall = st.number_input("Overall Quality", min_value=1, max_value=10, value=5)

year = st.number_input("Year Built", min_value=1800, max_value=2100, value=2000)

bsmtfin = st.number_input("Finished Basement Area (BsmtFinSF1)", min_value=0.0)

basement = st.number_input("Total Basement Area", min_value=0.0)

area = st.number_input("Ground Living Area (GrLivArea)", min_value=0.0)

garage = st.number_input("Garage Cars", min_value=0)

garagearea = st.number_input("Garage Area", min_value=0.0)

# Predict button
if st.button("💰Predict Price"):

    data = pd.DataFrame({
        "OverallQual": [overall],
        "YearBuilt": [year],
        "BsmtFinSF1": [bsmtfin],
        "TotalBsmtSF": [basement],
        "GrLivArea": [area],
        "GarageCars": [garage],
        "GarageArea": [garagearea]
    })

    prediction = loaded_model.predict(data)

    st.success(f"Predicted House Price: ${prediction[0]:,.2f}")