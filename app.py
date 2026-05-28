import streamlit as st
import pickle
import numpy as np

# Model aur Scaler load karo
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


# App Title
st.title("California Housing Price Predictor")
st.write("Ghar ki details dalo aur price predict karo!")

# User Inputs
median_income = st.slider("Median Income", min_value=0.0, max_value=15.0, value=5.0)
housing_median_age = st.slider("Housing Median Age", min_value=1, max_value=52, value=25)
total_rooms = st.number_input("Total Rooms", min_value=1, max_value=10000, value=2000)
total_bedrooms = st.number_input("Total Bedrooms", min_value=1, max_value=3000, value=400)
population = st.number_input("Population", min_value=1, max_value=35000, value=1000)
households = st.number_input("Households", min_value=1, max_value=6000, value=400)
ocean_proximity = st.selectbox("Ocean Proximity", 
    ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])

# Feature Engineering
rooms_per_household = total_rooms / households
bedrooms_per_room = total_bedrooms / total_rooms
population_per_household = population / households

# Encoding
ocean_proximity_encoded = {
    '<1H OCEAN': [1, 0, 0, 0, 0],
    'INLAND':    [0, 1, 0, 0, 0],
    'ISLAND':    [0, 0, 1, 0, 0],
    'NEAR BAY':  [0, 0, 0, 1, 0],
    'NEAR OCEAN':[0, 0, 0, 0, 1]
}

# Predict Button
if st.button("Predict Price"):
    ocean = ocean_proximity_encoded[ocean_proximity]
    
    input_data = np.array([[
        0, 0, housing_median_age, total_rooms, total_bedrooms,
        population, households, median_income,
        rooms_per_household, bedrooms_per_room, population_per_household,
        ocean[0], ocean[1], ocean[2], ocean[3], ocean[4]
    ]])
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    
    st.success(f"Predicted House Price: ${prediction[0]:,.2f}")

