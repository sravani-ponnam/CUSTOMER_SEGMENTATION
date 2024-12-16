import streamlit as st
import xgboost as xgb
import pandas as pd
import datetime

# Load the trained model
model = xgb.Booster()
model.load_model('your_model.json')  # Ensure the model path is correct

# Define mapping dictionaries for categorical features
manufacturer_mapping = {
    "gmc": 0, "chevrolet": 1, "toyota": 2, "ford": 3, "jeep": 4,
    "nissan": 5, "ram": 6, "mazda": 7, "cadillac": 8, "honda": 9,
    "dodge": 10, "lexus": 11, "jaguar": 12, "buick": 13, "chrysler": 14,
    "volvo": 15, "audi": 16, "infiniti": 17, "lincoln": 18, "alfa-romeo": 19,
    "subaru": 20, "acura": 21, "hyundai": 22, "mercedes-benz": 23,
    "bmw": 24, "mitsubishi": 25, "volkswagen": 26, "porsche": 27,
    "kia": 28, "rover": 29, "ferrari": 30, "mini": 31, "pontiac": 32,
    "fiat": 33, "tesla": 34, "saturn": 35, "mercury": 36, "harley-davidson": 37
}

region_mapping = {
    "prescott": 0, "fayetteville": 1, "florida keys": 2, "worcester / central MA": 3,
    "greensboro": 4, "hudson valley": 5, "medford-ashland": 6, "erie": 7,
    "el paso": 8, "bellingham": 9, "skagit / island / SJI": 10, "la crosse": 11,
    "auburn": 12, "birmingham": 13, "dothan": 14, "florence / muscle shoals": 15,
    "gadsden-anniston": 16, "huntsville / decatur": 17, "mobile": 18, "montgomery": 19,
    "tuscaloosa": 20, "anchorage / mat-su": 21
}

fuel_mapping = {
    "Gasoline": 0, "Diesel": 1, "Electric": 2, "Hybrid": 3
}

condition_mapping = {
    "New": 0, "Like New": 1, "Used": 2, "Fair": 3, "Salvage": 4
}

title_status_mapping = {
    "clean": 0, "rebuilt": 1, "lien": 2, "salvage": 3, "missing": 4, "parts only": 5
}

transmission_mapping = {
    "other": 0, "automatic": 1, "manual": 2
}

drive_mapping = {
    "rwd": 0, "4wd": 1, "fwd": 2
}

size_mapping = {
    "full-size": 0, "mid-size": 1, "compact": 2, "sub-compact": 3
}

type_mapping = {
    "pickup": 0, "truck": 1, "other": 2, "coupe": 3, "SUV": 4,
    "hatchback": 5, "mini-van": 6, "sedan": 7, "offroad": 8,
    "bus": 9, "van": 10, "convertible": 11, "wagon": 12
}

paint_color_mapping = {
    "white": 0, "blue": 1, "red": 2, "black": 3, "silver": 4, "grey": 5,
    "brown": 6, "yellow": 7, "orange": 8, "green": 9, "custom": 10, "purple": 11
}

state_mapping = {
    "az": 0, "ar": 1, "fl": 2, "ma": 3, "nc": 4, "ny": 5,
    "or": 6, "pa": 7, "tx": 8, "wa": 9, "wi": 10, "al": 11, "ak": 12
}

# User input for prediction
st.title("Vehicle Price Prediction Dashboard")
st.sidebar.header("Input Features")

region = st.sidebar.selectbox("Region", options=list(region_mapping.keys()))
year = st.sidebar.number_input("Year of Vehicle", min_value=1900, max_value=2024, value=2020)
manufacturer = st.sidebar.selectbox("Manufacturer", options=list(manufacturer_mapping.keys()))
model_input = st.sidebar.number_input("Model (encoded)", min_value=0, value=0)
condition = st.sidebar.selectbox("Condition", options=list(condition_mapping.keys()))
cylinders = st.sidebar.number_input("Cylinders", min_value=0, value=4)
fuel = st.sidebar.selectbox("Fuel Type", options=list(fuel_mapping.keys()))
odometer = st.sidebar.number_input("Odometer Reading (in miles)", min_value=0, value=50000)
title_status = st.sidebar.selectbox("Title Status", options=list(title_status_mapping.keys()))
transmission = st.sidebar.selectbox("Transmission", options=list(transmission_mapping.keys()))
drive = st.sidebar.selectbox("Drive", options=list(drive_mapping.keys()))
size = st.sidebar.selectbox("Size", options=list(size_mapping.keys()))
type_input = st.sidebar.selectbox("Type", options=list(type_mapping.keys()))
paint_color = st.sidebar.selectbox("Paint Color", options=list(paint_color_mapping.keys()))
state = st.sidebar.selectbox("State", options=list(state_mapping.keys()))
posting_date = st.sidebar.number_input("Posting Date (encoded)", min_value=0, value=0)

# Convert human-readable inputs to encoded values
encoded_region = region_mapping[region]
encoded_manufacturer = manufacturer_mapping[manufacturer]
encoded_condition = condition_mapping[condition]
encoded_fuel = fuel_mapping[fuel]
encoded_title_status = title_status_mapping[title_status]
encoded_transmission = transmission_mapping[transmission]
encoded_drive = drive_mapping[drive]
encoded_size = size_mapping[size]
encoded_type = type_mapping[type_input]
encoded_paint_color = paint_color_mapping[paint_color]
encoded_state = state_mapping[state]

# Current year for calculating car_age
current_year = datetime.datetime.now().year

# Create a DataFrame for the input features
input_data = pd.DataFrame([[encoded_region, year, encoded_manufacturer, model_input, encoded_condition, cylinders, 
                             encoded_fuel, odometer, encoded_title_status, encoded_transmission, encoded_drive, 
                             encoded_size, encoded_type, encoded_paint_color, encoded_state, posting_date]], 
                          columns=['region', 'year', 'manufacturer', 'model', 'condition', 'cylinders', 
                                   'fuel', 'odometer', 'title_status', 'transmission', 'drive', 
                                   'size', 'type', 'paint_color', 'state', 'posting_date'])

# Calculate car_age and add to the DataFrame
input_data['car_age'] = current_year - input_data['year']

# Make prediction when the button is pressed
if st.button("Predict"):
    dmatrix = xgb.DMatrix(input_data)  # Create DMatrix, required format for XGBoost
    try:
        prediction = model.predict(dmatrix)
        st.write(f"Predicted Price: ${prediction[0]:,.2f}")  # Display the prediction result
    except ValueError as e:
        st.error(f"Error during prediction: {e}")

# Optional sections for model explanation and additional resources
st.sidebar.header("Model Explanation")
st.sidebar.write("This model predicts the selling price of a vehicle based on features like year, manufacturer, fuel type, condition, title status, transmission, drive, size, type, paint color, state, and posting date.")

st.sidebar.header("Additional Resources")
st.sidebar.write("For more information about the model and its features, please refer to the documentation or contact support.")

st.sidebar.write("Developed by MANEESHA PALEM, SRAVANI PONNAM, CHANDU KATIPALLY")
