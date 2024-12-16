import streamlit as st
import xgboost as xgb
import pandas as pd

# Load the trained model
model = xgb.Booster()
model.load_model('your_model.json')  # Ensure the model path is correct

# Define mapping dictionaries for categorical features
manufacturer_mapping = {
    "Toyota": 0, "Honda": 1, "Ford": 2, "Chevrolet": 3, "Nissan": 4,
    # Add more manufacturers as needed
}

fuel_mapping = {
    "Gasoline": 0, "Diesel": 1, "Electric": 2,
    # Add more fuel types as needed
}

condition_mapping = {
    "New": 0, "Like New": 1, "Used": 2, "Fair": 3, "Salvage": 4,
    # Add more conditions as needed
}

# User input for prediction
st.title("Vehicle Price Prediction Dashboard")
st.sidebar.header("Input Features")
region = st.sidebar.number_input("Region", min_value=0, value=0)
year = st.sidebar.number_input("Year of Vehicle", min_value=1900, max_value=2024, value=2020)
manufacturer = st.sidebar.selectbox("Manufacturer", options=list(manufacturer_mapping.keys()))
model_input = st.sidebar.number_input("Model (encoded)", min_value=0, value=0)  # Assuming model is still encoded
condition = st.sidebar.selectbox("Condition", options=list(condition_mapping.keys()))
cylinders = st.sidebar.number_input("Cylinders", min_value=0, value=4)
fuel = st.sidebar.selectbox("Fuel Type", options=list(fuel_mapping.keys()))
odometer = st.sidebar.number_input("Odometer Reading (in miles)", min_value=0, value=50000)
title_status = st.sidebar.number_input("Title Status (encoded)", min_value=0, value=0)
transmission = st.sidebar.number_input("Transmission (encoded)", min_value=0, value=0)
drive = st.sidebar.number_input("Drive (encoded)", min_value=0, value=0)
size = st.sidebar.number_input("Size (encoded)", min_value=0, value=0)
type_input = st.sidebar.number_input("Type (encoded)", min_value=0, value=0)
paint_color = st.sidebar.number_input("Paint Color (encoded)", min_value=0, value=0)
state = st.sidebar.number_input("State (encoded)", min_value=0, value=0)
posting_date = st.sidebar.number_input("Posting Date (encoded)", min_value=0, value=0)

# Convert human-readable inputs to encoded values
encoded_manufacturer = manufacturer_mapping[manufacturer]
encoded_condition = condition_mapping[condition]
encoded_fuel = fuel_mapping[fuel]

# Create a DataFrame for the input features
input_data = pd.DataFrame([[region, year, encoded_manufacturer, model_input, encoded_condition, cylinders, 
                             encoded_fuel, odometer, title_status, transmission, drive, size, 
                             type_input, paint_color, state, posting_date]], 
                          columns=['region', 'year', 'manufacturer', 'model', 'condition', 'cylinders', 
                                   'fuel', 'odometer', 'title_status', 'transmission', 'drive', 
                                   'size', 'type', 'paint_color', 'state', 'posting_date'])

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
st.sidebar.write("This model predicts the selling price of a vehicle based on various features such as year, manufacturer, condition, fuel type, odometer reading, and more.")

st.sidebar.header("Additional Resources")
st.sidebar.write("For more information about the model and its features, please refer to the documentation or contact support.")

st.sidebar.write("Developed by [Your Name] - [Your Contact Information]")
