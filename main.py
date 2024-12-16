import streamlit as st
import xgboost as xgb
import pandas as pd


# Load the trained model
model = xgb.Booster()
model.load_model('your_model.json')

# User input for prediction
st.title("XGBoost Prediction Dashboard")

# Example input fields for features
st.sidebar.header("Input Features")
feature1 = st.sidebar.number_input("Feature 1 (e.g., Year)", min_value=1900, max_value=2024, value=2020)
feature2 = st.sidebar.number_input("Feature 2 (e.g., Odometer)", min_value=0, value=50000)
feature3 = st.sidebar.number_input("Feature 3 (e.g., Engine Size)", min_value=0.0, value=2.0)

# Add more features as needed
feature4 = st.sidebar.number_input("Feature 4 (e.g., Number of Doors)", min_value=1, max_value=5, value=4)
feature5 = st.sidebar.selectbox("Feature 5 (e.g., Fuel Type)", options=["Gasoline", "Diesel", "Electric"])

# Create a DataFrame for the input features
input_data = pd.DataFrame([[feature1, feature2, feature3, feature4, feature5]], 
                          columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])

# Make prediction
if st.button("Predict"):
    # Convert categorical features to numerical if necessary
    input_data['feature5'] = input_data['feature5'].map({"Gasoline": 0, "Diesel": 1, "Electric": 2})  # Example mapping

    dmatrix = xgb.DMatrix(input_data)
    prediction = model.predict(dmatrix)
    
    # Display the prediction result
    st.write(f"**Predicted Price:** ${prediction[0]:,.2f}")

# Optional: Add a section to explain the model
st.sidebar.header("Model Explanation")
st.sidebar.write("This model predicts the selling price of a vehicle based on various features such as year, odometer reading, engine size, number of doors, and fuel type.")

# Optional: Add a section for additional information or resources
st.sidebar.header("Additional Resources")
st.sidebar.write("For more information about the model and its features, please refer to the documentation or contact support.")

# Optional: Add a footer
st.sidebar.write("Developed by [Your Name] - [Your Contact Information]")