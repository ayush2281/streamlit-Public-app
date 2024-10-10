# -*- coding: utf-8 -*-

"""
Created on Wed Oct  9 19:44:23 2024

@author: lenovo
"""
import streamlit as st
import numpy as np
import pickle 

# Loading the saved model
loaded_model = pickle.load(open(r'C:\Users\lenovo\OneDrive\Desktop\deploy ML-Model(API)\TrafficPrediction_Proj\Deployed Using Streamlit\trained_model.sav', 'rb'))

# Prediction system
def predict_traffic(input_data):
    # Convert the input data into a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Make prediction
    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return "Traffic is Normal"
    else:
        return "Traffic is Low"

def main():
    # Giving a title 
    st.title('Traffic Prediction Web App')
    
    # Getting the input data from the user
    Date = st.text_input("Enter the Date")
    Day_of_the_week = st.text_input("Enter the Day of the week")
    CarCount = st.text_input("Enter the Car Count")
    BikeCount = st.text_input("Enter the Bike Count")
    BusCount = st.text_input("Enter the Bus Count")
    TruckCount = st.text_input("Enter the Truck Count")
    Total = st.text_input("Enter the Total")

    # Code for prediction
    Traffic_Situation = ''
    
    # Creating the button for prediction
    if st.button('Click To Predict the Traffic Situation'):
        try:
            # Convert inputs to appropriate types
            input_data = [int(Date), int(Day_of_the_week), int(CarCount), int(BikeCount), int(BusCount), int(TruckCount), int(Total)]
            Traffic_Situation = predict_traffic(input_data)
            st.success(Traffic_Situation)
        except ValueError:
            st.error("Please enter valid numeric values for all inputs.")

if __name__ == '__main__':
    main()
