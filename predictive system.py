# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 13:05:43 2024

@author: lenovo
"""

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

# Example input data
input_data = (10, 1, 31, 0, 4, 4, 39)  # Example values for Date, Day of the week, CarCount, BusCount, TruckCount

# Make a prediction
result = predict_traffic(input_data)
print(result)