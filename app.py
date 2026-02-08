import streamlit as st
import numpy as pd
import pickle
with open('svm_car_type_model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
engine_size = st.number_input('Engine Size (in CC)', min_value=500, max_value=5000, value=1500, step=100)
seats=st.selectbox('Number of Seats', options=[2, 4, 5, 6])
if st.button('Predict Car Type'):
    input_data = np.array([[engine_size, seats]])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    if prediction[0] == 1:
        st.success('The car is likely to be a SUV.')
        st.balloons()
    else:
        st.success('The car is likely to be a Sedan.')
        st.snow()
