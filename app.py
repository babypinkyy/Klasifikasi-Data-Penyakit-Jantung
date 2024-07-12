import streamlit as st
import pickle
import numpy as np

# Load the trained models
with open('knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)

with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
    
with open('xgb_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

with open('knn_model_normalisasi.pkl', 'rb') as f:
    knn_model_normalisasi = pickle.load(f)
    
with open('rf_model_normalisasi.pkl', 'rb') as f:
    rf_model_normalisasi = pickle.load(f)

with open('xgb_model_normalisasi.pkl', 'rb') as f:
    xgb_model_normalisasi = pickle.load(f)

# Application title
st.title('Heart Disease Prediction')

# Sidebar for selecting model
st.sidebar.header('Select Model')
model_option = st.sidebar.selectbox(
    'Choose a model', 
    (
        'KNN Model Tanpa Normalisasi', 
        'Random Forest Model Tanpa Normalisasi', 
        'XGBoost Model Tanpa Normalisasi', 
        'KNN Model Dengan Normalisasi', 
        'Random Forest Model Dengan Normalisasi', 
        'XGBoost Model Dengan Normalisasi'
    )
)

# Input form
st.header('Input Patient Data')
age = st.number_input('Age', min_value=0, max_value=120, value=30)
sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
cp = st.selectbox('Chest Pain Type (CP)', options=[1, 2, 3, 4], format_func=lambda x: {
    1: 'Typical Angina', 
    2: 'Atypical Angina', 
    3: 'Non-anginal Pain', 
    4: 'Asymptomatic'
}[x])
trestbps = st.number_input('Resting Blood Pressure (trestbps)', min_value=0, max_value=300, value=120)
chol = st.number_input('Serum Cholestoral in mg/dl (chol)', min_value=0, max_value=600, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', options=[0, 1], format_func=lambda x: 'False' if x == 0 else 'True')
restecg = st.number_input('Resting Electrocardiographic Results (restecg)', min_value=0, max_value=2, value=1)
thalach = st.number_input('Maximum Heart Rate Achieved (thalach)', min_value=0, max_value=250, value=150)
exang = st.selectbox('Exercise Induced Angina (exang)', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
oldpeak = st.number_input('ST Depression Induced by Exercise (oldpeak)', min_value=0.0, max_value=10.0, value=1.0)
# slope = st.number_input('Slope of the Peak Exercise ST Segment (slope)', min_value=0, max_value=2, value=1)
# ca = st.number_input('Number of Major Vessels Colored by Fluoroscopy (ca)', min_value=0, max_value=4, value=0)
# thal = st.number_input('Thalassemia (thal)', min_value=0, max_value=3, value=2)

# Prepare the input data as a numpy array
input_data = np.array([[
    age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak
]])

# Prediction button
if st.button('Predict'):
    if model_option == 'KNN Model Tanpa Normalisasi':
        prediction = knn_model.predict(input_data)
    elif model_option == 'Random Forest Model Tanpa Normalisasi':
        prediction = rf_model.predict(input_data)
    elif model_option == 'XGBoost Model Tanpa Normalisasi':
        prediction = xgb_model.predict(input_data)
    elif model_option == 'KNN Model Dengan Normalisasi':
        prediction = knn_model_normalisasi.predict(input_data)
    elif model_option == 'Random Forest Model Dengan Normalisasi':
        prediction = rf_model_normalisasi.predict(input_data)
    elif model_option == 'XGBoost Model Dengan Normalisasi':
        prediction = xgb_model_normalisasi.predict(input_data)
    
    # Display the prediction result
    st.write(f'Prediction value: {prediction[0]}')
    st.header('Prediction Result')
    if prediction[0] == 1:
        st.write('Pasien Terindikasi Penyakit Jantung Ringan')
    elif prediction[0] == 2:
        st.write('Pasien Terindikasi Penyakit Jantung Sedang')
    elif prediction[0] == 3:
        st.write('asien Terindikasi Penyakit Jantung Parah')
    elif prediction[0] == 4:
        st.write('asien Terindikasi Penyakit Jantung Sangat Parah')
    else:
        st.write('Pasien Tidak Terindikasi Penyakit Jantung')
