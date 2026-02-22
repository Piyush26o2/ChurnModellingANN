import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle 

# Load trained models
with open('models/ohe.pkl', 'rb') as file:
    ohe_geo = pickle.load(file)

with open('models/label_encoder_gender.pkl', 'rb') as file:
    le_gender = pickle.load(file)

with open('models/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit App
st.title("Customer Churn Prediction")

# Geography dropdown (using OneHotEncoder categories)
geography = st.selectbox(
    "Geography",
    ohe_geo.categories_[0]
)

# Gender dropdown (using LabelEncoder classes)
gender = st.selectbox(
    "Gender",
    le_gender.classes_
)

# Numeric inputs
age = st.slider("Age", 18, 90)
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)

balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")

# Binary inputs
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1]) 

input_data = pd.DataFrame({
    'CreditScore' : [credit_score],
    'Gender': [le_gender.transform([gender])[0]],
    'Age' : [age],
    'Tenure': [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active_member],
    'EstimatedSalary' : [estimated_salary],
})

# One Hot Encode Geography
geo_encoded = ohe_geo.transform(pd.DataFrame({"Geography": [geography]}))
geo_df=pd.DataFrame(geo_encoded, columns=ohe_geo.get_feature_names_out(["Geography"]) )
input_df = pd.concat([input_data.reset_index(drop=True), geo_df], axis=1)

# Scale the input data
num_cols = ["CreditScore","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]
input_df[num_cols] = scaler.transform(input_df[num_cols])

# Converting into tensors
device = "cuda" if torch.cuda.is_available() else "cpu"
input_tensor = torch.tensor(input_df.values, dtype=torch.float32).to(device)

# Prediction
model = nn.Sequential(
    nn.Linear(12, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1) 
)

model.load_state_dict(torch.load(r"models/best_model.pt", map_location=device))
model.to(device)
model.eval()

with torch.no_grad():
    output = model(input_tensor)
    probability = torch.sigmoid(output)
    prediction = (probability > 0.5).float()

st.write(f'Churn Probability : {probability.item():.2f}')

if prediction.item()==1 :
    print("The customer is likely to churn") 
else : 
    print("The customer is not likely to churn")

