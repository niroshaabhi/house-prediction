import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("🏠 Bangalore House Price Prediction")

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("bengaluru_house_prices.csv")

# -----------------------------
# Data Cleaning
# -----------------------------

# Convert size → BHK
df["BHK"] = df["size"].str.split().str[0]
df["BHK"] = pd.to_numeric(df["BHK"], errors="coerce")

# Convert total_sqft (handle ranges like 2100-2850)
def convert_sqft(x):
    try:
        if "-" in str(x):
            tokens = x.split("-")
            return (float(tokens[0]) + float(tokens[1])) / 2
        return float(x)
    except:
        return None

df["total_sqft"] = df["total_sqft"].apply(convert_sqft)

# Drop unnecessary columns
df = df.drop(["size", "society", "availability"], axis=1)

# Remove missing values
df = df.dropna()

# -----------------------------
# Features & Target
# -----------------------------
X = df.drop("price", axis=1)
y = df["price"]

categorical_cols = ["area_type", "location"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough"
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Train Model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# User Input Section
# -----------------------------
st.subheader("Enter House Details")

area_type = st.selectbox("Area Type", df["area_type"].unique())
location = st.selectbox("Location", df["location"].unique())
total_sqft = st.number_input("Total Square Feet", min_value=300.0)
bath = st.number_input("Bathrooms", min_value=1)
balcony = st.number_input("Balconies", min_value=0)
bhk = st.number_input("BHK", min_value=1)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Price"):
    input_df = pd.DataFrame([{
        "area_type": area_type,
        "location": location,
        "total_sqft": total_sqft,
        "bath": bath,
        "balcony": balcony,
        "BHK": bhk
    }])
    prediction = model.predict(input_df)        # indent with 4 spaces
    predicted_price = max(prediction[0], 5.0)   # indent with 4 spaces
    st.success(f"Estimated Price: ₹ {predicted_price:.2f} Lakhs")  # indent with 4 spaces



