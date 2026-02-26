Bangalore House Price Prediction using Machine Learning
📌 Project Overview

This project predicts house prices in Bangalore based on various features such as area type, location, total square feet, number of bathrooms, balconies, and BHK configuration.

The system uses Linear Regression to estimate property prices and is deployed as an interactive web application using streamlit.

🚀 Technologies Used

Python

pandas – Data processing

numpy – Numerical operations

scikit-learn – Machine learning model

streamlit – Web application deployment

📊 Dataset

Dataset used: bengaluru_house_prices.csv

Key features:

area_type

location

size

total_sqft

bath

balcony

price

🔎 Project Workflow

Data Cleaning

Converted "size" column to BHK

Handled range values in total_sqft

Removed missing values

Dropped unnecessary columns

Feature Engineering

Applied OneHotEncoding for categorical features (area_type, location)

Model Training

Used Linear Regression algorithm

Split data into training and testing sets

Evaluated using R² Score

Deployment

Built interactive UI using Streamlit

Users can enter house details and get predicted price instantly

💡 Key Features

✔ Clean and structured ML pipeline
✔ Categorical data handling using OneHotEncoder
✔ Interactive web-based prediction system
✔ Real-time price estimation

▶ How to Run the Project
pip install -r requirements.txt
streamlit run app.py

Then open:

http://localhost:8501
🎯 Conclusion

This project demonstrates end-to-end Machine Learning workflow including:

Data preprocessing

Model building

Evaluation

Deployment as a web application

It showcases practical implementation of regression techniques for real-world price prediction problems.
