# real-estate_project
"A machine learning project to predict property prices in Gurgaon using Linear Regression. Built as part of Code With Harry's Data Science course."
# 🏡 Gurgaon Real Estate Price Prediction

This project uses machine learning to predict property prices in **Gurgaon**, India based on features like location, area (sqft), number of bedrooms (BHK), and ocean_proximity. Built as part of the **Data Science Course by Code With Harry**, this project demonstrates data preprocessing, model training using **Linear Regression**, **DecisionTreeRegressor**, **RandomForestRegressor** and a custom prediction function.

---

## 📌 Project Goals

- Predict property prices using supervised machine learning.
- Perform data cleaning, preprocessing, and outlier removal.
- Train a regression model and evaluate its performance.
- Create a function to make predictions based on user input.

---

## 📂 Dataset

The dataset includes:

- `longitude`
- `latitude`
- `housing_median_age`
- `total_rooms`
- `total_bedrooms`
- `population`
- `households`
- `median_income`
- `median_house_value`
- `ocean_proximity`

---

## 🔧 Technologies Used

- **Python**
- **Jupyter Notebook**
- **pandas, numpy** – Data manipulation
- **matplotlib, seaborn** – Visualization
- **scikit-learn** – Model training
- **joblib** – Model saving for deployment

---

## 🧹 Data Preprocessing

- Converted non-numeric columns to numeric using dummy variables.
- Removed entries with missing or inconsistent data.
- Identified and removed outliers based on BHK-to-bathroom ratio and price-per-sqft.
- Encoded categorical locations into one-hot vectors.

---

## 🤖 Model Training

- Used **Linear Regression** from `scikit-learn`.
- Evaluated performance using **R² Score** and cross-validation.
- Tuned the model for best generalization on unseen data.

---
