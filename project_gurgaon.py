import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score


# 1. Load the DataSet
housing = pd.read_csv("housing.csv")

# 2. Create a stratified test set
housing['income_cat'] = pd.cut(housing['median_income'], 
                               bins = [0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                               labels = [1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)

for train_index, test_index in split.split(housing, housing['income_cat']):
    strate_train_set = housing.loc[train_index].drop("income_cat", axis = 1)
    strate_test_set = housing.loc[test_index].drop("income_cat", axis = 1)

# we will work on copy of training data
housing = strate_train_set.copy()

# 3. Separate features and labels
housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis = 1)

# print(housing, housing_labels)

# 4. List the numarical and categorical columns

num_attribs = housing.drop("ocean_proximity", axis = 1).columns.tolist()
cat_attribs = ["ocean_proximity"]

# 5. lets make the pipeline

# For numerical columns
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy = "median")),
    ("scaler", StandardScaler())
])

# For categorical columns
cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# construct the full pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])

# 6. Transform the data
housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared.shape)

# 7. Train the model

# Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_preds = lin_reg.predict(housing_prepared)
lin_rmse = root_mean_squared_error(housing_labels, lin_preds)
print(f"The root mean squared error for linear regression is {lin_rmse}")

# Decision Tree model
dec_reg = DecisionTreeRegressor()
dec_reg.fit(housing_prepared, housing_labels)
dec_preds = dec_reg.predict(housing_prepared)
# des_rmse = root_mean_squared_error(housing_labels, dec_preds)
dec_rmse = -cross_val_score(dec_reg, housing_prepared, housing_labels, scoring = "neg_root_mean_squared_error", cv = 10)
# print(f"The root mean squared error for Decision Tree regressor is {dec_rmse}")
print(pd.Series(dec_rmse).describe())

# Random Forest model
random_forest_reg = RandomForestRegressor()
random_forest_reg.fit(housing_prepared, housing_labels)
random_forest_preds = random_forest_reg.predict(housing_prepared)
random_forest_rmse = root_mean_squared_error(housing_labels, random_forest_preds)
# print(f"The root mean squared error for Random Forest regressor is {random_forest_rmse}")
random_forest_rmse = -cross_val_score(random_forest_reg, housing_prepared, housing_labels, scoring = "neg_root_mean_squared_error", cv = 10)
# print(f"The root mean squared error for Decision Tree regressor is {random_forest_rmse}")
print(pd.Series(random_forest_rmse).describe())