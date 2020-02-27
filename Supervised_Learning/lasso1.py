# Add import statements
from sklearn.linear_model import Lasso
import pandas
import numpy as np

# Assign the data to predictor and outcome variables
# Load the data
train_data = pandas.read_csv('data3.csv', header = None)

X = train_data.iloc[:,0:6]
y = train_data.iloc[:,6]

print(X)
print(y)

# Create the linear regression model with lasso regularization.
lasso_reg = Lasso()

# Fit the model.
lasso_reg.fit(X, y)

# Retrieve and print out the coefficients from the regression model.
reg_coef = lasso_reg.coef_
print(reg_coef)
