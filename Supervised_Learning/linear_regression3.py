# Add import statements
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas

# Assign the data to predictor and outcome variables
# Load the data
train_data = pandas.read_csv('data2.csv')
X = np.reshape([train_data['Var_X']], (20, 1))
y = train_data['Var_Y']
print(X)
print(X.shape)

# Create polynomial features
# Create a PolynomialFeatures object, then fit and transform the
# predictor feature
poly_feat = PolynomialFeatures(degree = 3)
X_poly = poly_feat.fit_transform(X)

# Make and fit the polynomial regression model
# Create a LinearRegression object and fit it to the polynomial predictor
# features
poly_model = LinearRegression()
poly_model.fit(X_poly, y)

# Once you've completed all of the steps, select Test Run to see your model
# predictions against the data, or select Submit Answer to check if the degree
# of the polynomial features is the same as ours!