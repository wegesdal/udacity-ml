import numpy as np
import pandas
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

train_data = pandas.read_csv('data4.csv', header = None)
X = train_data.iloc[:,0:6]
y = train_data.iloc[:, 6]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

lasso_reg = Lasso()

lasso_reg.fit(X_scaled, y)

ref_coef = lasso_reg.coef_
print(ref_coef)
