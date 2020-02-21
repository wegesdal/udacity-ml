
#%%

points = [(2, -2), (5, 6), (-4, -4), (-7, 1), (8, 14)]

sum_mean_abs_error = 0
sum_mean_squared_error = 0

for point in points:
    x = point[0]
    y = point[1]
    y_hat = 1.2 * x + 2
    sum_mean_abs_error = sum_mean_abs_error + abs(y - y_hat)
    sum_mean_squared_error = sum_mean_squared_error + pow(y - y_hat, 2)


mean_abs_error = sum_mean_abs_error / len(points)
mean_squared_error = sum_mean_squared_error / (len(points) * 2)

print ('mean abs error:' + str(mean_abs_error))
print ('mean squared error:' + str(mean_squared_error))

#%%
import numpy as np

learn_rate = 0.01
X = [2, 5, -4, -7, 8]
y = [-2, 6, -4, 1, 14]
W = [1.2]
b = 2

y_hat = np.multiply(W, X)
y_hat += b

error = y - y_hat

W_new = W + learn_rate * np.matmul(error, X)
b_new = b + learn_rate * error.sum() 

print(W_new)
print(b_new)