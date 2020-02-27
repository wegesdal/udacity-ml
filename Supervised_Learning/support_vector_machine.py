# %%

# support vector machines in sklearn

# Import statements 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Read the data.
data = np.asarray(pd.read_csv('data7.csv', header=None))
# Assign the features to the variable X, and the labels to the variable y. 
X = data[:,0:2]
y = data[:,2]


best_fit = ()
best_score = 0

for i in range(1, 10):
    for j in range(1, 10):
        # Create the model and assign it to the variable model.
        # Find the right parameters for this model to achieve 100% accuracy on the dataset.
        model = SVC(kernel='rbf', C=i, gamma=j)
        # Fit the model.
        model.fit(X, y)
        # Make predictions. Store them in the variable y_pred.
        y_pred = model.predict(X)
        # Calculate the accuracy and assign it to the variable acc.
        acc = accuracy_score(y, y_pred)
        if acc > best_score:
            best_fit = (i, j)
            best_score = acc
            print('new best score: ' + str(best_score))
print('best_fit: ' + str(best_fit))
print('best score: ' + str(best_score))

print('\n\n*****WINNING TEST*****')
print('kernel: rbf')
print('C: ' + str(best_fit[0]))
print('gamma: ' + str(best_fit[1]))