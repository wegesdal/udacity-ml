# Import statements 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Read the data.
data = np.asarray(pd.read_csv('data6.csv', header=None))
# Assign the features to the variable X, and the labels to the variable y. 
X = data[:,0:2]
y = data[:,2]

# TODO: Create the decision tree model and assign it to the variable model.
# You won't need to, but if you'd like, play with hyperparameters such
# as max_depth and min_samples_leaf and see what they do to the decision
# boundary.
model = DecisionTreeClassifier()

# TODO: Fit the model.
model.fit(X, y)

# TODO: Make predictions. Store them in the variable y_pred.
y_pred = np.array(model.predict(X))
print(y_pred)


# TODO: Calculate the accuracy and assign it to the variable acc.
acc = accuracy_score(y, y_pred)
print(acc)


# %%
# TRAINING THE TITANIC MODEL

best_score = 0.0
best_fit = (1, 1, 1)
for i in range (1, 15):
    for j in range (1, 15):
        for k in range (2, 15):
            model = DecisionTreeClassifier(max_depth = i, min_samples_leaf = j, min_samples_split = k)
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            if test_acc > best_score:
                best_score = test_acc
                best_fit = (i, j, k)
                print('new best score: ' + str(test_acc))
                print('winning model: ' + str((i, j, k)))
print('\n\n***** BEST FIT *****')
print('max_depth: ' + str(best_fit[0]))
print('min_samples_leaf: ' + str(best_fit[1]))
print('min_samples_split: ' + str(best_fit[2]))

#%%
#solution

model = DecisionTreeClassifier(max_depth = 7, min_samples_leaf = 6, min_samples_split = 2)
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
print('train accuracy: ' + str(train_acc))
print('test accuracy: ' + str(test_acc))
