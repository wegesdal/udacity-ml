#%%
# Import our libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import seaborn as sns
sns.set(style="ticks")

%matplotlib inline

# Read in our dataset
diabetes = pd.read_csv('diabetes.csv')

# Take a look at the first few rows of the dataset
diabetes.head()



# %%
X = np.array(diabetes[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']])
y = np.array(diabetes['Outcome'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#from sklearn.ensemble import AdaBoostClassifier
# build a classifier for ada boost

clf_ada = AdaBoostClassifier()

# Set up the hyperparameter search
# look at  setting up your search for n_estimators, learning_rate
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
param_dist = {
    "n_estimators": list(range(10, 100))
}

# Run a randomized search over the hyperparameters
random_search = RandomizedSearchCV(clf_rf, param_distributions=param_dist)

# Fit the model on the training data
random_search.fit(X_train, y_train)


# Make predictions on the test data
ada_preds = random_search.best_estimator_.predict(X_test)

# Return your metrics on test data
ch.print_metrics(y_test, ada_preds, 'adaboost')


#%%


#from sklearn.ensemble import AdaBoostClassifier
# build a classifier for ada boost

clf_ada = AdaBoostClassifier()

# Set up the hyperparameter search
# look at  setting up your search for n_estimators, learning_rate
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
param_dist = {
    "n_estimators": list(range(10, 100)),
    "learning_rate": [0.01, 0.1, 1, 2, 5, 10]
}

# Run a randomized search over the hyperparameters
random_search = RandomizedSearchCV(clf_ada, param_distributions=param_dist)

# Fit the model on the training data
random_search.fit(X_train, y_train)


# Make predictions on the test data
ada_preds = random_search.best_estimator_.predict(X_test)
print(random_search.best_estimator_)

# Return your metrics on test data
ch.print_metrics(y_test, ada_preds, 'adaboost')


#%%

# build a classifier for support vector machines

clf_svm = SVC()

# Set up the hyperparameter search
# look at setting up your search for C (recommend 0-10 range), 
# kernel, and degree
# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

param_dist = {
    "C": list(range(1,10)),
    "kernel": ["linear", "poly", "rbf"],
    "degree": list(range(2, 5))
}

# Run a randomized search over the hyperparameters
random_search = RandomizedSearchCV(clf_svm, param_distributions=param_dist)

# Fit the model on the training data
random_search.fit(X_train, y_train)

# Make predictions on the test data
svc_preds = random_search.best.estimator_.predict(X_test)


# Return your metrics on test data
ch.print_metrics(y_test, svc_preds, 'svc')


#%%

# Show your work here - the plot below was helpful for me
# https://stackoverflow.com/questions/44101458/random-forest-feature-importance-chart-using-python


model = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1,
          n_estimators=26, random_state=None)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

features = diabetes.columns[[0,1,2,3,4,5,6,7]]
importances = model.feature_importances_
indices = np.argsort(importances)

plt.figure(1)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')