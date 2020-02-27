


# %%


# %%
# '!' allows you to run bash commands from jupyter notebook.
print("List all the files in the current directory\n")
!ls
# The required data table can be found under smsspamcollection/SMSSpamCollection
print("\n List all the files inside the smsspamcollection directory\n")
!ls smsspamcollection

# %%
import pandas as pd
# Dataset available using filepath 'smsspamcollection/SMSSpamCollection'
df = pd.read_table('smsspamcollection', sep='\t', names=['label', 'sms_message'])

# Output printing out first 5 rows
df.head()

# %%
'''
Solution
'''
df['label'] = df.label.map({'ham':0,'spam':1})

# %%
'''
Solution:
'''
documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']

lower_case_documents = []
for i in documents:
    lower_case_documents.append(i.lower())
print(lower_case_documents)

# %%
'''
Solution:
'''
sans_punctuation_documents = []
import string

for i in lower_case_documents:
    sans_punctuation_documents.append(i.translate(str.maketrans('', '', string.punctuation)))
    
print(sans_punctuation_documents)

# %%
'''
Solution:
'''
preprocessed_documents = []
for i in sans_punctuation_documents:
    preprocessed_documents.append(i.split(' '))
print(preprocessed_documents)

#%%
'''
Solution
'''
frequency_list = []
import pprint
from collections import Counter

for i in preprocessed_documents:
    frequency_list.append(Counter(i))
    
pprint.pprint(frequency_list)

# %%
'''
Here we will look to create a frequency matrix on a smaller document set to make sure we understand how the 
document-term matrix generation happens. We have created a sample document set 'documents'.
'''
documents = ['Hello, how are you!',
                'Win money, win from home.',
                'Call me now.',
                'Hello, Call hello you tomorrow?']

# %%
'''
Solution
'''
from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer(lowercase = True, token_pattern = '(?u)\\b\\w\\w+\\b') #stop_words = 'english'

# %%
'''
Practice node:
Print the 'count_vector' object which is an instance of 'CountVectorizer()'
'''
# No need to revise this code
print(count_vector)

# %%
'''
Solution:
'''
# No need to revise this code
count_vector.fit(documents)
count_vector.get_feature_names()

# %%
'''
Solution
'''
doc_array = count_vector.transform(raw_documents=documents).toarray()
doc_array

# %%
'''
Solution
'''
frequency_matrix = pd.DataFrame(data=doc_array, columns = count_vector.get_feature_names())
frequency_matrix

# %%
'''
Solution 
'''
# split into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['sms_message'], 
                                                    df['label'], 
                                                    random_state=1)

print('Number of rows in the total set: {}'.format(df.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))

# %%
'''
Solution
'''
# Instantiate the CountVectorizer method
count_vector = CountVectorizer()

# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)

# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)