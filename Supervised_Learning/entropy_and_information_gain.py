
# %%
#entropy of bucket with 4 red balls and 10 blue balls
import math
m = 4
n = 10

e = -((m / (m + n)) * math.log(m / (m + n), 2)) - (n / (m + n) * math.log(n / (m + n), 2))
 
print(e)

# %%

m = 4
n = 10
p1 = m / (m + n)
p2 = n / (m + n)

e = -p1*math.log(p1, 2) -p2*math.log(p2, 2)
print(e)

# %%
# Multi-class entropy
import numpy as np
# list of balls in bucket
b = np.array([8, 3, 2])
P = np.multiply(-b / b.sum(), np.log2(b / b.sum())) #populate with -p*log2p
ent = P.sum()
print(ent)

# %%
# information gain is entropy (parent) - sum of [prob of child*entropy(child)]
import numpy as np
import pandas as pd

def entropy(A):
    return np.multiply(-A / A.sum(), np.log2(A / A.sum())).sum()

def info_gain(parent, children):
    return entropy(parent) - np.multiply(c / c.sum(), entropy(c)).sum()

bug_data = pd.read_csv('ml-bugs.csv').to_numpy()

#create two new sets, these are the children
def split(data, cond):
    return [data[cond], data[~cond]]

#count the frequency of named features and return a table
parent = bug_data[:,0:1]

def get_freq_table(arr):
    u, c = np.unique(arr, return_counts=True)
    return np.asarray((u, c)).T

def entropy_from_freq_table(ftbl):
    print(ftbl)
    return entropy(np.array(ftbl[:,1], dtype='float'))

parent_freq_table = get_freq_table(parent)
parent_entropy = entropy_from_freq_table(parent_freq_table)
print(parent_entropy)

conds = ["bug_data[:,1:2] == 'Brown'", "bug_data[:,1:2] == 'Blue'", "bug_data[:,1:2] == 'Green'", "bug_data[:,2:3] < 17.0", "bug_data[:,2:3] < 20.0"]

for cond in conds:
    print('\n\nentropy of children with cond ' + cond)
    children = split(bug_data[:,0:1], eval(cond))
    weighted_sum_of_children_entropy = 0;
    m = len(children[0])
    n = len(children[1])
    for i in range (len(children)):
        if i == 0:
            print('true: ')
        else:
            print('false')
        weighted_sum_of_children_entropy += entropy_from_freq_table(get_freq_table(children[i])) * (len(children[i]) / (m + n))
    print('\ninformation gain for ' + cond + ' : ')
    print(parent_entropy - weighted_sum_of_children_entropy)