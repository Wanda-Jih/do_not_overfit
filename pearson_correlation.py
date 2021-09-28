import os
import pandas as pd

data_dir = 'data/'

# get training dataset
data = os.path.join(data_dir,'train.csv')
trainingSet = pd.read_csv(data)

# use pearsno's correlation
correlation = trainingSet.corr(method='pearson')

# leave the first col
correlation = correlation.iloc[1] 
print(correlation)

# collect the positive correlation
positive_corr = []
count = 0
for i, row in enumerate(correlation):
    if i > 1 and row > 0:
        positive_corr.append(i-2)
        count += 1

# we got 126 features
print(positive_corr) 
