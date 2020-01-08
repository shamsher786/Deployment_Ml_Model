# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 22:20:49 2020

@author: sa_kh
"""

import pandas as pd
import pickle

dataset = pd.read_csv('hiring.csv')

dataset['experience'].fillna(0,inplace=True)
dataset['test_score'].fillna(dataset['test_score'].mean() , inplace=True)

X = dataset.iloc[: , :3]  #independent variable

def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4 , 'five':5,
                 'six':6, 'seven':7,'eight':8, 'nine':9, 'ten':10,
                 'eleven' :11 , 'twelve':12, 'thirteen':13, 0:0}
    return word_dict[word]

X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[: ,-1]  #dependent feature

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X,y)

# save model to disk
pickle.dump(regressor , open('model.pkl','wb'))

#load model for cpmparing the results

model = pickle.load(open('model.pkl','rb'))