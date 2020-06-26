import numpy as np
from numpy import genfromtxt
import math

#file path of original data
DataLocation = 'Data/'
#where to save data in:

SamplesPerSecond = 10 # see from files how many reading per second
from numpy import genfromtxt

file = '5'
data = genfromtxt(DataLocation + file + '.csv', delimiter=',')

#print(trainRange.astype(int))

data = genfromtxt(DataLocation + file + '.csv', delimiter=',')

#print('Initial data shape is ', initial_data.shape)
#For determining start time
#print(initial_data)

SamplesPerSecond = 10

import dill
filename = 'Model.sav'
p = dill.load(open(filename,'rb'))
result = p(data)

print(result)

#import pickle

#filename2 = 'Model_0.sav'
#loaded_model = pickle.load(open(filename2, 'rb'))
#result = loaded_model.predict(result)

#print(result)