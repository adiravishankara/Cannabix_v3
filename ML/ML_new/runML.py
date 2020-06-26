import pickle
import numpy as np
import dill
import math
from sklearn.svm import SVC

testFileLoc = 'f1.csv'
SamplesPerSecond = 10

testFile_array = np.genfromtxt('{}'.format(testFileLoc), delimiter=',')
print('Test File 1 Shape: {}'.format(testFile_array.shape))
testFile_array = testFile_array.transpose()

# model = dill.load(open('Model_0.sav', 'rb'))
# #model = pickle.load(open('Model.sav', 'rb'))
# print(model)
# # result = model(testFile_array)

import pickle

filename2 = 'Model_0.sav'
loaded_model = pickle.load(open(filename2, 'rb'))
result = loaded_model.predict(testFile_array)

print(result)


# print(result)

