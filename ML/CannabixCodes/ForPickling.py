#This code pickles the preprocessing of the data + the model itself into one file
import numpy as np
import math
from numpy import genfromtxt

#file path of original data
DataLocation = '/Data/MLTraining'
#where to save data in:
SaveDataLocation = '/Data/MLTraining/Processed'
SamplesPerSecond = 10 # see from files how many reading per second

#print(trainRange.astype(int))
initial_data = genfromtxt(DataLocation + str(1) + '.csv', delimiter=',')

#For determining start time

def Model1(initial_data):
    current = initial_data[:,1]
    baseline = np.mean(current[0:30])
    bl_std_dev = np.std(current[0:30])

    counter = 0

    for p in current:

        a = abs(p-baseline)
        b = bl_std_dev*9

        if a > b :
            current = initial_data[counter:(counter+(200*SamplesPerSecond)),1:4]
                    #the number is the number of seconds after the start of the graph
            break

        counter=counter+1
    current_stacked = np.hstack((current[:,0], current[:,1], current[:,2]))
    current_stacked = current_stacked.reshape((current_stacked.shape[0], 1))

        #Downsampling parameters
    desiredTimeBetweenSamples = 1
    timeBetweenSamples = 1/SamplesPerSecond
    samplingRatio = math.floor(desiredTimeBetweenSamples / timeBetweenSamples)

        #Moving average filter (filters noise)
    samples = 5
    smoothedData = np.zeros((current_stacked.shape[0], current_stacked.shape[1]))

    for j in range(samples, current_stacked.shape[0] - samples):
        sum = 0
        for k in range(-1 * samples,samples + 1):
            sum = sum + current_stacked[j + k][0] #delete [0]

        smoothedData[j] = sum / (2 * samples + 1)

    for j in range(smoothedData.shape[0]):
        if smoothedData[j][0] == 0:
            smoothedData[j][0] = current_stacked[j]

            # Downsample - takes the values at time samples of multiples of 1 sec only, so one point from each 10
    downsampledData = np.zeros((1, 1))

    for j in range(smoothedData.shape[0]):
        if (j % samplingRatio == 0):
            if (j == 0):
                downsampledData[0][0] = np.array([[smoothedData[j, 0]]])
            else:
                downsampledData = np.vstack((downsampledData, np.array([[smoothedData[j, 0]]])))

    downsampledData = np.transpose(downsampledData)

    #import pickle

    #filename2 = 'Model_0.sav'

    #loaded_model = pickle.load(open(filename2, 'rb'))
    #result = loaded_model.predict(downsampledData)
    #if result == 0:
    #    x = 'No THC Detected'
    #else:
    #    x = 'THC Detected'
    return downsampledData

v = Model1(initial_data)

import dill
filename = 'Model.sav'
dill.dump(Model1,open(filename,'wb'))
