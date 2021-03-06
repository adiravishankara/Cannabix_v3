import numpy as np
import math
import pandas as pd
import sklearn
import scipy.io
import os
from numpy import genfromtxt
import re
from os import listdir
from os.path import isfile, join

# THIS FILE TAKES RAW INPUT FROM SENSORS THEN APPLIES A MOVING AVERAGE AND DOWNSAMPLES THE FILES
#it also automatcally finds an value to start with from exposure so that all features are similar.
#CHANGE: 1) DataLocations, 2) NoOfFiles, 3) SamplesPerSecond, 4) Depending on Quad or triple, change line 67, 5) SignalLengthArray


#file path of original data
DataLocation = 'Data/'
#where to save data in:
SaveDataLocation = 'pre_process/'

NoOfFiles = 23
StaringFile = 1

trainRange = np.linspace(StaringFile, StaringFile+NoOfFiles-1, NoOfFiles) #[i for i in range(1481, NoOfFiles+1)]  # defines an array with numbers starting from -- to --
SamplesPerSecond = 10 # see from files how many reading per second
SignalLengthArray = [200,220]#,350,400,450,500,550,600]

#print(trainRange.astype(int))
trainRange = trainRange.astype(int)

for SignalLength in SignalLengthArray:

    for i in trainRange:  # for each data file, save to current,

        # Get each test individually at the specified location
        print(i)
        initial_data = genfromtxt('{}{}.csv'.format(DataLocation, i), delimiter=',')

        print('Initial data shape is ', initial_data.shape)

        # Discard time column, and first row having humidity and temp data and reshape data
        #current = current[:,1] + current[:,2] #+ current[:,3] #- current[:, 1]
        #For cannabix:
        #current_data = np.vstack((current[:,1], current[:,2], current[:,3]))
        #print(current_data.shape)

        current = initial_data[:,1] #just take the first sensor data to determine the baseline and hence exposure time

        #Determining the exposure time by locating the point where the baseline changes.
        baseline = np.mean(current[0:30])
        bl_std_dev = np.std(current[0:30])

        counter = 0

        print(baseline, bl_std_dev)

        for p in current:

            a = abs(p-baseline)
            b = bl_std_dev*9

            if a > b:
                print("exp at", counter, p)
                current = initial_data[counter:(counter+(SignalLength*SamplesPerSecond)),1:4]
                #the number is the number of seconds after the start of the graph
                break

            counter=counter+1
        print("current shape is", current.shape)
        current_stacked = np.hstack((current[:,0], current[:,1], current[:,2]))#, current[:,3]))
        print("current stacked shape is", current_stacked.shape)

        #print(average, std_dev)
        #print(len(current))
        current_stacked = current_stacked.reshape((current_stacked.shape[0], 1))

        # Downsampling parameters
        desiredTimeBetweenSamples = 1
        timeBetweenSamples = 1/SamplesPerSecond
        samplingRatio = math.floor(desiredTimeBetweenSamples / timeBetweenSamples)

        # Moving average filter (filters noise)
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

        if i == trainRange[0]:
            stacked = downsampledData

        else:
            stacked = np.hstack((stacked, downsampledData))

    train = stacked
    print("train shape: " + str(train.shape))
    np.savetxt('{}pd{}.csv'.format(SaveDataLocation, SignalLength), train, fmt='%.10f', delimiter=',')

#targets = np.genfromtxt(DataLocation + "Targets.csv", delimiter=',')

#to save a file of data with targets
#targets = np.transpose(targets[:,1:3])

#print(data.shape,targets.shape)
#dwt = np.transpose(np.vstack((data,targets)))
#np.savetxt(SaveDataLocation + "Data+Targets.csv", dwt, fmt='%.10f', delimiter=',')