#This code trains the loaded data through a SVC. It uses grid search to determine
# the optimum combination of hyper-parameters. It currently has the option to use PCA and ICA
# as a data representation techniques
#Chang: 1) Folders, 2) SignalLengthArray.

from sklearn.metrics import confusion_matrix
import numpy as np
import sklearn
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputRegressor
import pywt
from sklearn.decomposition import PCA, FastICA
import sklearn.model_selection
from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
import os

print(os.getcwd())

TrainingDataLocation = 'Data/'
TestingDataLocation = 'Data/'
LabelsLocation = 'Data/'

SignalStart = 0

#SignalLength = 105


def loadData(DataFile, LabelsFile, folder1, folder2): #name of datafile,

    data = np.genfromtxt(folder1 + DataFile, delimiter=',')
    data = data[:,:]
    data = np.transpose(data)

    labels = np.genfromtxt(folder2 + LabelsFile, delimiter=',')
    labels = labels[:,1]
    #print(labels)
    #train_targets = train_targets.reshape(len(train_data), 1)
    #test_targets = test_targets.reshape(len(test_data), 1)
    #data = data.reshape(len(data), 1)

    return data, labels


maxTestAccuracy = []#
maxTrainAccuracy = []
PCs = []
j=0
pp = 0

NLIST = [4,34,39,82,13,121,458,37]#[458,4,39,890,121,82,113,34]#p.linspace(200,600,41) #for random test and train sets assignment
SignalLengthArray = [200]#,350,400,450,500,550,600]

for SignalLength in SignalLengthArray: #199

    TrainAccArray = []
    TestAccArray = []

    for n in range(6,7): #40
        AllTranAcc = []
        AllTestAcc = []

        for i in NLIST:

            train_data, train_targets = loadData('processed_data' + str(SignalLength) + '.csv', 'Targets.csv', TrainingDataLocation, LabelsLocation)
            train_data, train_targets, test_data, test_targets = train_test_split()
            from sklearn.model_selection import train_test_split
            #train_data, test_data, train_targets, test_targets = train_test_split(data, targets[0:23], test_size = 0.2, random_state=i)
            #print(i)
            #train_data, val_data, train_targets, val_targets = train_test_split(train_data, train_targets, test_size = 0.08, random_state = 0)

            train_data = np.array(train_data)
            train_targets = np.array(train_targets)
            #pred_data = np.array(pred_data)
            #pred_targets = np.array(pred_targets)
            test_data = train_data#np.array(test_data)
            test_targets = train_targets#np.array(test_targets)


            print("training data and target shapes", train_data.shape, train_targets.shape, type(train_targets))
            #print(train_targets, i)
            print("test data and target shapes", test_data.shape, test_targets.shape, type(test_targets))

            clf = SVC(kernel='linear', C=1000, gamma=0.001).fit(train_data,train_targets)
            clf.probability=True
            clf.fit(train_data,train_targets)

            pred_t = clf.predict(test_data)
            #print classification report
            #print(classification_report(test_targets, grid_predictions))

            acctest = clf.score(test_data, test_targets)
            acctrain = clf.score(train_data, train_targets)
            test_pred_targets = clf.decision_function(test_data)
            train_pred_targets = clf.decision_function(train_data)

            test_rpred_targets = clf.predict(test_data)
            train_rpred_targets = clf.predict(train_data)

            print(i, "-----------", acctrain)

            test_pred_targets = test_pred_targets.reshape(test_pred_targets.shape[0], 1)
            train_pred_targets = train_pred_targets.reshape(train_pred_targets.shape[0], 1)
            train_rpred_targets = train_rpred_targets.reshape(train_rpred_targets.shape[0], 1)

            train_targets = train_targets.reshape(train_targets.shape[0], 1)
            test_targets = test_targets.reshape(test_targets.shape[0], 1)

            test_rpred_targets = test_rpred_targets.reshape(len(test_rpred_targets), 1)
            test_pred_targets = test_pred_targets.reshape(len(test_pred_targets), 1)

            print(test_pred_targets.shape, test_rpred_targets.shape, test_targets.shape)
            CombinedRealTargets = np.hstack((test_pred_targets, test_rpred_targets, test_targets))
            print(CombinedRealTargets)
            print(i, "-----------", acctest, '\n')

            AllTranAcc.append(acctrain)
            AllTestAcc.append(acctest)

            if pp == 0:
                pp = 1
                PredictionPropabilities = test_pred_targets
                Predictions = test_rpred_targets

                TrainingPropabilities = train_pred_targets
                TrPredictions = train_rpred_targets

                TestLabels = test_targets
                TrLabels = train_targets

            else:
                PredictionPropabilities = np.vstack((PredictionPropabilities, test_pred_targets))
                TrainingPropabilities = np.vstack((TrainingPropabilities, train_pred_targets))

                Predictions = np.vstack((Predictions, test_rpred_targets))
                TrPredictions = np.vstack((TrPredictions, train_rpred_targets))

                TestLabels = np.vstack((TestLabels, test_targets))
                TrLabels = np.vstack((TrLabels, train_targets))

            print(confusion_matrix(test_targets, pred_t))
            print("True test targets:", np.transpose(test_targets))
            print("Predicted targets:", pred_t)

        TrainAcc = sum(AllTranAcc) / (len(NLIST))
        TestAcc = sum(AllTestAcc) / (len(NLIST))

        #this is for pca
        print(AllTestAcc)
        print("For Signal length of " + str(SignalLength)+ str(n)+ ", train accuracy is " + str(TrainAcc) + ", and test accuracy is " + str(TestAcc))

# print(clf.predict(np.transpose(np.genfromtxt('f1.csv',delimiter=','))))

import pickle
filename = 'Model_2.sav'
pickle.dump(clf, open(filename, 'wb'))