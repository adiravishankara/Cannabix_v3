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
import matplotlib.pyplot as plt

TrainingDataLocation = '/Users/mohamedtarekaly/Desktop/CannabixData/Apr2020/'
TestingDataLocation = '/Users/mohamedtarekaly/Desktop/CannabixData/Apr2020/'
LabelsLocation = '/Users/mohamedtarekaly/Desktop/CannabixData/Apr2020/'

SignalStart = 0

#SignalLength = 105

def Standardise(train, test):

    std = sklearn.preprocessing.StandardScaler()
    std.fit(train)
    train = std.transform(train)
    test = std.transform(test)

    return train, test

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

def PCA_transfom(X_train, X_test, n):

    pca = PCA(n_components=n)
    pca.fit(X_train)
    X_train_PCA = pca.transform(X_train)
    X_test_PCA = pca.transform(X_test)

    return X_train_PCA, X_test_PCA
#A_ = ica.mixing_  # Get estimated mixing matrix

def FastICA_transfom(X_train, X_test, n):

    ica = FastICA(n_components=n)
    ica.fit(X_train)
    X_train_ICA = ica.fit_transform(X_train)
    X_test_ICA = ica.fit_transform(X_test)

    return X_train_ICA, X_test_ICA

def DW_transform(X_train, X_val, X_test):

    coeffs_train = pywt.wavedec(X_train, 'db6', level = 4)
    coeffs_val = pywt.wavedec(X_val, 'db6', level=4)
    coeffs_test = pywt.wavedec(X_test, 'db6', level = 4)

    X_train_enc = coeffs_train[0]
    X_val_enc = coeffs_val[0]
    X_test_enc = coeffs_test[0]

    return X_train_enc, X_val_enc, X_test_enc

maxTestAccuracy = []#
maxTrainAccuracy = []
PCs = []
j=0

NLIST = [458,4,39,890,121,82,113,34]#p.linspace(200,600,41) #for random test and train sets assignment
SignalLengthArray = [200]#,350,400,450,500,550,600]

#print(NLIST)
#for SignalStart in range(50,80):

for SignalLength in SignalLengthArray: #199

    TrainAccArray = []
    TestAccArray = []

    for n in range(16,17): #40
        AllTranAcc = []
        AllTestAcc = []

        for i in NLIST:

            data, targets = loadData('processed_data' + str(SignalLength) + '.csv', 'Targets.csv', TrainingDataLocation, LabelsLocation)

            from sklearn.model_selection import train_test_split
            train_data, test_data, train_targets, test_targets = train_test_split(data, targets[0:23], test_size = 0.2, random_state=i)
            #print(i)
            #train_data, val_data, train_targets, val_targets = train_test_split(train_data, train_targets, test_size = 0.08, random_state = 0)

            train_data = np.array(train_data)
            train_targets = np.array(train_targets)
            #pred_data = np.array(pred_data)
            #pred_targets = np.array(pred_targets)
            test_data = np.array(test_data)
            test_targets = np.array(test_targets)

            #train_data, test_data = Standardise(train_data, test_data)
            #train_data, test_data = PCA_transfom(train_data, test_data, 10)
            #train_data, test_data = FastICA_transfom(train_data, test_data, 5)
            #train_data, test_data = Standardise(train_data, test_data)

            print("training data and target shapes", train_data.shape, train_targets.shape, type(train_targets))
            #print(train_targets, i)
            print("test data and target shapes", test_data.shape, test_targets.shape, type(test_targets))

            #data = np.vstack((train_data, test_data))
            #labels = np.vstack((train_targets, test_targets))

            Cs = [1000, 100, 10, 1]
            gammas = [0.001, 0.01, 10, 100, 1000]
            kernels=['rbf','linear', 'sigmoid']

            #if j == 0:
            param_grid = {'C': Cs, 'gamma': gammas, 'kernel': kernels}
            grid_search = sklearn.model_selection.GridSearchCV(sklearn.svm.SVC(), param_grid, cv=5)
            print(i)
            grid_search.fit(train_data, train_targets)
            #print(grid_search.best_params_)
            #j=1
            #print(j)

            PARAMETERS = grid_search.best_params_

            #clf = SVC(kernel=PARAMETERS['kernel'], C=PARAMETERS['C'], gamma=PARAMETERS['gamma']).fit(train_data,train_targets)
            pred_t = grid_search.predict(test_data)
            #print classification report
            #print(classification_report(test_targets, grid_predictions))

            acctest = grid_search.score(test_data, test_targets)
            acctrain = grid_search.score(train_data, train_targets)

            AllTranAcc.append(acctrain)
            AllTestAcc.append(acctest)

            print(confusion_matrix(test_targets, pred_t))
            print("True test targets:", test_targets)
            print("Predicted targets:", pred_t)

        TrainAcc = sum(AllTranAcc) / len(NLIST)
        TestAcc = sum(AllTestAcc) / len(NLIST)

        #this is for pca
        print(AllTestAcc)
        print("For Signal length of " + str(SignalLength) + ", train accuracy is " + str(TrainAcc) + ", and test accuracy is " + str(TestAcc))

#ACCURACIES = np.transpose(ACCURACIES)
#np.savetxt(TrainingDataLocation + 'SVM_PCA_accuracies.csv', ACCURACIES, fmt='%.5f', delimiter=',')
#plt.plot(maxTestAccuracy)
#plt.plot(PCs)
#results = np.vstack((maxTestAccuracy, maxTrainAccuracy))

#plt.show()
print('[TN FP]\n[FN TP]')

import pickle
filename = 'Model_0.sav'
pickle.dump(grid_search, open(filename, 'wb'))
