import numpy as np
import sklearn
import pywt
import pandas as pd
import scipy.io
import os
import re
from sklearn.decomposition import PCA, FastICA
import sklearn.model_selection
from sklearn.metrics import classification_report, confusion_matrix

trainingDirectory = ''
testingDirectory = ''
labelDirectory = ''

signal_start = 0


def main():
	print('Do something')
	print(os.getcwd())

if __name__ == '__main__':
	main()
