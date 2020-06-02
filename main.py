### This file will run the intro page, which will then open one of two files.

import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pyqtgraph as pg

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
import random
import sys
import time
from datetime import date
import csv
import os
from numpy import genfromtxt
from math import *
from pandas import read_csv as rc
from statistics import mean

prm_list = rc('parameters.csv', delimiter=',', header='infer')
sVersion = prm_list['Software Version'].values
dVersion = prm_list['Device Version'].values
dID = prm_list['Device ID'].values

# Testing Calling Values
print('Software Version: {} \nDevice Version: {} \nDevice ID: {}'.format(sVersion, dVersion, dID))

loadScn = QApplication([])
loadScn.setStyle('fusion')
ls = QWidget()
ls.setWindowTitle('Title Screen')
# # lsLayout = QGridLayout()
picLabel = QLabel()
lPic = QPixmap('/venv/media/canLogoFull.png')
picLabel.setPixmap(lPic)
# ls.resize(lPic.width(), lPic.height())

ls.show()
loadScn.exec_()
