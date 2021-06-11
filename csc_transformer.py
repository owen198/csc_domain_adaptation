from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import mean_squared_error

import pandas as pd
import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random

from keras import metrics
from keras import Model
from keras import models
from keras import utils
from keras import losses
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Input, BatchNormalization, Activation
from keras.callbacks import EarlyStopping