from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import mean_squared_error

import pandas as pd
import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

from tensorflow import keras

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

import kerastuner as kt
import os
import tensorflow


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_devices = tensorflow.config.experimental.list_physical_devices('GPU')
tensorflow.config.experimental.set_memory_growth(gpu_devices[0], True)
gpus = tensorflow.test.gpu_device_name()


epoch = sys.argv[0]
print('epoch=', epoch)

timesteps = 128
n_features = 390
epoch = 10
retrain = True
path = '/data/'

tag_dict = {'source':'W4662FM0605',
            'source_training_from': datetime.datetime(2020,3,1,0,0),
            'source_training_to': datetime.datetime(2020,4,1,0,0),
            'source_end': datetime.datetime(2020,7,1,0,0),
            
            'target':'W4662FM0606',
            'target_training_from': datetime.datetime(2020,9,1,0,0), 
            'target_training_to': datetime.datetime(2021,1,1,0,0), 
            'target_end': datetime.datetime(2021,2,1,0,0)}

globals()[tag_dict['target']] = pd.concat([pd.read_csv(path + tag_dict['target'] + '_202009.csv'),
                                          pd.read_csv(path + tag_dict['target'] + '_202010.csv'),
                                          pd.read_csv(path + tag_dict['target'] + '_202011.csv'),
                                          pd.read_csv(path + tag_dict['target'] + '_202012.csv'),
                                          pd.read_csv(path + tag_dict['target'] + '_202101.csv'),
                                          pd.read_csv(path + tag_dict['target'] + '_202102.csv')])

globals()[tag_dict['source']] = pd.concat([pd.read_csv(path + tag_dict['source'] + '_202003.csv'),
                                          pd.read_csv(path + tag_dict['source'] + '_202004.csv'),
                                          pd.read_csv(path + tag_dict['source'] + '_202005.csv'),
                                          pd.read_csv(path + tag_dict['source'] + '_202006.csv'),
                                          pd.read_csv(path + tag_dict['source'] + '_202007.csv')])

globals()[tag_dict['source']]['datetime'] = globals()[tag_dict['source']]['timestamp'].astype('int').astype("datetime64[s]")
globals()[tag_dict['target']]['datetime'] = globals()[tag_dict['target']]['timestamp'].astype('int').astype("datetime64[s]")

globals()[tag_dict['source']+'_training'] = globals()[tag_dict['source']][
                                                  (globals()[tag_dict['source']]['datetime'] > tag_dict['source_training_from']) &
                                                  (globals()[tag_dict['source']]['datetime'] <= tag_dict['source_training_to']) ]

globals()[tag_dict['target']+'_training'] = globals()[tag_dict['target']][
                                                  (globals()[tag_dict['target']]['datetime'] > tag_dict['target_training_from']) &
                                                  (globals()[tag_dict['target']]['datetime'] <= tag_dict['target_end']) ]

drop_list = ['Unnamed: 0', '_id','type','scada','timestamp','device', 'datetime']

globals()[tag_dict['source']+'_training'] = globals()[tag_dict['source']+'_training'].drop(columns=drop_list)
globals()[tag_dict['target']+'_training'] = globals()[tag_dict['target']+'_training'].drop(columns=drop_list)

shape_min = min (globals()[tag_dict['source']+'_training'].shape[0], 
                 globals()[tag_dict['target']+'_training'].shape[0])
shape_max = max (globals()[tag_dict['source']+'_training'].shape[0], 
                 globals()[tag_dict['target']+'_training'].shape[0])

index = sorted(random.sample(range(0, shape_max), shape_min))

normalizer = preprocessing.MinMaxScaler()
source_normalizer = normalizer.fit(globals()[tag_dict['source']+'_training'])

normalizer = preprocessing.MinMaxScaler()
target_normalizer = normalizer.fit(globals()[tag_dict['target']+'_training'])

X_source = pd.DataFrame(source_normalizer.transform(globals()[tag_dict['source']+'_training']))
X_target = pd.DataFrame(source_normalizer.transform(globals()[tag_dict['target']+'_training']))

def temporalize(X, y, lookback):
    output_X = []
    output_y = []

    for i in range(X.shape[0]-lookback-1):
        t = []
        for j in range(1,lookback+1):
            # Gather past records upto the lookback period
            t.append(X[[(i+j+1)], :])
        output_X.append(t)
        output_y.append(y[i+lookback+1])
    return output_X, output_y

if len(X_target) > len(X_source):
    X = X_target.iloc[index]
    Y = X_source
else:
    X = X_target
    Y = X_source.iloc[index]

#X, y = temporalize(X = timeseries, y = np.zeros(len(timeseries)), lookback = timesteps)
X, _ = temporalize(X = X.values, y = np.zeros(len(X)), lookback = timesteps)
Y, _ = temporalize(X = Y.values, y = np.zeros(len(Y)), lookback = timesteps)

X = np.array(X)
X = X.reshape(X.shape[0], timesteps, n_features)

Y = np.array(Y)
Y = Y.reshape(np.array(Y).shape[0], timesteps, n_features)

model_name = '../data/W4/' + tag_dict['target'] + '_encoded'
model_name

if retrain:

    encoder = Sequential(
        [
            LSTM(180, activation='tanh', return_sequences=True),
            LSTM(128, activation='tanh', return_sequences=True),
            LSTM(16, activation='tanh', return_sequences=False),
            RepeatVector(timesteps)
        ],
        name='encoder'
    )

    decoder = Sequential(
        [
            LSTM(16, activation='tanh', return_sequences=True),
            LSTM(128, activation='tanh', return_sequences=True),
            LSTM(780, activation='tanh', return_sequences=True),
            TimeDistributed(Dense(n_features))
        ],
        name='decoder'
    )

    source = Input(shape=(timesteps,n_features), name='source')
    target = Input(shape=(timesteps,n_features), name='target')
    synthetic = decoder(encoder(source))


    model = Model(
        inputs={'source': source, 'target': target},
        outputs=synthetic
    )

    transformation_loss = losses.mean_squared_error(target, synthetic)
    model.add_loss(transformation_loss)
    model.add_metric(transformation_loss, name='transformation_loss')

    model.compile(optimizer='adam', metrics=[metrics.RootMeanSquaredError()])
    model.summary()

    lstm_ae_model = model.fit(
                        x={'source': Y, 'target': X},
                        epochs=epoch,
                        batch_size=16, 
                        verbose=1,
                        validation_split=0.2
    )

    model.save(model_name)

else:
    model = models.load_model(model_name)  

if retrain:
    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(lstm_ae_model.history['loss'], marker='.', label='loss')
    ax.plot(lstm_ae_model.history['val_loss'], marker='.', label='validation loss')
    ax.plot(lstm_ae_model.history['transformation_loss'], marker='.', label='transformation loss')

    ax.legend()
    ax.grid(True)

    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig('epoches.png', dpi=300)
    plt.show()

synthetic_data = model.predict(x={'source': Y, 'target': X}, verbose=0)
synthetic_pd = pd.DataFrame.from_records([i[0] for i in synthetic_data])

def scorer_(Y_pred):
    a = (Y_pred[Y_pred == -1].size)/(Y_pred.size)
    return a*100

def plot_score (score_list, date_list, tag):

    fig, ax = plt.subplots(figsize=(10, 3))
    plt.xticks(rotation=45)
    ax.plot(date_list, score_list, '.-')
    ax.set(xlabel='date', ylabel='score', title=tag)
    ax.grid()
    plt.ylim(0, 100)
    plt.tight_layout()
    #fig.savefig(path+tag+'-'+training_from.strftime("%Y%m%d")+'-'+training_to.strftime("%Y%m%d")+'.png', dpi=300)
    plt.show()

def get_score (data_df, start_date, end_date, normalizer, prediction_model):

    score_list = []
    date_list = []
    delta = datetime.timedelta(days=1)

    
    while start_date <= end_date:

        validation_df = data_df[(data_df['datetime'] > start_date) & 
                                (data_df['datetime'] <= start_date + delta)]

        #print(validation_df.shape, start_date)

        
        if validation_df.shape[0] > 0:

            validation_df_score = validation_df.drop(columns=drop_list)
            validation_df_score = normalizer.transform(validation_df_score)
            validation_df_score = prediction_model.predict(validation_df_score)

            score_list.append(scorer_(validation_df_score))
            date_list.append(start_date)

        start_date += delta
        
    return score_list, date_list

model_source = svm.OneClassSVM(nu=0.01, kernel="rbf", gamma=0.01).fit(X_source)
model_target = svm.OneClassSVM(nu=0.01, kernel="rbf", gamma=0.01).fit(X_target)
model_synthetic = svm.OneClassSVM(nu=0.01, kernel="rbf", gamma=0.01).fit(synthetic_pd)

source_test_pd = pd.DataFrame()
source_test_pd = globals()[tag_dict['source']]
source_test_pd = source_test_pd.drop(columns=drop_list)

index_2 = sorted(random.sample(range(0, source_test_pd.shape[0]), shape_min))

source_test_pd = pd.DataFrame(target_normalizer.transform(source_test_pd)).iloc[index_2]

source_test_np, _ = temporalize(X = source_test_pd.values, 
                                y = np.zeros(source_test_pd.shape[0]), 
                                lookback = timesteps)

source_test_np = np.array(source_test_np)
source_test_np = source_test_np.reshape(source_test_np.shape[0], timesteps, n_features)

synthetic_source = model.predict(x={'source': source_test_np, 'target': X}, verbose=0)
synthetic_source_pd = pd.DataFrame.from_records([i[0] for i in synthetic_source])

#diff = globals()[tag_dict['source']].shape[0] - synthetic_source_pd.shape[0]
#len = globals()[tag_dict['source']].shape[0] - diff

for elements in drop_list:
    synthetic_source_pd[elements] = globals()[tag_dict['source']][elements].iloc[index_2].tail(synthetic_source_pd.shape[0]).values

def get_syntheic_score (data_df, start_date, end_date, prediction_model):

    score_list = []
    date_list = []
    delta = datetime.timedelta(days=1)

    while start_date <= end_date:

        validation_df = data_df[(data_df['datetime'] > start_date) & 
                                (data_df['datetime'] <= start_date + delta)]

        if validation_df.shape[0] > 0:

            validation_df_score = validation_df.drop(columns=drop_list)
            validation_df_score = prediction_model.predict(validation_df_score)

            score_list.append(scorer_(validation_df_score))
            date_list.append(start_date)


        start_date += delta
        
    return score_list, date_list

synthetic_score, synthetic_date = get_syntheic_score(synthetic_source_pd, 
                                            tag_dict['source_training_from'], 
                                            tag_dict['source_end'], 
                                            model_target)

source_score_cv, source_date_cv = get_score(globals()[tag_dict['source']], 
                                            tag_dict['source_training_from'], 
                                            tag_dict['source_end'], 
                                            source_normalizer,
                                            model_target)

source_score, source_date = get_score(globals()[tag_dict['source']], 
                                            tag_dict['source_training_from'], 
                                            tag_dict['source_end'], 
                                            source_normalizer,
                                            model_source)

sy_rmse = 0
cv_rmse = 0
plot_score (source_score, 
            source_date, 
            'Detect ' + tag_dict['source'] +' conditions by using '+ tag_dict['source'] +' model')

plot_score (source_score_cv, 
            source_date_cv, 
            'Detect ' + tag_dict['source'] +' conditions by using '+ tag_dict['target'] +' model, RMSE='+ "{:.3f}".format(cv_rmse))

plot_score (synthetic_score, 
            synthetic_date, 
            'Detect synthetic conditions by using '+ tag_dict['target'] +' model, RMSE='+ "{:.3f}".format(sy_rmse))