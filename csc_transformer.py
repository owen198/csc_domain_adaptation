from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import mean_squared_error

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import os
import tensorflow
import datetime

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


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_devices = tensorflow.config.experimental.list_physical_devices('GPU')
tensorflow.config.experimental.set_memory_growth(gpu_devices[0], True)
gpus = tensorflow.test.gpu_device_name()

# python csc_transformer.py W4662FM0605 W4662FM0606 2 4 128 64

source = sys.argv[1]
target = sys.argv[2]
epoch = int(sys.argv[3])
timesteps = int(sys.argv[4])
units_layer_1 = int(sys.argv[5])
units_layer_2 = int(sys.argv[6])

filename = sys.argv[1]+'_'+target+'_'+str(epoch)+'_'+ \
                    str(timesteps)+'_'+str(units_layer_1)+'_'+str(units_layer_2)



n_features = 390
retrain = True
path = '/data/'


tag_pd = pd.read_csv('csc_w4.csv')


source_training_from_month = int(tag_pd[(tag_pd['tag']==sys.argv[1]) & (tag_pd['Normal']==1)].head(1)['data'].values[0].split('_')[1].split('.')[0][4:6])
source_training_from_year = int(tag_pd[(tag_pd['tag']==sys.argv[1]) & (tag_pd['Normal']==1)].head(1)['data'].values[0].split('_')[1].split('.')[0][0:4])
source_training_to_month = int(tag_pd[(tag_pd['tag']==sys.argv[1]) & (tag_pd['Normal']==1)].tail(1)['data'].values[0].split('_')[1].split('.')[0][4:6])
source_training_to_year = int(tag_pd[(tag_pd['tag']==sys.argv[1]) & (tag_pd['Normal']==1)].tail(1)['data'].values[0].split('_')[1].split('.')[0][0:4])
source_end_month = int(tag_pd[(tag_pd['tag']==sys.argv[1]) & (tag_pd['Normal']==0)].tail(1)['data'].values[0].split('_')[1].split('.')[0][4:6])
source_end_year = int(tag_pd[(tag_pd['tag']==sys.argv[1]) & (tag_pd['Normal']==0)].tail(1)['data'].values[0].split('_')[1].split('.')[0][0:4])

target_training_from_month = int(tag_pd[(tag_pd['tag']==sys.argv[2]) & (tag_pd['Normal']==1)].head(1)['data'].values[0].split('_')[1].split('.')[0][4:6])
target_training_from_year = int(tag_pd[(tag_pd['tag']==sys.argv[2]) & (tag_pd['Normal']==1)].head(1)['data'].values[0].split('_')[1].split('.')[0][0:4])
target_training_to_month = int(tag_pd[(tag_pd['tag']==sys.argv[2]) & (tag_pd['Normal']==1)].tail(1)['data'].values[0].split('_')[1].split('.')[0][4:6])
target_training_to_year = int(tag_pd[(tag_pd['tag']==sys.argv[2]) & (tag_pd['Normal']==1)].tail(1)['data'].values[0].split('_')[1].split('.')[0][0:4])
target_end_month = int(tag_pd[(tag_pd['tag']==sys.argv[2]) & (tag_pd['Normal']==0)].tail(1)['data'].values[0].split('_')[1].split('.')[0][4:6])
target_end_year = int(tag_pd[(tag_pd['tag']==sys.argv[2]) & (tag_pd['Normal']==0)].tail(1)['data'].values[0].split('_')[1].split('.')[0][0:4])



tag_dict = {'source':sys.argv[1],
            'source_training_from': datetime.datetime(source_training_from_year,source_training_from_month,1,0,0),
            'source_training_to': datetime.datetime(source_training_to_year,source_training_to_month,1,0,0),
            'source_end': datetime.datetime(source_end_year,source_end_month,1,0,0),
            
            'target':sys.argv[2],
            'target_training_from': datetime.datetime(target_training_from_year,target_training_from_month,1,0,0), 
            'target_training_to': datetime.datetime(target_training_to_year,target_training_to_month,1,0,0), 
            'target_end': datetime.datetime(target_end_year,target_end_month,1,0,0)}

globals()[tag_dict['target']] = pd.DataFrame()
for file_list in tag_pd[tag_pd['tag']==tag_dict['target']]['data'].to_list():
    globals()[tag_dict['target']] = globals()[tag_dict['target']].append(pd.read_csv(path + file_list))

globals()[tag_dict['source']] = pd.DataFrame()
for file_list in tag_pd[tag_pd['tag']==tag_dict['source']]['data'].to_list():
    globals()[tag_dict['source']] = globals()[tag_dict['source']].append(pd.read_csv(path + file_list))

print('file length:', globals()[tag_dict['source']].shape, globals()[tag_dict['target']].shape)
print(tag_dict)

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
X_target = pd.DataFrame(target_normalizer.transform(globals()[tag_dict['target']+'_training']))

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


### Model construction

model_name = '../data/W4/' + tag_dict['target'] + '_encoded'

def lstm_ae():
    # define model
    model = Sequential(name='test')
    model.add(LSTM(units_layer_1, activation='relu', input_shape=(timesteps,n_features), return_sequences=True))
    model.add(LSTM(units_layer_2, activation='relu', return_sequences=False))
    model.add(RepeatVector(timesteps))
    model.add(LSTM(units_layer_2, activation='relu', return_sequences=True))
    model.add(LSTM(units_layer_1, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(n_features)))
    model.compile(optimizer='adam', loss='mse', metrics=[metrics.RootMeanSquaredError()])
    model.summary()

    return model


if retrain:
    early_stopping = EarlyStopping(monitor='val_loss', patience=150, verbose=2)

    model = lstm_ae ()
    lstm_ae_model = model.fit(x=Y, 
                              y=X, 
                              epochs=epoch, batch_size=16, verbose=1, validation_split=0.2, callbacks=[early_stopping])
    model.save(model_name)
else:
    model = models.load_model(model_name)

'''
if retrain:

    encoder = Sequential(
        [
            LSTM(units_layer_1, activation='tanh', return_sequences=True),
            LSTM(units_layer_2, activation='tanh', return_sequences=False),
            RepeatVector(timesteps)
        ],
        name='encoder'
    )

    decoder = Sequential(
        [
            LSTM(units_layer_2, activation='tanh', return_sequences=True),
            LSTM(units_layer_1, activation='tanh', return_sequences=True),
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
'''

if retrain:
    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(lstm_ae_model.history['loss'], marker='.', label='loss')
    ax.plot(lstm_ae_model.history['val_loss'], marker='.', label='validation loss')
    #ax.plot(lstm_ae_model.history['transformation_loss'], marker='.', label='transformation loss')

    ax.legend()
    ax.grid(True)

    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(filename+'-'+'epoches.png', dpi=300)
    plt.show()

#synthetic_data = model.predict(x={'source': Y, 'target': X}, verbose=0)
#synthetic_pd = pd.DataFrame.from_records([i[0] for i in synthetic_data])




## Validation

### Initializing the validation

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

    print(filename+'-'+tag.split(' ')[1]+'-'+tag.split(' ')[5]+'.png')
    fig.savefig(filename+'-'+tag.split(' ')[1]+'-'+tag.split(' ')[5]+'.png', dpi=300)
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


source_test_pd = pd.DataFrame()
source_test_pd = globals()[tag_dict['source']]
source_test_pd = source_test_pd.drop(columns=drop_list)

index_2 = sorted(random.sample(range(0, source_test_pd.shape[0]), shape_min))

source_test_pd = pd.DataFrame(source_normalizer.transform(source_test_pd)).iloc[index_2]

source_test_np, _ = temporalize(X = source_test_pd.values, 
                                y = np.zeros(source_test_pd.shape[0]), 
                                lookback = timesteps)

source_test_np = np.array(source_test_np)
source_test_np = source_test_np.reshape(source_test_np.shape[0], timesteps, n_features)

synthetic_source = model.predict(source_test_np, verbose=0)
synthetic_source_pd = pd.DataFrame.from_records([i[0] for i in synthetic_source])


model_source = svm.OneClassSVM(nu=0.01, kernel="rbf", gamma=0.01).fit(X_source)
model_target = svm.OneClassSVM(nu=0.01, kernel="rbf", gamma=0.01).fit(X_target)
model_synthetic = svm.OneClassSVM(nu=0.01, kernel="rbf", gamma=0.01).fit(synthetic_source_pd)

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

#sy_rmse = mean_squared_error(synthetic_score, source_score, squared=False)
sy_rmse = 0
cv_rmse = mean_squared_error(source_score_cv, source_score, squared=False)

plot_score (source_score, 
            source_date, 
            'Detect ' + tag_dict['source'] +' (source)conditions by using '+ tag_dict['source'] +' (source)model')

plot_score (source_score_cv, 
            source_date_cv, 
            'Detect ' + tag_dict['source'] +' (source)conditions by using '+ tag_dict['target'] +' (target)model, RMSE='+ "{:.3f}".format(cv_rmse))

plot_score (synthetic_score, 
            synthetic_date, 
            'Detect synthetic conditions by using '+ tag_dict['target'] +' (target)model, RMSE='+ "{:.3f}".format(sy_rmse))


### Detecting target domain by synthetic model

target_score_cv, target_date_cv = get_score(globals()[tag_dict['target']], 
                                            tag_dict['target_training_from'], 
                                            tag_dict['target_end'], 
                                            target_normalizer,
                                            model_source)

target_score, target_date = get_score(globals()[tag_dict['target']], 
                                            tag_dict['target_training_from'], 
                                            tag_dict['target_end'], 
                                            target_normalizer,
                                            model_target)

target_score_da, target_date_da = get_score(globals()[tag_dict['target']], 
                                            tag_dict['target_training_from'], 
                                            tag_dict['target_end'], 
                                            target_normalizer,
                                            model_synthetic)

da_rmse = mean_squared_error(target_score_da, target_score, squared=False)
cv_rmse = mean_squared_error(target_score_cv, target_score, squared=False)

plot_score (target_score, 
            target_date, 
            'Detect ' + tag_dict['target'] +' (target)conditions by using '+ tag_dict['target'] +' (target)model')

plot_score (target_score_cv, 
            target_date_cv, 
            'Detect ' + tag_dict['target'] +' (target)conditions by using '+ tag_dict['source'] +' (source)model, RMSE='+ "{:.3f}".format(cv_rmse))

plot_score (target_score_da, 
            target_date_da, 
            'Detect ' + tag_dict['target'] +' (target)conditions by using '+ tag_dict['source'] +' synthetic model, RMSE=' + "{:.3f}".format(da_rmse))


### show the difference between real data and synthetic data

feature_index = 206
duration = 26000
interval = 100
fig, ax = plt.subplots(figsize=(10,5))

ax.plot(range(duration, duration+interval), X_target[feature_index].head(duration).tail(interval), label='target', marker='.')
ax.plot(range(duration, duration+interval), X_source[feature_index].head(duration).tail(interval), label='source', marker='.')
ax.plot(range(duration, duration+interval), synthetic_source_pd[feature_index].head(duration).tail(interval), label='synthetic', marker='.')

ax.legend()
ax.grid(True)
plt.tight_layout()

plt.savefig(filename + '-' +'realdata.png', dpi=300)
plt.show()

