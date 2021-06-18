from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import mean_squared_error

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import random
import sys
import os
import tensorflow
import datetime
import logging
import warnings

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

# setup logger
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# python csc_transformer.py W4662FM0605 W4662FM0606 1 4 128 64 0

logging.info('argv:' + str(sys.argv))

source = sys.argv[1]
target = sys.argv[2]
epoch = int(sys.argv[3])
timesteps = int(sys.argv[4])
units_layer_1 = int(sys.argv[5])
units_layer_2 = int(sys.argv[6])
gpu_num = int(sys.argv[7])

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
gpu_devices = tensorflow.config.experimental.list_physical_devices('GPU')
tensorflow.config.experimental.set_memory_growth(gpu_devices[0], True)
gpus = tensorflow.test.gpu_device_name()

filename = source[-3:]+'_'+target[-3:]+'_'+str(epoch)+'_'+ \
                    str(timesteps)+'_'+str(units_layer_1)+'_'+str(units_layer_2)


if os.path.isfile('results/'+filename + '-' +'realdata.png'):
    logging.info(filename + '-' +'realdata.png' + ' exists, exit')
    exit()


n_features = 390
retrain = True
path = '/data/'





def data_loader (source, target):
    logging.info('data_loader')

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

    tag_dict = {'source':source,
                'source_training_from': datetime.datetime(source_training_from_year,source_training_from_month,1,0,0),
                'source_training_to': datetime.datetime(source_training_to_year,source_training_to_month,1,0,0),
                'source_end': datetime.datetime(source_end_year,source_end_month,1,0,0),
                
                'target':target,
                'target_training_from': datetime.datetime(target_training_from_year,target_training_from_month,1,0,0), 
                'target_training_to': datetime.datetime(target_training_to_year,target_training_to_month,1,0,0), 
                'target_end': datetime.datetime(target_end_year,target_end_month,1,0,0)}

    globals()[tag_dict['target']] = pd.DataFrame()
    for file_list in tag_pd[tag_pd['tag']==tag_dict['target']]['data'].to_list():
        globals()[tag_dict['target']] = globals()[tag_dict['target']].append(pd.read_csv(path + file_list))

    globals()[tag_dict['source']] = pd.DataFrame()
    for file_list in tag_pd[tag_pd['tag']==tag_dict['source']]['data'].to_list():
        globals()[tag_dict['source']] = globals()[tag_dict['source']].append(pd.read_csv(path + file_list))

    logging.info('source shape:' + str(globals()[tag_dict['target']].shape))
    logging.info('target shape:' + str(globals()[tag_dict['source']].shape))


    globals()[tag_dict['source']]['datetime'] = globals()[tag_dict['source']]['timestamp'].astype('int').astype("datetime64[s]")
    globals()[tag_dict['target']]['datetime'] = globals()[tag_dict['target']]['timestamp'].astype('int').astype("datetime64[s]")

    globals()[tag_dict['source']+'_training'] = globals()[tag_dict['source']][
                                                    (globals()[tag_dict['source']]['datetime'] > tag_dict['source_training_from']) &
                                                    (globals()[tag_dict['source']]['datetime'] <= tag_dict['source_training_to']) ]

    globals()[tag_dict['target']+'_training'] = globals()[tag_dict['target']][
                                                    (globals()[tag_dict['target']]['datetime'] > tag_dict['target_training_from']) &
                                                    (globals()[tag_dict['target']]['datetime'] <= tag_dict['target_training_to']) ]

    drop_list = ['Unnamed: 0', '_id','type','scada','timestamp','device', 'datetime']

    globals()[tag_dict['source']+'_training'] = globals()[tag_dict['source']+'_training'].drop(columns=drop_list)
    globals()[tag_dict['target']+'_training'] = globals()[tag_dict['target']+'_training'].drop(columns=drop_list)


    return globals()[tag_dict['source']+'_training'], 
            globals()[tag_dict['target']+'_training'], 
            globals()[tag_dict['source']], 
            globals()[tag_dict['target']]

def get_shapes (data_1, data_2):

    logging.info('get_shapes')

    shape_min = min (data_1.shape[0], data_2.shape[0])
    shape_max = max (data_1.shape[0], data_2.shape[0])

    return shape_min, shape_max

def resample (data_1, data_2):

    logging.info('resample')

    shape_min, shape_max = get_shapes (source_training, target_training)
    index = sorted(random.sample(range(0, shape_max), shape_min))

    if len(X_target) > len(X_source):
        X = X_target.iloc[index]
        Y = X_source
    else:
        X = X_target
        Y = X_source.iloc[index]

    return X, Y

def temporalize (X, y, lookback):
    logging.info('temporalize')

    output_X = []
    output_y = []

    for i in range(X.shape[0]-lookback-1):
        t = []
        for j in range(1,lookback+1):
            # Gather past records upto the lookback period
            t.append(X[[(i+j+1)], :])
        output_X.append(t)
        output_y.append(y[i+lookback+1])

    output_X = np.array(output_X)
    output_X = output_X.reshape(output_X.shape[0], timesteps, n_features)

    return output_X, output_y

def lstm_ae():
    # define model
    model = Sequential(name='test')
    model.add(LSTM(units_layer_1, activation='tanh', input_shape=(timesteps,n_features), return_sequences=True))
    model.add(LSTM(units_layer_2, activation='tanh', return_sequences=False))
    model.add(RepeatVector(timesteps))
    model.add(LSTM(units_layer_2, activation='tanh', return_sequences=True))
    model.add(LSTM(units_layer_1, activation='tanh', return_sequences=True))
    model.add(TimeDistributed(Dense(n_features)))
    model.compile(optimizer='adam', loss='mse', metrics=[metrics.RootMeanSquaredError()])
    model.summary()

    return model

def training_lstm_model (input_data, output_data):

    early_stopping = EarlyStopping(monitor='val_loss', patience=150, verbose=2)

    model = lstm_ae ()
    lstm_ae_model = model.fit(x=input_data, 
                              y=output_data, 
                              epochs=epoch, 
                              batch_size=16, 
                              verbose=1, 
                              validation_split=0.2, 
                              callbacks=[early_stopping])

    model_name = 'model/' + filename
    model.save(model_name)

    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(lstm_ae_model.history['loss'], marker='.', label='loss')
    ax.plot(lstm_ae_model.history['val_loss'], marker='.', label='validation loss')
    #ax.plot(lstm_ae_model.history['transformation_loss'], marker='.', label='transformation loss')

    ax.legend()
    ax.grid(True)

    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig('results/'+filename+'-'+'epoches.png', dpi=300)
    plt.show()

    return model



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
    fig.savefig('results/'+filename+'-'+tag.split(' ')[1]+'-'+tag.split(' ')[5]+'.png', dpi=300)
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

def get_synthetic_data (data, lstm_model):

    drop_list = ['Unnamed: 0', '_id','type','scada','timestamp','device', 'datetime']

    source_test_pd = data.drop(columns=drop_list)

    #index_2 = sorted(random.sample(range(0, source_test_pd.shape[0]), shape_min))

    source_test_pd = pd.DataFrame(source_normalizer.transform(source_test_pd))

    source_test_np, _ = temporalize(X = source_test_pd.values, 
                                    y = np.zeros(source_test_pd.shape[0]), 
                                    lookback = timesteps)

    #source_test_np = np.array(source_test_np)
    #source_test_np = source_test_np.reshape(source_test_np.shape[0], timesteps, n_features)

    synthetic_source = lstm_model.predict(source_test_np, verbose=0)
    synthetic_source_pd = pd.DataFrame.from_records([i[0] for i in synthetic_source])
    synthetic_source_pd['datatime'] = data['datatime']

    return synthetic_source_pd

def training_ocsvm_models (X_source, X_target, X_synthetic):

    model_source = svm.OneClassSVM(nu=0.01, kernel="rbf", gamma=0.01).fit(X_source)
    model_target = svm.OneClassSVM(nu=0.01, kernel="rbf", gamma=0.01).fit(X_target)
    model_synthetic = svm.OneClassSVM(nu=0.01, kernel="rbf", gamma=0.01).fit(X_synthetic)

    return model_source, model_target, model_synthetic

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





source_training, target_training, source_validation, target_validation = data_loader (sys.argv[1], sys.argv[2])
#shape_min, shape_max = get_shapes (source_training, target_training)

normalizer = preprocessing.MinMaxScaler()
source_normalizer = normalizer.fit(source_training)

normalizer = preprocessing.MinMaxScaler()
target_normalizer = normalizer.fit(target_training)

X_source = pd.DataFrame(source_normalizer.transform(source_training))
X_target = pd.DataFrame(target_normalizer.transform(target_training))

X, Y = resample(X_source, X_target)

#X, y = temporalize(X = timeseries, y = np.zeros(len(timeseries)), lookback = timesteps)
X, _ = temporalize(X = X.values, y = np.zeros(len(X)), lookback = timesteps)
Y, _ = temporalize(X = Y.values, y = np.zeros(len(Y)), lookback = timesteps)

logging.info('source shape (after temporalize):' + str(Y.shape))
logging.info('target shape (after temporalize):' + str(X.shape))

lstm_model = training_lstm_model(X, Y)

X_synthetic = get_synthetic_data(target_validation, lstm_model)
logging.info('X_synthetic shape:' + str(X_synthetic.shape))
logging.info('X_source shape:' + str(X_source.shape))
logging.info('X_target shape:' + str(X_target.shape))

model_source, model_target, model_synthetic = training_ocsvm_models (X_source, X_target, X_synthetic.drop(columns=['datetime']))



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
    plt.savefig('results/'+filename+'-'+'epoches.png', dpi=300)
    plt.show()
'''
#synthetic_data = model.predict(x={'source': Y, 'target': X}, verbose=0)
#synthetic_pd = pd.DataFrame.from_records([i[0] for i in synthetic_data])




## Validation

### Initializing the validation


#X_synthetic, index_2 = get_synthetic_data(globals()[tag_dict['target']], lstm_model, drop_list, shape_min)
#index_2 = sorted(random.sample(range(0, X_target.shape[0]), shape_min))
#X_synthetic = lstm_model.predict(X, verbose=0)
#X_synthetic = pd.DataFrame.from_records([i[0] for i in X_synthetic])
#model_source, model_target, model_synthetic = training_ocsvm_models (X_source, X_target, X_synthetic)

#for elements in drop_list:
#    X_synthetic[elements] = globals()[tag_dict['target']][elements].iloc[index_2].tail(X_synthetic.shape[0]).values

# logging.info('source shape:' + str(globals()[tag_dict['source']].shape))
# logging.info('target shape:' + str(globals()[tag_dict['target']].shape))
# logging.info('X_synthetic shape:' + str(X_synthetic.shape))


### RQ1: Detecting target domain by synthetic model
rq1_score, rq1_date = get_score(globals()[tag_dict['source']], 
                                            tag_dict['source_training_from'], 
                                            tag_dict['source_end'], 
                                            source_normalizer,
                                            model_synthetic)    

### RQ2: Detecting target domain by synthetic model
rq2_score, rq2_date = get_syntheic_score(X_synthetic, 
                                            tag_dict['target_training_from'], 
                                            tag_dict['target_end'], 
                                            model_source)                                             

## Cross-validation
source_score_cv, source_date_cv = get_score(globals()[tag_dict['source']], 
                                            tag_dict['source_training_from'], 
                                            tag_dict['source_end'], 
                                            source_normalizer,
                                            model_target)

target_score_cv, target_date_cv = get_score(globals()[tag_dict['target']], 
                                            tag_dict['target_training_from'], 
                                            tag_dict['target_end'], 
                                            target_normalizer,
                                            model_source)

# Benchmark
source_score, source_date = get_score(globals()[tag_dict['source']], 
                                            tag_dict['source_training_from'], 
                                            tag_dict['source_end'], 
                                            source_normalizer,
                                            model_source)


target_score, target_date = get_score(globals()[tag_dict['target']], 
                                            tag_dict['target_training_from'], 
                                            tag_dict['target_end'], 
                                            target_normalizer,
                                            model_target)

                                       
#sy_rmse = mean_squared_error(synthetic_score, source_score, squared=False)
N=len(target_score)

rq1_rmse = mean_squared_error (rq1_score[-N:], target_score, squared=False)
rq1_cv_rmse = mean_squared_error (target_score_cv, target_score, squared=False)
rq2_rmse = mean_squared_error (rq2_score, target_score[-N:], squared=False)
rq2_cv_rmse = mean_squared_error (source_score_cv, source_score, squared=False)


# rq2
plot_score (rq2_score, 
            rq2_date, 
            'Detect synthetic conditions by using '+ tag_dict['source'][-3:] +' (source)model, RMSE='+ "{:.3f}".format(rq2_rmse))

# rq1
plot_score (rq1_score, 
            rq1_date, 
            'Detect ' + tag_dict['source'][-3:] +' (source)conditions by using '+ 'synthetic model, RMSE=' + "{:.3f}".format(rq1_rmse))

plot_score (source_score, 
            source_date, 
            'Detect ' + tag_dict['source'][-3:] +' (source)conditions by using '+ tag_dict['source'][-3:] +' (source)model')

plot_score (source_score_cv, 
            source_date_cv, 
            'Detect ' + tag_dict['source'][-3:] +' (source)conditions by using '+ tag_dict['target'][-3:] +' (target)model, RMSE='+ "{:.3f}".format(rq2_cv_rmse))

plot_score (target_score, 
            target_date, 
            'Detect ' + tag_dict['target'][-3:] +' (target)conditions by using '+ tag_dict['target'][-3:] +' (target)model')

plot_score (target_score_cv, 
            target_date_cv, 
            'Detect ' + tag_dict['target'][-3:] +' (target)conditions by using '+ tag_dict['source'][-3:] +' (source)model, RMSE='+ "{:.3f}".format(rq1_cv_rmse))







### show the difference between real data and synthetic data

feature_index = 206
duration = 26000
interval = 100
fig, ax = plt.subplots(figsize=(10,5))

ax.plot(range(duration, duration+interval), X_target[feature_index].head(duration).tail(interval), label='target', marker='.')
ax.plot(range(duration, duration+interval), X_source[feature_index].head(duration).tail(interval), label='source', marker='.')
ax.plot(range(duration, duration+interval), X_synthetic[feature_index].head(duration).tail(interval), label='synthetic', marker='.')

ax.legend()
ax.grid(True)
plt.tight_layout()

plt.savefig('results/'+filename + '-' +'realdata.png', dpi=300)
plt.show()

