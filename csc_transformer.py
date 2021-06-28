from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

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

# Setup GPU
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
gpu_devices = tensorflow.config.experimental.list_physical_devices('GPU')
tensorflow.config.experimental.set_memory_growth(gpu_devices[0], True)
gpus = tensorflow.test.gpu_device_name()


filename = source[-3:]+'_'+target[-3:]+'_'+str(epoch)+'_'+ \
                    str(timesteps)+'_'+str(units_layer_1)+'_'+str(units_layer_2)


if os.path.isfile('results/'+filename + '-' +'hist.png'):
    logging.info(filename + '-' +'hist.png' + ' exists, exit')
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


    return globals()[tag_dict['source']+'_training'], globals()[tag_dict['target']+'_training'], globals()[tag_dict['source']], globals()[tag_dict['target']], tag_dict

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
    model.add(LSTM(units_layer_1, activation='relu', input_shape=(timesteps,n_features), return_sequences=True))
    model.add(LSTM(units_layer_2, activation='relu', return_sequences=False))
    model.add(RepeatVector(timesteps))
    model.add(LSTM(units_layer_2, activation='relu', return_sequences=True))
    model.add(LSTM(units_layer_1, activation='relu', return_sequences=True))
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
    logging.info('plot_score')

    fig, ax = plt.subplots(figsize=(10, 2.5))
    plt.xticks(rotation=45)
    ax.plot(date_list, score_list, '.-')
    #ax.set(xlabel='date', ylabel='score', title=tag)
    ax.set(xlabel='date', ylabel='score')
    ax.grid()
    plt.ylim(0, 100)
    plt.tight_layout()

    print(filename+'-'+tag.split(' ')[1]+'-'+tag.split(' ')[5]+'.png')
    fig.savefig('results/'+filename+'-'+tag.split(' ')[1]+'-'+tag.split(' ')[5]+'.png', dpi=300)
    plt.show()

def get_score (data_df, start_date, end_date, normalizer, prediction_model):
    logging.info('get_score')

    score_list = []
    date_list = []
    delta = datetime.timedelta(days=1)

    drop_list = ['Unnamed: 0', '_id','type','scada','timestamp','device', 'datetime']

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

def get_synthetic_data (data, lstm_model, normalizer):
    logging.info('get_synthetic_data')

    drop_list = ['Unnamed: 0', '_id','type','scada','timestamp','device', 'datetime']

    target_pd = data.drop(columns=drop_list)
    target_pd = pd.DataFrame(normalizer.transform(target_pd))

    target_np, _ = temporalize(X = target_pd.values, 
                                    y = np.zeros(target_pd.shape[0]), 
                                    lookback = timesteps)

    synthetic_np = lstm_model.predict(target_np, verbose=0)
    synthetic_pd = pd.DataFrame.from_records([i[0] for i in synthetic_np])

    synthetic_pd['datetime'] = data['datetime'].tail(len(synthetic_pd)).values

    return synthetic_pd

def training_ocsvm_models (X_source, X_target, X_synthetic):
    logging.info('training_ocsvm_models')

    model_source = svm.OneClassSVM(nu=0.01, kernel="rbf", gamma=0.01).fit(X_source)
    model_target = svm.OneClassSVM(nu=0.01, kernel="rbf", gamma=0.01).fit(X_target)
    model_synthetic = svm.OneClassSVM(nu=0.01, kernel="rbf", gamma=0.01).fit(X_synthetic)

    return model_source, model_target, model_synthetic

def get_syntheic_score (data_df, start_date, end_date, prediction_model):
    logging.info('get_syntheic_score')

    score_list = []
    date_list = []
    delta = datetime.timedelta(days=1)

    while start_date <= end_date:

        validation_df = data_df[(data_df['datetime'] > start_date) & 
                                (data_df['datetime'] <= start_date + delta)]

        if validation_df.shape[0] > 0:

            validation_df_score = validation_df.drop(columns=['datetime'])
            validation_df_score = prediction_model.predict(validation_df_score)

            score_list.append(scorer_(validation_df_score))
            date_list.append(start_date)


        start_date += delta
        
    return score_list, date_list

def lstm_ae_2():
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

    return model


source_training, target_training, source_validation, target_validation, tag_dict = data_loader (sys.argv[1], sys.argv[2])

normalizer = preprocessing.MinMaxScaler()
source_normalizer = normalizer.fit(source_training)

normalizer = preprocessing.MinMaxScaler()
target_normalizer = normalizer.fit(target_training)

X_source = pd.DataFrame(source_normalizer.transform(source_training))
X_target = pd.DataFrame(target_normalizer.transform(target_training))

#X_source = pd.DataFrame(source_training)
#X_target = pd.DataFrame(target_training)

X, Y = resample(X_source, X_target)

X, _ = temporalize(X = X.values, y = np.zeros(len(X)), lookback = timesteps)
Y, _ = temporalize(X = Y.values, y = np.zeros(len(Y)), lookback = timesteps)

logging.info('source shape (after temporalize):' + str(Y.shape))
logging.info('target shape (after temporalize):' + str(X.shape))

lstm_model = training_lstm_model(X, Y)
X_synthetic = get_synthetic_data(target_validation, lstm_model, source_normalizer)

logging.info('X_synthetic shape:' + str(X_synthetic.shape))
logging.info('X_source shape:' + str(source_validation.shape))
logging.info('X_target shape:' + str(target_validation.shape))

#normalizer = preprocessing.MinMaxScaler()
#synthetic_normalizer = normalizer.fit(X_synthetic)

model_source, model_target, model_synthetic = training_ocsvm_models (X_source, 
                                                                     X_target, 
                                                                     X_synthetic.drop(columns=['datetime']))


### RQ1: Detecting source domain by synthetic model
rq1_score, rq1_date = get_score(source_validation, 
                                            tag_dict['source_training_from'], 
                                            tag_dict['source_end'], 
                                            source_normalizer,
                                            model_synthetic)    

source_score_syn, source_date_syn = get_score(source_validation, 
                                            tag_dict['source_training_from'], 
                                            tag_dict['source_end'], 
                                            source_normalizer,
                                            model_synthetic)    

### RQ2: Detecting target->synthetic domain by source model
rq2_score, rq2_date = get_syntheic_score (X_synthetic, 
                                            tag_dict['target_training_from'], 
                                            tag_dict['target_end'],
                                            model_source)

syn_score, syn_date = get_syntheic_score (X_synthetic, 
                                            tag_dict['target_training_from'], 
                                            tag_dict['target_end'],
                                            model_source)             

## Cross-validation
source_score_cv, source_date_cv = get_score(source_validation, 
                                            tag_dict['source_training_from'], 
                                            tag_dict['source_end'], 
                                            source_normalizer,
                                            model_target)

target_score_cv, target_date_cv = get_score(target_validation, 
                                            tag_dict['target_training_from'], 
                                            tag_dict['target_end'], 
                                            target_normalizer,
                                            model_source)

# Benchmark
source_score, source_date = get_score(source_validation, 
                                            tag_dict['source_training_from'], 
                                            tag_dict['source_end'], 
                                            source_normalizer,
                                            model_source)

target_score, target_date = get_score(target_validation, 
                                            tag_dict['target_training_from'], 
                                            tag_dict['target_end'], 
                                            target_normalizer,
                                            model_target)

                                       
#sy_rmse = mean_squared_error(synthetic_score, source_score, squared=False)
N = min(len(target_score), len(source_score), len(rq1_score), len(rq2_score), len(source_score_syn), len(syn_score))


logging.info('target score shape:' + str(len(target_score)))
logging.info('source score shape:' + str( len(source_score)))
logging.info('rq1 score shape:' + str( len(rq1_score)))
logging.info('rq2 score shape:' + str( len(rq2_score)))

rq1_rmse = mean_squared_error (rq1_score[-N:], source_score[-N:], squared=False)
rq1_cv_rmse = mean_squared_error (target_score_cv[-N:], target_score[-N:], squared=False)
rq2_rmse = mean_squared_error (rq2_score[-N:], target_score[-N:], squared=False)
rq2_cv_rmse = mean_squared_error (source_score_cv[-N:], source_score[-N:], squared=False)


# log optimun record
record_pd = pd.read_csv('csc_record.csv')
os.remove('csc_record.csv')
record_list = [source, target, epoch, timesteps, units_layer_1, units_layer_2, rq1_rmse, rq2_rmse]
record_pd.iloc[len(record_pd)] = record_list
record_pd.to_csv('csc_record.csv')

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

plot_score (source_score_syn, 
            source_date_syn, 
            'Detect ' + tag_dict['source'][-3:] +' (target)conditions by using '+ 'synthetic model, RMSE='+ "{:.3f}".format(rq1_cv_rmse))

plot_score (syn_score, 
            syn_date, 
            'Detect synthetic conditions by using '+ 'synthetic model, RMSE='+ "{:.3f}".format(rq1_cv_rmse))



### show the difference between real data and synthetic data
#feature_index = 206
feature_index = 1
duration = 26000
interval = 100
fig, ax = plt.subplots(figsize=(6,2))

ax.plot(range(duration, duration+interval), X_target[feature_index].head(duration).tail(interval), label='target', marker='.', color='tab:blue', linewidth=1)
ax.plot(range(duration, duration+interval), X_source[feature_index].head(duration).tail(interval), label='source', marker='.', color='tab:red', linewidth=1)
ax.plot(range(duration, duration+interval), X_synthetic[feature_index].head(duration).tail(interval), label='synthetic', marker='.', color='tab:green', linewidth=1)

ax.legend()
ax.grid(True)
plt.xlabel('timestamp')
plt.ylabel('value')
plt.tight_layout()

plt.savefig('results/'+filename + '-' +'realdata.png', dpi=300)
plt.show()


### show distribution
pca_scale = PCA(n_components=2)
pca_scale = pca_scale.fit(X_source)

X_source_dist = pca_scale.transform(X_source)
x_min, x_max = X_source_dist.min(0), X_source_dist.max(0)
X_norm = (X_source_dist-x_min) / (x_max-x_min)  #Normalize
X_source_dist_df = pd.DataFrame(X_norm, columns = ['dim1','dim2'])

X_target_dist = pca_scale.transform(X_target)
x_min, x_max = X_target_dist.min(0), X_target_dist.max(0)
X_norm = (X_target_dist-x_min) / (x_max-x_min)  #Normalize
X_target_dist_df = pd.DataFrame(X_norm, columns = ['dim1','dim2'])

X_synthetic_dist = pca_scale.transform(X_synthetic.drop(columns=['datetime']))
x_min, x_max = X_synthetic_dist.min(0), X_synthetic_dist.max(0)
X_norm = (X_synthetic_dist-x_min) / (x_max-x_min)  #Normalize
X_synthetic_dist_df = pd.DataFrame(X_norm, columns = ['dim1','dim2'])

fig, ax = plt.subplots(figsize=(6,2))

ax.scatter(X_source_dist_df['dim1'], X_source_dist_df['dim2'], alpha=0.3, label='source', color='tab:blue')
ax.scatter(X_target_dist_df['dim1'], X_target_dist_df['dim2'], alpha=0.3, label='target', color='tab:red')
ax.scatter(X_synthetic_dist_df['dim1'], X_synthetic_dist_df['dim2'], alpha=0.3, label='synthetic', color='tab:green')

ax.legend(loc='best')
ax.grid(True)

#plt.ylim(0, 1)
#plt.xlim(0, 1)
plt.savefig('results/'+filename + '-' +'distribution.png', dpi=300)
plt.show()




### show histogram ###
fig, ax = plt.subplots(figsize=(6,2))

#ax.scatter(X_source_dist_df['dim1'], X_source_dist_df['dim2'], alpha=0.1, label='source')
#ax.scatter(X_target_dist_df['dim1'], X_target_dist_df['dim2'], alpha=0.1, label='target')
#ax.scatter(X_synthetic_dist_df['dim1'], X_synthetic_dist_df['dim2'], alpha=0.1, label='synthetic')
a = X_target[feature_index]
b = X_source[feature_index]
c = X_synthetic[feature_index]
bins=np.histogram(np.hstack((a, b, c)), bins=80)[1] #get the bin edges

plt.hist(a, alpha=0.5, label='target', color='tab:blue', bins=bins)
plt.hist(b, alpha=0.5, label='source', color='tab:red', bins=bins)
plt.hist(c, alpha=0.5, label='synthetic', color='tab:green', bins=bins)

plt.xlabel('values')
plt.ylabel('frequency')

ax.legend()
ax.grid(True)

#plt.ylim(0, 1)
plt.xlim(0, 1)
plt.savefig('results/'+filename + '-' +'hist.png', dpi=300)
plt.show()