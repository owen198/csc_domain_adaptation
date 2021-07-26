import pandas as pd
import logging
import warnings
import sys
import datetime

from sklearn import preprocessing
from sklearn import svm

# setup logger
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

path = './data/W4/'


def loader (source, target):
    logging.info(source+'_'+target+'_'+'data_loader')

    tag_pd = pd.read_csv('csc_w4.csv')

    source_training_from_month = int(tag_pd[(tag_pd['tag']==source) & (tag_pd['Normal']==1)].head(1)['data'].values[0].split('_')[1].split('.')[0][4:6])
    source_training_from_year = int(tag_pd[(tag_pd['tag']==source) & (tag_pd['Normal']==1)].head(1)['data'].values[0].split('_')[1].split('.')[0][0:4])
    source_training_to_month = int(tag_pd[(tag_pd['tag']==source) & (tag_pd['Normal']==1)].tail(1)['data'].values[0].split('_')[1].split('.')[0][4:6])
    source_training_to_year = int(tag_pd[(tag_pd['tag']==source) & (tag_pd['Normal']==1)].tail(1)['data'].values[0].split('_')[1].split('.')[0][0:4])
    source_end_month = int(tag_pd[(tag_pd['tag']==source) & (tag_pd['Normal']==0)].tail(1)['data'].values[0].split('_')[1].split('.')[0][4:6])
    source_end_year = int(tag_pd[(tag_pd['tag']==source) & (tag_pd['Normal']==0)].tail(1)['data'].values[0].split('_')[1].split('.')[0][0:4])

    target_training_from_month = int(tag_pd[(tag_pd['tag']==target) & (tag_pd['Normal']==1)].head(1)['data'].values[0].split('_')[1].split('.')[0][4:6])
    target_training_from_year = int(tag_pd[(tag_pd['tag']==target) & (tag_pd['Normal']==1)].head(1)['data'].values[0].split('_')[1].split('.')[0][0:4])
    target_training_to_month = int(tag_pd[(tag_pd['tag']==target) & (tag_pd['Normal']==1)].tail(1)['data'].values[0].split('_')[1].split('.')[0][4:6])
    target_training_to_year = int(tag_pd[(tag_pd['tag']==target) & (tag_pd['Normal']==1)].tail(1)['data'].values[0].split('_')[1].split('.')[0][0:4])
    target_end_month = int(tag_pd[(tag_pd['tag']==target) & (tag_pd['Normal']==0)].tail(1)['data'].values[0].split('_')[1].split('.')[0][4:6])
    target_end_year = int(tag_pd[(tag_pd['tag']==target) & (tag_pd['Normal']==0)].tail(1)['data'].values[0].split('_')[1].split('.')[0][0:4])

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


    return globals()[tag_dict['source']], globals()[tag_dict['target']], tag_dict

def normalization (normal_df):
    
    min_max_scaler = preprocessing.MinMaxScaler()
    feature_names = list(normal_df)
    
    min_max_scaler = min_max_scaler.fit(normal_df.values)
    X_raw_minmax = min_max_scaler.transform(normal_df.values)
    normal_df = pd.DataFrame(X_raw_minmax, columns=feature_names)

    return normal_df, min_max_scaler

def scorer_(Y_pred):
    a = (Y_pred[Y_pred == -1].size)/(Y_pred.size)
    return a*100

def labeler (data_df, training_from, training_to, end):

    drop_list = ['Unnamed: 0', '_id','type','scada','timestamp','device', 'datetime']

    oneClass_predictor = svm.OneClassSVM(nu=0.01, kernel="rbf", gamma=0.01)

    #print(tag_dict)
    #print(type(tag_dict['source_training_to']))

    training_df = data_df[(data_df['datetime'] > training_from) &
                          (data_df['datetime'] < training_to)]

    training_df = training_df.drop(columns=drop_list)
    training_df, normalizer = normalization(training_df)
    predict_model = oneClass_predictor.fit(training_df)

    data_df['label'] = 0

    #for index, row in training_df.iterrows():
    #    print(index, predict_model.predict(row))


    delta = datetime.timedelta(days=1)
    score_list = []
    datetime_list = []

    s_date = training_from
    e_date = end

    while s_date <= e_date:

        validation_df = data_df[(data_df['datetime'] > s_date) & 
                                (data_df['datetime'] <= s_date + delta)]

        if len(validation_df) > 0:
        
            validation_df = validation_df.drop(columns=drop_list)
            validation_df = validation_df.drop(columns=['label'])
            validation_df = normalizer.transform(validation_df)
            validation_df = predict_model.predict(validation_df)

            if scorer_(validation_df) > 50:
                index = data_df[(data_df['datetime'] > s_date) & (data_df['datetime'] <= s_date + delta)].index
                for ix in index:
                    data_df.at[ix, 'label']=1

        s_date += delta

    data_df = data_df.drop(columns=drop_list)
    return data_df