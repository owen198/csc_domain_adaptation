import pandas as pd
import logging
import warnings
import sys

# setup logger
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
warnings.filterwarnings("ignore", category=DeprecationWarning) 


def loader (source, target):
    logging.info(source+'_'+target+'_'+'data_loader')

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
