from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
import pandas as pd
import os
from datetime import datetime
from sklearn.externals import joblib
import hashlib
import pickle

dpath = '/prev-output'
data_path = os.path.join(dpath, 'train_split.csv')
ds_train = pd.read_csv(data_path)
ds_train.drop('id', inplace=True, axis=1)

ds_valid = pd.read_csv(os.path.join(dpath, 'valid_split.csv'))
ds_valid.drop('id', inplace=True, axis=1)

columns = ds_train.columns.copy()
print('convert feature ...')
for column in columns:
    print('convert', column, '...')
    if (column == 'click') or (column == 'id'):
        continue
    elif column == 'hour':
        ds_train['weekday'] = ds_train[column].map(lambda x: datetime.strptime('20' + str(x), '%Y%m%d%H').weekday())
        ds_train[column] = ds_train[column].map(lambda x: datetime.strptime('20' + str(x), '%Y%m%d%H').hour)
        ds_valid['weekday'] = ds_valid[column].map(lambda x: datetime.strptime('20' + str(x), '%Y%m%d%H').weekday())
        ds_valid[column] = ds_valid[column].map(lambda x: datetime.strptime('20' + str(x), '%Y%m%d%H').hour)
        continue
    feature_map = dict()
    for index,(feature, _) in enumerate(sorted(ds_train[column].value_counts().items(), key=lambda x: x[1], reverse=True)):
        feature_map[feature] = index
    ds_train[column] = ds_train[column].map(feature_map)

    total_index = len(set(feature_map.values()))
    for feature in ds_valid[column]:
        if feature not in feature_map:
            feature_encoded = (column + '_' + str(feature)).encode()
            index = int(hashlib.md5(feature_encoded).hexdigest(), 16) % total_index
            feature_map[feature] = index
    ds_valid[column] = ds_valid[column].map(feature_map)

with open('/output/feature_map.pickle', 'wb') as f:
    pickle.dump(feature_map, f, pickle.HIGHEST_PROTOCOL)
del feature_map

label_train = ds_train['click']
label_valid = ds_valid['click']
ds_train = ds_train.drop(['click'], axis=1).values
ds_valid = ds_valid.drop(['click'], axis=1).values

print('build gbdt model ...')
gbdt = GradientBoostingClassifier(loss='deviance',n_estimators=1000, learning_rate=0.3, max_depth=10, subsample=0.8,
                                  min_samples_split=2000, min_samples_leaf=1000, random_state=0, verbose=1, warm_start=True)


for i in range(20):
    gbdt.set_params(n_estimators=i*1000)
    print('fit model ...', i)
    gbdt.fit(ds_train, label_train)

    print('predict...')
    proba = gbdt.predict_proba(ds_valid)
    print('valid score', log_loss(label_valid, proba))
    print('dump model to output')
    joblib.dump(gbdt, '/output/gbdt' + str(i) + '.pkl')
