from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import os
from datetime import datetime
from sklearn.externals import joblib

dpath = '/prev-output'
data_path = os.path.join(dpath, 'train_split.csv')
ds = pd.read_csv(data_path, nrows=500000)
ds.drop('id', inplace=True, axis=1)

columns = ds.columns.copy()
print('convert feature ...')
for column in columns:
    print('convert', column, '...')
    if (column == 'click') or (column == 'id'):
        continue
    elif column == 'hour':
        ds['weekday'] = ds[column].map(lambda x: datetime.strptime('20' + str(x), '%Y%m%d%H').weekday())
        ds[column] = ds[column].map(lambda x: datetime.strptime('20' + str(x), '%Y%m%d%H').hour)
        continue
    feature_map = dict()
    for index,(feature, _) in enumerate(sorted(ds[column].value_counts().items(), key=lambda x: x[1], reverse=True)):
        feature_map[feature] = index
    ds[column] = ds[column].map(feature_map)
del feature_map

print('build gbdt model ...')
gbdt = GradientBoostingClassifier(loss='deviance',n_estimators=2000, learning_rate=0.3, max_depth=10, subsample=0.8,
                                  min_samples_split=2000, min_samples_leaf=1000, random_state=0, verbose=1)

label = ds['click']
ds.drop(['click'], inplace=True, axis=1)

print('fit model ...')
gbdt.fit(ds, label)

print('dump model to output')
joblib.dump(gbdt, '/output/gbdt.pkl')
