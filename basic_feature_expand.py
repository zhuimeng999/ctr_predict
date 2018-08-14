import os, pickle
from datetime import datetime


doutpath = 'output/feature_dict'


def dump_to_pickle(obj, filename : str):
    with open(os.path.join(doutpath, filename), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def basic_expand_basic(count_dict: dict, feature: str):
    tmp = list(count_dict[feature].items())
    feature_expand_dict = {}
    for i, context in enumerate(tmp):
            feature_expand_dict[context[0]] = i
    dump_to_pickle(feature_expand_dict, feature.lower() + '_expand_dict.pickle')


def basic_hour_expand(count_dict: dict):
    hour_expand_dict = {}
    for k in count_dict['hour']:
        curr_time = datetime.strptime('20' + k, '%Y%m%d%H')
        hour_expand_dict[k] = (curr_time.weekday(), curr_time.hour)
    dump_to_pickle(hour_expand_dict, 'hour_expand_dict.pickle')


def basic_feature_expand():
    with open(os.path.join('output', 'value_counts_train.pickle'), 'rb') as f:
        count_dict_train = pickle.load(f)
    basic_hour_expand(count_dict_train)
    basic_expand_basic(count_dict_train, 'C1')
    basic_expand_basic(count_dict_train, 'banner_pos')
    basic_expand_basic(count_dict_train, 'site_id')
    basic_expand_basic(count_dict_train, 'site_domain')
    basic_expand_basic(count_dict_train, 'site_category')
    basic_expand_basic(count_dict_train, 'device_id')
    basic_expand_basic(count_dict_train, 'device_ip')
    basic_expand_basic(count_dict_train, 'device_model')
    basic_expand_basic(count_dict_train, 'device_conn_type')
    basic_expand_basic(count_dict_train, 'C14')
    basic_expand_basic(count_dict_train, 'C15')
    basic_expand_basic(count_dict_train, 'C16')
    basic_expand_basic(count_dict_train, 'C17')
    basic_expand_basic(count_dict_train, 'C18')
    basic_expand_basic(count_dict_train, 'C19')
    basic_expand_basic(count_dict_train, 'C20')
    basic_expand_basic(count_dict_train, 'C21')


if __name__ == '__main__':
    basic_feature_expand()
