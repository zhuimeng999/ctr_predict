import os, sys
import logging
import collections
import pickle

logger = logging.getLogger('FE')

if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))

_RAW_COLUMN_NAMES = \
                ['id',          'click',            'hour',         'C1',           'banner_pos',           'site_id',
                 'site_domain', 'site_category',    'app_id',       'app_domain',   'app_category',         'device_id',
                 'device_ip',   'device_model',     'device_type',  'device_conn_type', 'C14',              'C15',
                 'C16',         'C17',              'C18',          'C19',          'C20',                  'C21']
progress_step = 400000
train_filename = '../data/train.csv'
test_filename = '../data/test.csv'

train_dump_filename = 'train_count.pickle'
valid_dump_filename = 'valid_count.pickle'
test_dump_filename = 'test_count.pickle'

output_dir = 'info'
train_dump_path = os.path.join(output_dir, train_dump_filename)
valid_dump_path = os.path.join(output_dir, valid_dump_filename)
test_dump_path = os.path.join(output_dir, test_dump_filename)

feature_number = len(_RAW_COLUMN_NAMES) - 1
train_count_list = [collections.defaultdict(lambda: 0) for _ in range(feature_number)]
valid_count_list = [collections.defaultdict(lambda: 0) for _ in range(feature_number)]
test_count_list = [collections.defaultdict(lambda: 0) for _ in range(feature_number)]

with open(train_filename, 'r') as f:
    header = f.readline().strip().split(',')
    for line_no, line in enumerate(f):
        features = line.strip().split(',')
        if features[2][4:6] != '30':
            for index, feature in enumerate(features[1:]):
                train_count_list[index][feature] += 1
        else:
            for index, feature in enumerate(features[1:]):
                valid_count_list[index][feature] += 1
        if (line_no % progress_step) == 0:
            logger.info('train valid progress %d', line_no)
    logger.info('train valid progress %d, done!!!', line_no)

with open(test_filename, 'r') as f:
    header = f.readline().strip().split(',')
    for line_no, line in enumerate(f):
        features = line.strip().split(',')
        for index, feature in enumerate(features[1:]):
                valid_count_list[index][feature] += 1
        if (line_no % progress_step) == 0:
            logger.info('test progress %d', line_no)
    logger.info('test progress %d, done!!!', line_no)

train_count_dict = dict(map(lambda x: (x[0], dict(x[1])), zip(_RAW_COLUMN_NAMES[1:], train_count_list)))
valid_count_dict = dict(map(lambda x: (x[0], dict(x[1])), zip(_RAW_COLUMN_NAMES[1:], valid_count_list)))
test_count_dict = dict(map(lambda x: (x[0], dict(x[1])), zip(_RAW_COLUMN_NAMES[2:], test_count_list)))

with open(train_dump_path, 'wb') as f:
    pickle.dump(train_count_dict, f, pickle.HIGHEST_PROTOCOL)

with open(valid_dump_path, 'wb') as f:
    pickle.dump(valid_count_dict, f, pickle.HIGHEST_PROTOCOL)

with open(test_dump_path, 'wb') as f:
    pickle.dump(test_count_dict, f, pickle.HIGHEST_PROTOCOL)
