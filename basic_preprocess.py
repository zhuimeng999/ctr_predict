import os
import collections
import argparse
import pickle
from datetime import datetime

dpath = 'prev-output'
doutpath = 'output'


def split_train_valid(data_filename):
    train_split = open(os.path.join(doutpath, 'train_split.csv'), 'wb')
    valid_split = open(os.path.join(doutpath, 'valid_split.csv'), 'wb')

    progress = 0
    with open(data_filename, 'rb') as ftrain:
        header = ftrain.readline()
        train_split.write(header)
        valid_split.write(header)
        for line in ftrain:
            timestamp = line.split(b',')[2]
            assert len(timestamp) == 8, timestamp
            assert timestamp[0:2] == b'14', timestamp
            assert timestamp[2:4] == b'10', timestamp
            if timestamp[4:6] == b'30':
                valid_split.write(line)
            else:
                train_split.write(line)

            progress += 1
            if progress%10000 == 0:
                print('\rProgress ', progress, end='')

    train_split.close()
    valid_split.close()


def count_values(filename, filename_save, with_id=False):
    count_dict = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    with open(filename, 'r') as f:
        header = f.readline().strip().split(',')
        for line in f:
            for index, value in enumerate(line.strip().split(',')):
                if with_id:
                    count_dict[header[index]][value] += 1
                elif index != 0:
                    count_dict[header[index]][value] += 1

    count_dict = dict(count_dict)
    for k, v in count_dict.items():
        count_dict[k] = dict(v)
    with open(filename_save, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(count_dict, f, pickle.HIGHEST_PROTOCOL)

    return count_dict


def count_dict_to_feature_dict(count_dict_name, feature_dict_name):
    with open(count_dict_name, 'rb') as f:
        count_dict = pickle.load(f)

    feature_dict = dict()
    for k, v in count_dict.items():
        v = sorted(v.items(), key=lambda x: x[1], reverse=True)
        tmp = dict()
        if v[-1][1] < 0:
            for i, (cat, count) in enumerate(v):
                if count >= 1000:
                    tmp[cat] = i + 1
                else:
                    tmp['others'] = 0
                    break
        else:
            for i, (cat, count) in enumerate(v):
                tmp[cat] = i
        feature_dict[k] = tmp

    with open(feature_dict_name, 'wb') as f:
        pickle.dump(feature_dict, f, pickle.HIGHEST_PROTOCOL)

    return feature_dict


def convert_origin_file(org_filename, new_file_name, feature_dict_name):
    with open(feature_dict_name, 'rb') as f:
        feature_dict = pickle.load(f)
    with open(org_filename, 'r') as forg, open(new_file_name, 'w') as fnew:
        tmp = forg.readline().strip().split(',')
        tmp.append('weekday')
        fnew.write(','.join(tmp) + '\n')
        for line in forg:
            line = line.strip().split(',')
            for i in range(len(line)):
                if i <= 1:
                    pass
                elif i == 2:
                    try:
                        time = datetime.strptime('20' + line[2], '%Y%m%d%H')
                    except:
                        print(line[2])
                        raise
                    line[2] = str(time.hour)
                    line.append(str(time.weekday()))
                else:
                    # print(line[i])
                    # print(tmp[i])
                    # print(feature_dict[tmp[i]])
                    line[i] = str(feature_dict[tmp[i]].get(line[i], 0))
            fnew.write(','.join(line) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split-train-valid', action="store_true")
    parser.add_argument('--count', action="store_true")
    parser.add_argument('--count-no-id', action="store_true")
    parser.add_argument('--data-set-preprocess', action="store_true")
    #args = parser.parse_args()

    print('split train valid...')
    split_train_valid('output/train.gz')
    print('count_values...')
    count_values(os.path.join(doutpath, 'train_split.csv'), os.path.join(doutpath, 'count_dict.pickle'))
    print('count_dict_to_feature_dict...')
    count_dict_to_feature_dict(os.path.join(doutpath, 'count_dict.pickle'), os.path.join(doutpath, 'feature_dict.pickle'))
    print('convert_origin_file')
    convert_origin_file(os.path.join(doutpath, 'train_split.csv'), os.path.join(doutpath, 'train_preprocessed.csv'),
                        os.path.join(doutpath, 'feature_dict.pickle'))
