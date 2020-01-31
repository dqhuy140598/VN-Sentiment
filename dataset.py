from underthesea import word_tokenize
import os
from os.path import abspath
import copy
import re
import pandas as pd
from utils.utils import normalize_text


def split_array(arr, condition):
    if len(arr) == 0:
        return []
    result = []
    accumulated = [arr[0]]
    for ele in arr[1:]:
        if condition(ele):
            result.append(copy.deepcopy(accumulated))
            accumulated = [copy.deepcopy(ele)]
        else:
            accumulated.append(copy.deepcopy(ele))
    result.append(copy.deepcopy(accumulated))
    return result


def read_file(file_path, is_train=True):
    file_path = abspath(file_path)
    data_lines = list(
        filter(lambda x: x != '', open(file_path).read().split('\n')))
    pattern = ('train' if is_train else 'test') + '_[0-9]{5}'
    datas = split_array(data_lines, lambda x: bool(re.match(pattern, x)))
    if is_train:
        result_array = list(map(
            lambda x: [x[0], ' '.join(x[1:-1]), int(x[-1])], datas))
    else:
        result_array = list(map(lambda x: [x[0], ' '.join(x[1:])], datas))
    columns = ['name', 'text', 'label'] if is_train else ['name', 'text']
    return pd.DataFrame(result_array, columns=columns)


def create_processed_dataset(outfile_path, file_path, is_train=True):
    df = read_file(file_path, is_train)
    df.to_csv(outfile_path, sep=',')


if __name__ == '__main__':
    train_path = 'data/train.crash'
    test_path = 'data/test.crash'
    out_train = 'data/processed/train.csv'
    out_test = 'data/processed/test.csv'
    create_processed_dataset(out_train,file_path=train_path,is_train=True)
    create_processed_dataset(out_test,file_path=test_path,is_train=False)
