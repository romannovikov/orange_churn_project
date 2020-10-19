import os
import pickle

import numpy as np
import pandas as pd


def load_raw_data(data_dir='.'):
    # считываем обучающую выборку, метки целевого признака,
    # а также данные, для которых необходимо будет сделать предсказание
    train_path = os.path.join(data_dir, 'orange_small_churn_data.csv')
    test_path = os.path.join(data_dir, 'orange_small_churn_test_data.csv.zip')
    labels_path = os.path.join(data_dir, 'orange_small_churn_labels.csv')
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path, index_col=0)
    labels = pd.read_csv(labels_path, names=['target'])

    # из исходных данных нам известно какие признаки
    # являются числовыми, а какие категориальными
    # выделим их явно в списки
    numerical = train.columns.to_list()[:190]
    categorical = train.columns.to_list()[190:]

    # объединим метки классов с данными обучения
    train = pd.concat([train, labels], axis=1)

    return train, test, numerical, categorical


def load_data(data_dir='.'):
    # считываем обучающую выборку, метки целевого признака,
    # а также данные, для которых нужно будет сделать предсказание
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    
    train = pd.read_csv(train_path, index_col=0)
    test = pd.read_csv(test_path, index_col=0)

    # загрузим списки числовых и категориальных переменных
    numerical = np.load(os.path.join(data_dir, 'numerical.npy'))
    categorical = np.load(os.path.join(data_dir, 'categorical.npy'))

    return train, test, numerical, categorical


def load_obj(file):
    with open(file, 'rb') as pickle_file:
        obj = pickle.load(pickle_file)
        return obj
