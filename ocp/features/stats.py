import numpy as np
import pandas as pd
import scipy.stats as ss
import statsmodels.stats.api as sms

import collections


# функции для вычисления различного вида энтропии
def calculate_entropy(x):
    """
    Функция для вычисления информационнной энтропии H(X)
    (https://en.wikipedia.org/wiki/Entropy_(information_theory))
    """
    x_counter = collections.Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n / total_occurrences, x_counter.values()))
    entropy = ss.entropy(p_x)
    return entropy


def calculate_conditional_entropy(x, y):
    """
    Функция для вычисления условной энтропии H(X|Y)
    (https://en.wikipedia.org/wiki/Conditional_entropy)
    """
    y_counter = collections.Counter(y)
    x_y_counter = collections.Counter(list(zip(x, y)))
    total_occurrences = sum(y_counter.values())
    conditional_entropy = 0.0
    for xy in x_y_counter.keys():
        p_x_y = x_y_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        conditional_entropy += p_x_y * np.log(p_y / p_x_y)
    return conditional_entropy


def calculate_joint_entropy(x, y):
    """
    Функция для вычисления взаимной энтропии H(X,Y)
    (https://en.wikipedia.org/wiki/Joint_entropy)
    """
    # H(X, Y) = H(Y) + H(X|Y)
    return calculate_entropy(y) + calculate_conditional_entropy(x, y)


# функции для вычисления коэффициентов взаимосвязи
def cramers_v(x, y):
    """
    Функция для вычисления скорректированного коэффициента V Крамера,
    (источник - https://en.wikipedia.org/wiki/Cramér's_V)
    характеризующего наличие взаимосвязи между двумя категориальными переменными
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1)**2) / (n - 1)
    kcorr = k - ((k - 1)**2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def asymmetrical_theilsu(x, y):
    """
    Функция для вычисления коэффициента Тейла U(X|Y) или коэффициента
    неопределенности (источник - https://en.wikipedia.org/wiki/Uncertainty_coefficient),
    являющегося асимметричной мерой наличия взаимосвязи между двумя
    категориальными переменными, которая измеряет долю неопределенности (энтропии)
    в переменной X, которая объясняется переменной Y.
    """
    # H(X|Y)
    H_x_y = calculate_conditional_entropy(x, y)
    # H(X)
    H_x = calculate_entropy(x)
    if H_x == 0:
        return 1
    else:
        # (H(X) - H(X|Y)) / H(X)
        return (H_x - H_x_y) / H_x


def symmetrical_theilsu(x, y):
    """
    Функция для вычисления симметричной версии коэффициента Тейла U(X,Y) или
    симметричной версии коэффицента неопределенности (источник -
    https://en.wikipedia.org/wiki/Uncertainty_coefficient), которая определяется
    как средневзвешенное значение между U(X|Y) и U(Y|X)
    """
    # H(X), H(Y)
    H_x, H_y = calculate_entropy(x), calculate_entropy(y)
    # U(X|Y), U(Y|X)
    U_xy, U_yx = asymmetrical_theilsu(x, y), asymmetrical_theilsu(y, x)
    if (H_x + H_y) == 0:
        return 1
    else:
        # U(X,Y) = (H(X)U(X|Y) + H(Y)U(Y|X)) / (H_x + H_y)
        return (H_x*U_xy + H_y*U_yx) / (H_x + H_y)


# функции для создания корреляционных матриц
def asymmetrical_corrmat(categorical, coef='theilsu'):
    """
    Функция для вычисления матрицы, в которой элемент Xij
    является значением коэффициента взаимосвязи для i-го
    и j-го категориальных признаков
    """
    features = categorical.columns
    corrmat = pd.DataFrame(index=features, columns=features)
    for i in range(len(features)):
        for j in range(len(features)):
            if i == j: 
                corrmat.iloc[i, j] = 1.0; continue
            else:
                i_feature, j_feature = features[i], features[j]
                if coef == 'theilsu':
                    cell = asymmetrical_theilsu(categorical[i_feature],
                                                categorical[j_feature])
                else:
                    cell = None
                corrmat.iloc[i, j] = cell
    return corrmat.astype('float32') 


def symmetrical_corrmat(categorical, coef='cramersv'):
    """
    Функция для вычисления матрицы, в которой элементы Xij и Xji
    являются значением коэффициентов взаимосвязи для i-го и j-го
    категориальных признаков
    """
    features = categorical.columns
    corrmat = pd.DataFrame(index=features, columns=features)
    for i in range(len(features)):
        for j in range(i, len(features)):
            if i == j: 
                corrmat.iloc[i, j] = 1.0; continue
            else:
                i_feature, j_feature = features[i], features[j]
                if coef == 'cramersv':
                    cell = cramers_v(categorical[i_feature], 
                                     categorical[j_feature])
                elif coef == 'theilsu':
                    cell = symmetrical_theilsu(categorical[i_feature], 
                                               categorical[j_feature])
                else:
                    cell = None
                corrmat.iloc[i, j], corrmat.iloc[j, i] = cell, cell
    return corrmat.astype('float32')


def get_redundant_pairs(corrmat):
    """Get diagonal and lower triangular pairs of correlation matrix"""
    pairs_to_drop = set()
    columns = corrmat.columns
    for i in range(corrmat.shape[1]):
        for j in range(i+1):
            pairs_to_drop.add((columns[i], columns[j]))
    return pairs_to_drop


def get_top_abs_correlations(corrmat, threshold=.95):
    au_corr = corrmat.abs().unstack()
    labels_to_drop = get_redundant_pairs(corrmat)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[au_corr >= threshold]


def tconfint_mean(scores):
    return np.round(sms.DescrStatsW(scores).tconfint_mean(), 4)
