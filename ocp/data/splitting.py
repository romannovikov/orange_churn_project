import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, PredefinedSplit


def train_holdout_split(train, random_state=None):
    # разделим матрицу признаков и целевую переменную из данных обучения
    X, y = train.drop('target', axis=1), train.target
    
    # с помощью конкатенации создадим признак,
    # по которому будем в дальнейшем делить наши данные
    stratify = (train['Var134'].isna().astype(str) 
                + '_' 
                + train['target'].astype(str))
    
    # отложим часть выборки для финальной оценки модели
    X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, 
                                                              test_size=0.33, 
                                                              stratify=stratify, 
                                                              random_state=random_state)
    
    # т.к. в дальнейшем мы будем делать стратификацию внутри цикла
    # кросс-валидации на данных из X_train, то необходимо скорректировать
    # признак для стратификации по индексу из X_train 
    stratify = stratify[X_train.index]
    
    return (X, X_train, X_holdout, 
            y, y_train, y_holdout, 
            stratify)


def predefined_cv_strategy(X, stratify, n_splits=5, random_state=None):
    # задаем стратегию кросс-валидации
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    test_folds = pd.Series(index=X.index, dtype='int8')
    
    i = 0
    for _, test_index in skf.split(X, stratify):
        test_folds.iloc[test_index] = i
        i += 1
    
    return PredefinedSplit(test_folds)
