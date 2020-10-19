import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class UsefullFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, nans_thr=1, const_thr=1):
        self.nans_thr = nans_thr
        self.const_thr = const_thr
        self.useless_ = None

    def fit(self, X, y=None):
        self.useless_ = np.unique(self._nans_detect(X) +
                                  self._const_detect(X) +
                                  self._id_detect(X))
        print('Columns to drop:', len(self.useless_))
        return self

    def transform(self, X):
        return X.drop(self.useless_, axis=1)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X).transform(X)

    def _nans_detect(self, X):
        threshold = self.nans_thr

        # получаем сведения о пропусках в данных
        nans = X.isna().mean()

        # получаем список признаков, в которых
        # процент пропусков выше порога
        nans_features = nans[nans >= threshold].index.to_list()
        print('NaNs features were found:', len(nans_features))
        return nans_features

    def _const_detect(self, X):
        threshold = self.const_thr

        # создаем пустой список
        constant_features = []

        # для каждого признака
        for feature in X:
          
            # определяем самое часто 
            # встречающееся значение в признаке
            try:
                dominant = (X[feature]
                            .value_counts(normalize=True)
                            .sort_values(ascending=False)
                            .values[0])
            # если признак полностью состоит из пропусков, то пропускаем его
            except IndexError:
                continue

            # если доля такого значения превышает заданный порог
            # тогда добавляем признак в список
            if dominant >= threshold:
                constant_features.append(feature)
                
        print('Constant features were found:', len(constant_features))
        return constant_features

    @staticmethod
    def _id_detect(X):
        id_features = []
        for feature in X:
            rows = X[feature].dropna().shape[0]
            nunique = X[feature].dropna().nunique()
            if rows & (rows == nunique):
                id_features.append(feature)
        print('ID features were found:', len(id_features))
        return id_features


def correct_features_lists(all_features, numerical, categorical):
    numerical = np.intersect1d(numerical, all_features).tolist()
    categorical = np.intersect1d(categorical, all_features).tolist()
    print('After correction: numerical -', len(numerical), ', categorical: - ', len(categorical), '\n')
    return numerical, categorical
