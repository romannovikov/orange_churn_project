from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer

class PandasSimpleImputer(TransformerMixin):
    '''
    Класс, представляющий имплементацию sklearn.impute.SimpleImputer,
    которая возвращает pandas.DataFrame
    '''
    def __init__(self, *args, **kwargs):
        self.imputer = SimpleImputer(*args, **kwargs)

    def fit(self, X, y=None):
        self.imputer.fit(X)
        return self

    def transform(self, X, y=None):
        X[:] = self.imputer.transform(X)
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def set_params(self, **params):
        self.imputer.set_params(**params)
        return self