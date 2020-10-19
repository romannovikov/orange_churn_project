import os
import sys
import warnings; warnings.filterwarnings(action='ignore')

# добавляем в sys.path директорию со скриптами
src_dir = os.path.join(os.getcwd(), '..')
sys.path.append(src_dir)

# загружаем необходимые скрипты
from data.loading import load_data, load_obj
from data.saving import  save_obj
from data.splitting import train_holdout_split

SEED = 26


def train_models(data_dir):
    train, _, numerical, categorical = load_data(data_dir)

    categorical_idxs = [train.columns.get_loc(c) for c in categorical]

    (X, _, _,
     y, _, _,
     _) = train_holdout_split(train, random_state=SEED)

    models = load_obj('../../data/models_dictionary/models_final.pkl')

    base_models = list(models.keys())
    base_models.remove('stacking')

    for model in base_models:
        pipe = models[model]['pipe']

        if model == 'CatBoostClassifier':
            # передаем в параметры обучения индексы категориальных переменных
            fit_params = {'model__cat_features': categorical_idxs,
                          'model__verbose': False}
            pipe.fit(X, y, **fit_params)
        else:
            pipe.fit(X, y)

        save_obj(pipe, f'../../models/{model}_fitted.model')


if __name__ == '__main__':
    train_models('../../data/processed')
