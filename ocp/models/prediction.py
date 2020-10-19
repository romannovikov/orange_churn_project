import os
import sys
import warnings; warnings.filterwarnings(action='ignore')

# добавляем в sys.path директорию со скриптами
src_dir = os.path.join(os.getcwd(), '..')
sys.path.append(src_dir)

# загружаем необходимые скрипты
from data.loading import load_data, load_obj


def make_prediction(models_dir, to_predict):
    fitted_models = [load_obj(os.path.join(models_dir, f'{model}_fitted.model'))
                     for model in ('XGBClassifier', 'CatBoostClassifier')]
    xgb_probs, catboost_probs = [model.predict_proba(to_predict) for model in fitted_models]
    alpha = .6
    stacking_probs = alpha * xgb_probs + (1 - alpha) * catboost_probs

    return stacking_probs


if __name__ == '__main__':
    _, test, numerical, categorical = load_data('../../data/processed')

    print(make_prediction(models_dir='../../models/',
                          to_predict=test))
