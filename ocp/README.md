Структура директории
------------

     ocp             
     ├── __init__.py 
     ├── plotting.py <- Код, относящийся к работе с визуализациями
     ├── server.py   <- Скрипт для запуска веб-интерфейса на Flask
     │
     ├── data        <- Код, относящийся к работе с данными
     │   ├── loading.py
     │   ├── saving.py
     │   └── splitting.py
     │
     ├── features    <- Код, относящийся к работе с признаками
     │   ├── feature_engineering.py
     │   ├── feature_selection.py
     │   └── stats.py
     │
     └── models      <- Код, относящийся к работе с моделями
         ├── preprocessing.py
         ├── training.py
         └── prediction.py
