# Titanic Survival Analysis

Мой первый проект в Machine Learning. На базе данных Kaggle я предсказал выживаемость пассажиров Титаника, пройдя путь от очистки данных до создания автоматизированного ML-пайплайна.

**Dataset:** [Kaggle Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)

---

## Что сделано

### 1. EDA (Exploratory Data Analysis)
- Анализ пропусков и обработка отсутствующих значений
- Визуализация распределений признаков
- Корреляция признаков (Heatmap)

### 2. Feature Engineering
- `Title` — социальный статус пассажира  
- `FamilySize` — размер семьи  
- `IsAlone` — признак, путешествует ли пассажир один

### 3. Preprocessing Pipeline
Использован `ColumnTransformer` для:
- заполнения пропусков (`SimpleImputer`)
- масштабирования числовых признаков (`StandardScaler`)
- кодирования категориальных признаков (`OneHotEncoder`)

### 4. Моделирование
- Random Forest Classifier
- Accuracy на валидации: ~82%
- Метрика ROC-AUC оценена
- Подбор гиперпараметров через GridSearchCV

---

## Главный вывод
Наибольшее влияние на выживаемость оказали:
- Пол пассажира
- Социальный статус (`Title`)
- Класс каюты

---

## Структура проекта


TITANIC-PROJECT/
├── data/ — исходные CSV файлы (train.csv, test.csv)
├── notebooks/ — Jupyter Notebook с EDA и визуализациями
├── src/ — модульный код проекта
│ ├── preprocessing.py — обработка данных и сборка Pipeline
│ ├── train.py — обучение и сохранение модели
│ └── predict.py — генерация предсказаний
├── models/ — сохраненная модель пайплайна (.pkl)
├── results/ — итоговые предсказания (submission.csv)
├── images/ — графики, матрицы ошибок и визуализации признаков
├── README.md
└── requirements.txt


---

## Запуск проекта

1. Установка зависимостей:

```bash
pip install -r requirements.txt
Обучение модели (создаст пайплайн в models/):
python src/train.py
Получение предсказаний (генерация submission.csv в results/):
python src/predict.py

Также вы можете изучить пошаговый процесс в ноутбуке в папке notebooks/.

Визуализации

В папке images/ доступны:

Матрица ошибок (confusion_matrix.png)
Корреляционная тепловая карта признаков (correlation_heatmap.png)
Важность признаков (feature_importance.png)
Пропуски в данных (missing_data.png)

Автор: Danil (D12nzo)