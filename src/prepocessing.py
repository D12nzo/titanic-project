import pandas as pd
import numpy as np

def clean_data(df):
    """
    Функция для базовой очистки данных Титаника.
    Заполняет пропуски и создает новые признаки.
    """
    # 1. Заполнение пропусков
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # 2. Создание признаков (Feature Engineering)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = np.where(df['FamilySize'] == 1, 1, 0)
    
    # 3. Извлечение титулов
    titles = r'(?:,\s|\s+)(Mr\.|Mrs\.|Miss\.|Master\.|Dr\.|Rev\.|Col\.|Major\.|Lady\.|Sir\.|Capt\.)(?:\s|$)'
    df['Title'] = df['Name'].str.extract(titles)
    
    # Группируем редкие титулы
    rare_titles = df['Title'].value_counts()[df['Title'].value_counts() < 10].index
    df.loc[df['Title'].isin(rare_titles), 'Title'] = 'Rare'
    
    # 4. Удаление лишних колонок
    cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    return df

def encode_features(df):
    """
    Преобразование категориальных признаков в числа.
    """
    df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Pclass', 'Title'], dtype=int)
    return df