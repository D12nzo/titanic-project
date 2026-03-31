import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class TitanicFeatureExtractor(BaseEstimator, TransformerMixin):
    """Извлекает Title, FamilySize и IsAlone"""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # 1. FamilySize & IsAlone
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
        X['IsAlone'] = (X['FamilySize'] == 1).astype(int)
        
        # 2. Title extraction
        titles = r'(Mr\.|Mrs\.|Miss\.|Master\.|Dr\.|Rev\.|Col\.|Major\.|Lady\.|Sir\.|Capt\.)'
        X['Title'] = X['Name'].str.extract(titles, expand=False)
        X['Title'] = X['Title'].fillna('Rare')
        
        # Группируем редкие 
        main_titles = ['Mr.', 'Mrs.', 'Miss.', 'Master.']
        X.loc[~X['Title'].isin(main_titles), 'Title'] = 'Rare'
        
        return X[['Age', 'Fare', 'FamilySize', 'IsAlone', 'Sex', 'Embarked', 'Pclass', 'Title']]

def get_preprocessor():
    """Собирает финальный ColumnTransformer"""
    
    # Колонки, которые пойдут в числовую обработку
    num_cols = ['Age', 'Fare', 'FamilySize', 'IsAlone']
    # Колонки, которые пойдут в категориальную обработку
    cat_cols = ['Sex', 'Embarked', 'Pclass', 'Title']

    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Главный препроцессор
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

    # Собираем всё в цепочку
    return Pipeline(steps=[
        ('extractor', TitanicFeatureExtractor()),
        ('preprocessor', preprocessor)
    ])