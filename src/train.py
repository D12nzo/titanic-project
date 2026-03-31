import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from preprocessing import get_preprocessor

def train_model():
    # Загружаем данные
    df = pd.read_csv('data/train.csv')
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    # Собираем Pipeline (Препроцессинг + Модель)
    # Это гарантирует, что к новым данным применятся те же правила
    full_pipeline = Pipeline([
        ("preprocessor", get_preprocessor()),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    full_pipeline.fit(X, y)
    
    # Сохраняем весь пайплайн целиком!
    joblib.dump(full_pipeline, 'models/titanic_pipeline.pkl')
    print("Success: Pipeline saved to models/titanic_pipeline.pkl")

if __name__ == "__main__":
    train_model()