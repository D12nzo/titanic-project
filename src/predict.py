import pandas as pd
import joblib
import os

def make_prediction():
    # 1. Загрузка пайплайна (модель + препроцессинг)
    model_path = 'models/titanic_pipeline.pkl'
    if not os.path.exists(model_path):
        print("Ошибка: Сначала запусти train.py, чтобы создать модель!")
        return
        
    model = joblib.load(model_path)
    
    # 2. Загрузка тестовых данных
    test_df = pd.read_csv('data/test.csv')
    
    # 3. Предсказание 
    predictions = model.predict(test_df)
    
    # 4. Сохранение результата в формате Kaggle
    output = pd.DataFrame({
        'PassengerId': test_df.PassengerId, 
        'Survived': predictions
    })
    
    output.to_csv('results/submission.csv', sep = ';', index=False)
    print("Success: Предсказания сохранены в results/submission.csv")

if __name__ == "__main__":
    make_prediction()