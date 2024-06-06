from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from pymongo import MongoClient
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Параметры DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'mongodb_to_randomfores1t',
    default_args=default_args,
    description='A simple DAG to fetch data from MongoDB, preprocess it, and build a RandomForest model',
    schedule_interval=timedelta(days=7),
)

# Функция для сбора данных из MongoDB
def fetch_data_from_mongodb():
    client = MongoClient('mongodb://localhost:27017/')
    db = client['fyge']
    collection = db['tskv']
    data = list(collection.find())
    df = pd.DataFrame(data)
    df.to_csv('/tmp/mongodb_data.csv', index=False)

# Функция для предобработки данных
def preprocess_data():
    df = pd.read_csv('/tmp/mongodb_data.csv')
    # Пример предобработки данных
    df = df.dropna()
    df.to_csv('/tmp/preprocessed_data.csv', index=False)

def number_encode_features(init_df: pd.DataFrame) -> (pd.DataFrame, dict):
    """
    Eng: Function for encoding categorical features into numerical ones.
    Fra: Fonction pour encoder les caractéristiques catégorielles en numériques.
    Rus: Функция для кодирования категориальных признаков в числовые.
    Ger: Funktion zum Codieren kategorischer Merkmale in numerische.

    Parameters:
    init_df (pd.Data Frame): The original DataFrame.

    Returns:
    result (pd.Data Frame): A DataFrame with encoded attributes.
    encoders (dict): A dictionary with a LabelEncoder for each categorical feature.
    """
    result = init_df.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == object:
            encoders[column] = preprocessing.LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column])
    return result, encoders
    
   
# Функция для построения и оценки модели RandomForest
def build_and_evaluate_model():
    df = pd.read_csv('/tmp/preprocessed_data.csv')
    df,govno = number_encode_features(df)
    X = df.drop('name_ru', axis=1)
    y = df['name_ru']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

# Операторы для DAG
fetch_data_task = PythonOperator(
    task_id='fetch_data_from_mongodb',
    python_callable=fetch_data_from_mongodb,
    dag=dag,
)

preprocess_data_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

build_and_evaluate_model_task = PythonOperator(
    task_id='build_and_evaluate_model',
    python_callable=build_and_evaluate_model,
    dag=dag,
)

# Задание зависимостей
fetch_data_task >> preprocess_data_task >> build_and_evaluate_model_task
