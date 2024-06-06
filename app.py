import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier

# Загрузка предобученной модели
model = CatBoostClassifier()
model.load_model('weight.pkl')

# Заголовок страницы
st.title('Приложение для предсказания с использованием модели машинного обучения')

# Функция для отображения справки
def show_help():
    st.write("""
    ## Справка
    Это приложение позволяет загрузить файл CSV и использовать предобученную модель машинного обучения для предсказания на основе данных из этого файла.

    ### Функции:
    1. Загрузка файла CSV: Нажмите на кнопку 'Загрузите файл CSV' и выберите файл с данными.
    2. Отображение загруженных данных: После загрузки файла, данные из него будут отображены на странице.
    3. Предсказание: Приложение использует предобученную модель машинного обучения для выполнения предсказаний на основе загруженных данных.
    4. Отображение предсказаний: Результаты предсказаний будут отображены на странице.
    """)

# Добавление кнопки /help
st.sidebar.markdown("## Справка")
if st.sidebar.button("/help"):
    show_help()

# Загрузка файла CSV
uploaded_file = st.file_uploader('Загрузите файл CSV', type='csv')

# Ввод данных через форму
with st.form("input_form"):
    session_month_year1 = st.text_input("session_month_year1")
    session_duration = st.number_input("session_duration")
    is_weekend = st.checkbox("is_weekend")
    site10 = st.number_input("site10")
    site9 = st.number_input("site9")
    site3 = st.number_input("site3")
    site7 = st.number_input("site7")
    site5 = st.number_input("site5")
    site1 = st.number_input("site1")
    site4 = st.number_input("site4")
    site2 = st.number_input("site2")
    site6 = st.number_input("site6")
    site8 = st.number_input("site8")
    session_id = st.text_input("session_id")
    submitted = st.form_submit_button("Submit")

if uploaded_file is not None:
    # Чтение данных из файла CSV
    data = pd.read_csv(uploaded_file)

    # Отображение загруженных данных
    st.write('Загруженные данные:')
    st.write(data)

    # Предсказание с использованием модели
    predictions = model.predict(data)

    # Отображение предсказаний
    st.write('Предсказания:')
    st.write(predictions)

elif submitted:
    # Создание DataFrame из введенных данных
    data = pd.DataFrame({
        'session_month_year1': [session_month_year1],
        'session_duration': [session_duration],
        'is_weekend': [is_weekend],
        'site10': [site10],
        'site9': [site9],
        'site3': [site3],
        'site7': [site7],
        'site5': [site5],
        'site1': [site1],
        'site4': [site4],
        'site2': [site2],
        'site6': [site6],
        'site8': [site8],
        'session_id': [session_id]
    })

    # Отображение введенных данных
    st.write('Введенные данные:')
    st.write(data)

    # Предсказание с использованием модели
    predictions = model.predict(data)

    # Отображение предсказаний
    st.write('Предсказания:')
    st.write(predictions)
