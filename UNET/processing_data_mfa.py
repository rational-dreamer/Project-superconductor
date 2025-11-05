from PIL import Image
import os
import csv
import pandas as pd
import json
import shutil

def load_config(config_file='config.json'):
    with open(config_file) as f:
        return json.load(f)


def resize_pics(input_dir, output_dir):
    '''
    Функция меняющая размеры изображения до 224х224 пикселей

    Аргументы:
            input_dir: путь к папке с исходными изображениями
            output_dir: путь к папке с измененными изображениями
    '''

    # Создаём папку для сохранения обработанных изображений
    os.makedirs(output_dir, exist_ok=True)

    # Проходим по всем файлам в указанной папке
    for filename in os.listdir(input_dir):

        # Получаем полный путь к изображению
        img_path = os.path.join(input_dir, filename)

        # Открываем изображение
        img = Image.open(img_path)

        # Изменяем размер изображения до 224x224
        img_resized = img.resize((224, 224))

        # Сохраняем изменённое изображение в папке назначения
        img_resized.save(os.path.join(output_dir, filename))

def csv_name(output_dir, file_name):
    '''
    Функция для получения csv файла с названиями изображений

    Аргументы:
            output_dir: путь к папке с измененными изображениями
    '''
    
    # Получаем список файлов в папке
    files = os.listdir(output_dir)

    # Записываем названия файлов в CSV
    with open(file_name, mode='w', newline='') as file:
         writer = csv.writer(file)
         for filename in files:
            writer.writerow([filename])

def csv_path(output_dir, file_path):
    '''
    Функция для получения csv файла с путями изображений

    Аргументы:
            output_dir: путь к папке с измененными изображениями
    '''

    # Получаем список всех файлов в папке с их полными путями
    file_paths = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]

    # Записываем пути в CSV
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for path in file_paths:
            writer.writerow([path])

def create_labels(file_name, file_path, labels_path):
    '''
    Функция для создания csv файла с разметкой

    Аргументы:
            file_name: csv файл с названиями изображений
            file_path: csv файл с путями изображений
    '''

    # Читаем csv-файл c названиями
    name = pd.read_csv(file_name, header=None, sep = '_')

    # Удаляем ненужные колонки
    name = name.drop(name.columns[[0, 5, 6, 7]], axis=1)

    # Присваиваем новые имена колонкам
    name.columns = ['D', 'V', 'tb', 'tp']

    # Удаляем буквы из всех строк в DataFrame
    name = name.apply(lambda x: x.str.replace(r'[A-Za-z]', '', regex=True))

    # Читаем csv-файл с путями
    path = pd.read_csv(file_path, header=None)

    # Приваиваем колонке новое имя
    path.columns = ['Path']

    # Объединяем два датасета
    dataset = pd.concat([path, name], axis=1)

    # Сохраняем полученный датафрейм
    dataset.to_csv(labels_path, index=False)

    # Возвращаем путь к файлу с разметкой
    return labels_path

def split_data(labels_path, train_data_path, val_data_path, test_data_path):
    '''
    Функция для разделения датасета на train, test и val

    Аргументы:
            labels_path: csv файл с разметкой
    '''

    # Загружаем CSV-файл 
    data = pd.read_csv(labels_path) 
 
    # Извлекаем строки без заголовка (первая строка — заголовки) 
    data_rows = data.iloc[1:]  # Берём строки с 1 по 419 
 
    # Перемешиваем строки для случайного распределения 
    data_rows = data_rows.sample(frac=1, random_state=42).reset_index(drop=True) 
 
    train_size = int(0.6 * len(data_rows))
    val_size = int(0.2 * len(data_rows)) 

    train_data = data_rows[:train_size]
    val_data = data_rows[train_size:train_size + val_size]   
    test_data = data_rows[train_size + val_size:]    
 
    # Сохраняем данные в два CSV-файла 
    train_data.to_csv(train_data_path, index=False, header=True) 
    val_data.to_csv(val_data_path, index=False, header=True)
    test_data.to_csv(test_data_path, index=False, header=True)

def main():

    config = load_config()
    
    # Указываем пути из json
    output_dir = config['output_dir']             # Путь к измененным изображениям
    learning_dir = config['learning_dir']          # Путь до папки для хранения тренировочных данных

    csv_files = learning_dir + '/csv_files'                   # Папка для хранения всех csv файлов
    file_name = learning_dir + '/name_files.csv'    # Файл с названиями изображений
    file_path = learning_dir + '/path_files.csv'    # Файл с путями до изображений

    train_data_path = learning_dir + '/train_data.csv'   # Файл с тренировочным датасетом
    val_data_path = learning_dir + '/val_data.csv'       # Файл с валидационным датасетом
    test_data_path = learning_dir + '/test_data.csv'     # Файл с тестовым датасетом
    labels_path = learning_dir + '/labels.csv'           # Файл с размеченными данными

    # Создаём тренировочную папку (перезапись, если существует)
    if (os.path.exists(learning_dir)):
        shutil.rmtree(learning_dir)
    os.makedirs(learning_dir, exist_ok=True)
    os.makedirs(csv_files, exist_ok=True)

    print(f'Начало работы с данными\n')
    
    # Определяем какую директорию использовать для обработки
    if config['resize_images']:
        input_dir = config['input_dir']               # Путь к исходным изображениям
        print('Изменение размеров изображений')
        resize_pics(input_dir, output_dir)
        print(f'Измененные изображения находятся в директории: {output_dir}')
    else:
        print('Пропуск изменения размеров изображений (resize_images=false)')
        print(f'Используются исходные изображения из директории: {output_dir}')        

    csv_name(output_dir, file_name)
    print('Создан csv файл с названиями изображений')
    print()

    csv_path(output_dir, file_path)
    print('Создан csv файл с путями до изображений')
    print()

    labels_path = create_labels(file_name, file_path, labels_path)
    print(f'Размеченные данные находятся в директории: {labels_path}')
    print()

    split_data(labels_path, train_data_path, val_data_path, test_data_path)
    print(f'Датасет разделен на 3 файла: train_data, val_data, test_data\n'
          f'Файлы находятся в директории: {csv_files}')      
      
if __name__ == "__main__":
    main()